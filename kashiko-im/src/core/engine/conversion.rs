//! Conversion state handling (candidates, commit). The live-conversion
//! chunking lives in the sibling `chunk` module.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use kashiko_engine::KanaKanjiConverter;

use tracing::debug;

use super::*;

/// Maximum number of learning candidates to show
const MAX_LEARNING_CANDIDATES: usize = 3;

/// One unit of deep-search work — at most a couple of model calls, so the
/// server can serve key events between tasks instead of stalling behind the
/// whole search.
pub(super) enum DeepTask {
    /// Convert `variant` with a `beam`-wide pass; keep at most `cap` results.
    Convert {
        variant: String,
        beam: usize,
        cap: usize,
    },
    /// Convert the tail segment with the prefix conversion as left context,
    /// producing full-sentence tail alternatives.
    TailBeam {
        prefix_reading: String,
        tail_reading: String,
        beam: usize,
    },
}

/// A deep search ready to run, carrying everything the work needs.
///
/// Self-contained on purpose: this runs on a worker thread, so it holds its own
/// converter handle and its own copy of the context rather than reaching back
/// into the engine. Nothing here touches engine state, which is what lets the
/// main loop keep serving keys while a wide beam is grinding away.
pub struct DeepSearchJob {
    /// The request the job was started for; the outcome is dropped when the
    /// buffer or conversion scope no longer matches (committed / re-edited /
    /// resized).
    request: DeepSearchRequest,
    /// The scoped reading with stray ASCII stripped (used as the recorded
    /// reading for combined tail candidates).
    reading: String,
    queue: VecDeque<DeepTask>,
    found: Vec<Candidate>,
    seen: HashSet<String>,
    max_results: usize,
    /// Conversion context as of scheduling time.
    ctx: String,
    /// Character budget for the context handed to the model, so the worker can
    /// build a tail context the same way the engine would.
    max_ctx: usize,
    /// Chunk-cached conversion of the tail task's prefix, when the live
    /// conversion already produced one; saves the worker a model call.
    cached_prefix: Option<String>,
    /// Best score the quick pass recorded, the bar a finding has to clear.
    quality_best: Option<f32>,
    converter: Arc<KanaKanjiConverter>,
    /// Set by the engine when the conversion this was started for is gone, so a
    /// long job stops burning CPU on a result nobody will see.
    cancel: Arc<AtomicBool>,
}

/// What a finished deep search hands back to the engine.
pub struct DeepSearchOutcome {
    request: DeepSearchRequest,
    /// The scoped reading with stray ASCII stripped, as the job used it.
    reading: String,
    found: Vec<Candidate>,
}

/// Run a deep search to completion. Called on the worker thread; touches no
/// engine state.
pub fn run_deep_job(mut job: DeepSearchJob) -> DeepSearchOutcome {
    while let Some(task) = job.queue.pop_front() {
        if job.cancel.load(Ordering::Relaxed) || job.found.len() >= job.max_results {
            break;
        }
        run_deep_task(task, &mut job);
    }
    let found = if job.cancel.load(Ordering::Relaxed) {
        Vec::new()
    } else {
        gate_deep_findings(
            &job.converter,
            &job.reading,
            &job.ctx,
            job.request.kind,
            job.quality_best,
            std::mem::take(&mut job.found),
        )
    };
    DeepSearchOutcome {
        request: job.request,
        reading: job.reading,
        found,
    }
}

/// Convert `reading` with the job's converter, or nothing when it isn't kana.
fn job_convert(job: &DeepSearchJob, reading: &str, ctx: &str, n: usize) -> Vec<String> {
    if !kashiko_engine::contains_kana(reading) {
        return vec![];
    }
    let katakana = kashiko_engine::hiragana_to_katakana(reading);
    job.converter.convert(&katakana, ctx, n).unwrap_or_default()
}

/// Execute one task, appending clean findings to the job.
fn run_deep_task(task: DeepTask, job: &mut DeepSearchJob) {
    fn accept(job: &mut DeepSearchJob, out: String, variant: &str) -> bool {
        if out != variant
            && !kashiko_engine::has_long_unconverted_run(&out, variant, UNCONVERTED_RUN_MIN)
            && !kashiko_engine::has_unrenderable_char(&out)
            && job.seen.insert(out.clone())
        {
            job.found.push(Candidate {
                text: out,
                reading: Some(variant.to_string()),
                source_label: Some(CandidateSource::Model.label().to_string()),
                description: None,
            });
            true
        } else {
            false
        }
    }

    match task {
        DeepTask::Convert { variant, beam, cap } => {
            let mut added = 0usize;
            for out in job_convert(job, &variant, &job.ctx.clone(), beam) {
                if added >= cap || job.found.len() >= job.max_results {
                    break;
                }
                if accept(job, out, &variant) {
                    added += 1;
                }
            }
        }
        DeepTask::TailBeam {
            prefix_reading,
            tail_reading,
            beam,
        } => {
            let ctx = job.ctx.clone();
            let prefix_conv = match job.cached_prefix.clone() {
                Some(p) => Some(p),
                None => job_convert(job, &prefix_reading, &ctx, 1).into_iter().next(),
            };
            let Some(prefix_conv) = prefix_conv else {
                return;
            };
            let reading = job.reading.clone();
            let combined = format!("{ctx}{prefix_conv}");
            let tail_ctx: String = {
                let n = combined.chars().count();
                if n > job.max_ctx {
                    combined.chars().skip(n - job.max_ctx).collect()
                } else {
                    combined
                }
            };
            for tail_out in job_convert(job, &tail_reading, &tail_ctx, beam) {
                if job.found.len() >= job.max_results {
                    break;
                }
                accept(job, format!("{prefix_conv}{tail_out}"), &reading);
            }
        }
    }
}

/// Dictionary-form ending implied by a 連用形 ending.
///
/// A godan verb's 連用形 sits in the い-段 of its row, and its dictionary form
/// in the う-段 of the same row: `のり`→`のる`, `かき`→`かく`, `のみ`→`のむ`.
/// Anything else (ichidan `たべます`, and non-verbs) takes `る`, which is the
/// ichidan dictionary form and simply finds nothing when the guess is wrong.
fn dictionary_form_ending(renyoukei: char) -> Option<char> {
    Some(match renyoukei {
        'い' => 'う',
        'き' => 'く',
        'ぎ' => 'ぐ',
        'し' => 'す',
        'ち' => 'つ',
        'に' => 'ぬ',
        'び' => 'ぶ',
        'み' => 'む',
        'り' => 'る',
        // `ます` straight after the stem is ichidan (`たべ|ます`).
        'ま' => 'る',
        _ => return None,
    })
}

/// A candidate submitted to the quality gate: where it sits in the list, how
/// natural it reads (per-character NLL, lower is better; `None` = unscoreable),
/// and whether it is protected from removal.
#[derive(Debug, Clone, Copy)]
pub(super) struct Scored {
    pub index: usize,
    pub score: Option<f32>,
    /// Largest single-character NLL in the candidate. `None` = unscoreable.
    pub worst: Option<f32>,
    /// Never dropped, only reordered. Set for the user's own learned choices:
    /// re-ranking them is fair, discarding them is not.
    pub never_drop: bool,
}

/// NLL discount applied to a learned candidate before ranking.
///
/// A past choice is strong evidence, but it is recorded against the reading
/// alone and so carries no context — `のります` learned as `乗ります` from
/// `車に乗ります` would otherwise keep winning under `雑誌に`. The discount lets
/// the past choice take ties and near-ties while a clear contextual preference
/// still overrules it.
pub(super) const LEARNED_SCORE_BONUS: f32 = 0.15;

/// How much worse than the best candidate a candidate may score and still be
/// shown, in per-character NLL. Relative rather than absolute so it adapts to
/// how hard the sentence is: what matters is whether a candidate is close to
/// the best reading, not its raw likelihood.
const QUALITY_MARGIN_NLL: f32 = 0.55;

/// Absolute per-character NLL ceiling. Catches the case where even the best
/// candidate is poor, so "within the margin of the best" would still let
/// nonsense through.
const QUALITY_CEILING_NLL: f32 = 1.6;

/// Ceiling on the single worst character of a candidate.
///
/// The per-character average cannot judge a long sentence: one wrong character
/// in twenty-six barely moves it, so `…形で対欧できます` (average 0.62) sat
/// within the margin of `…形で対応できます` (0.03) and reached the window. Its
/// worst character scores 13.8 — the damage is local, and only a local measure
/// sees it.
///
/// Measured across the cases this gate exists for, legitimate candidates top
/// out around 5 (`皆さんおはよう御座います` 5.0, `きょうの天気は晴れです` 4.4)
/// while nonsense starts at 6 and climbs steeply (`官寺` 6.9, `かけている分廠`
/// 9.2, `皆さんお様ございます` 11.6, `伸ります` 15.6). Set above the highest
/// real candidate seen rather than as tight as the numbers allow, so a rare but
/// wanted reading is not cut for being unusual.
const QUALITY_WORST_TOKEN_NLL: f32 = 6.0;

/// Candidates always kept, however badly they score. The list must never be
/// gutted — the user needs something to pick even when the model is lost.
pub(super) const MIN_KEPT_SCORED: usize = 2;

/// Characters that never come out of a kana reading on their own: brackets and
/// decorative marks. A dictionary entry like `田中☆` or `任意(コピー)` is a
/// nickname or a gloss that has no business in a conversion list.
///
/// Deliberately a list rather than a character-class test, so the punctuation
/// that *is* legitimate in Japanese text (`ー`, `々`, `・`, `…`, `、`, `。`) is
/// untouched.
const NEVER_INVENTED: &[char] = &[
    '(', ')', '（', '）', '[', ']', '［', '］', '{', '}', '｛', '｝', '【', '】', '〔', '〕', '〈',
    '〉', '《', '》', '☆', '★', '♪', '♥', '♡', '◆', '◇', '■', '□', '▲', '▼', '●', '○', '※',
];

/// Whether `text` contains characters the reading never had.
///
/// A latin letter cannot come out of kana, so one the user did not type is the
/// model inventing text; the same argument covers the brackets and decorations
/// in [`NEVER_INVENTED`].
///
/// A **digit** is different, and treating it the same way was a bad bug: kana
/// number readings legitimately become digits, so the rule deleted every
/// conversion of a sentence containing a number. `こうほがよんけんみまんの…`
/// converts to `候補が4件未満の…` (NLL 0.05) and that candidate was thrown
/// away for spelling a `4`, which left only the kana fallbacks — and a fallback
/// is by definition a long unconverted run, so the typo-correction path then
/// fired and filled the window with `ウホが…` / `コホが…`. Every symptom came
/// from that one deletion.
///
/// So digits are judged by position instead: Japanese writes one before a
/// counter, and nowhere else. That still blocks a digit dropped into open text
/// while letting a counted quantity through.
fn invents_ascii(reading: &str, text: &str) -> bool {
    let reading_lower = reading.to_ascii_lowercase();
    let chars: Vec<char> = text.chars().collect();
    chars.iter().enumerate().any(|(i, &c)| {
        if c.is_ascii_digit() {
            return !reading.contains(c)
                && !chars
                    .get(i + 1)
                    .is_some_and(|&next| {
                        next.is_ascii_digit() || kashiko_engine::counter::starts_counter(next)
                    });
        }
        (c.is_ascii_alphabetic() && !reading_lower.contains(c.to_ascii_lowercase()))
            || (NEVER_INVENTED.contains(&c) && !reading.contains(c))
    })
}

/// Score-gate the deep search's findings before they join the list.
///
/// Findings come from beams over *edited* readings, so a wide beam or an
/// unlucky keyboard edit yields text that parses as Japanese but means
/// nothing. The quick pass gates its own candidates this way already;
/// skipping it here was why one Space press could offer nine candidates of
/// which only three were ever picked.
///
/// Everything is scored against the reading the user actually typed. That
/// is deliberately *not* the reading each finding assumes: a keyboard-edit
/// variant converts faithfully into its own hypothesis, so
/// `ほあさま → ホア様` scores beautifully against `ほあさま` and the gate
/// learns nothing. Scored against the typed reading the findings become
/// comparable to each other, which is the only comparison that separates
/// them.
///
/// What the margin is measured from depends on why the search ran:
///
/// - `MoreAlternatives` searched because the list was thin, so a finding
///   has to hold up against the list already on screen — the quick pass's
///   best (`last_quality_best`) joins the reference, and the absolute
///   ceiling applies.
/// - `TypoCorrection` searched because the input is broken, so every
///   finding departs from the typed reading by design and scores poorly
///   against it in absolute terms (measured on
///   `さいびるどまえだったtまえ…`: real corrections 1.9–2.3, junk 2.7–2.9).
///   Findings are ranked against each other only, with no ceiling.
///
/// Survivors come back best-first, and nothing caps the count: a search
/// that turns up eight good sentences shows eight, one that turns up only
/// junk shows none.
fn gate_deep_findings(
converter: &KanaKanjiConverter,
typed_reading: &str,
ctx: &str,
kind: DeepSearchKind,
quality_best: Option<f32>,
found: Vec<Candidate>,
) -> Vec<Candidate> {
    // Half-converted kana is no more welcome from the background search than
    // from the quick pass, and the rules that remove it there (10b3, 10d)
    // run too early to see these. Before scoring, so the scores stay aligned
    // with the candidates they belong to.
    let short = typed_reading.chars().count() <= 4;
    let loanword = kashiko_engine::loanword::english_for(typed_reading).is_some();
    let found: Vec<Candidate> = if short || loanword {
        let canonical = kashiko_engine::hiragana_to_katakana(typed_reading);
        found
            .into_iter()
            .filter(|c| {
                c.text == canonical
                    || c.text == typed_reading
                    || !kashiko_engine::is_all_kana(&c.text)
                    || kashiko_engine::is_pure_hiragana(&c.text)
                    || kashiko_engine::is_pure_full_katakana(&c.text)
            })
            .collect()
    } else {
        found
    };
    if found.is_empty() {
        return found;
    }

    let texts: Vec<String> = found.iter().map(|c| c.text.clone()).collect();
    let Ok(scores) = converter.score_candidates_detailed(typed_reading, ctx, &texts) else {
        return found;
    };
    let worst: Vec<Option<f32>> = scores.iter().map(|d| d.map(|d| d.worst)).collect();
    let scores: Vec<Option<f32>> = scores.iter().map(|d| d.map(|d| d.mean)).collect();

    // Short readings are ranked, never thinned — the same rule the quick pass
    // follows (`SHORT_READING_CHARS`), and for the same reason: with no sentence
    // around it a name reading scores badly whatever it is. Gating these would
    // throw away exactly the homophones this pass was scheduled to find (`優斗`
    // scores 2.13 against 0.10 for the verb reading `言うと`). The
    // worst-character rule still applies.
    let rank_only = short;
    let compare_with_shown = !rank_only && kind == DeepSearchKind::MoreAlternatives;
    let best = scores
        .iter()
        .copied()
        .flatten()
        .chain(compare_with_shown.then_some(quality_best).flatten())
        .fold(f32::INFINITY, f32::min);
    if !best.is_finite() {
        return found;
    }
    let ceiling = if compare_with_shown {
        QUALITY_CEILING_NLL
    } else {
        f32::INFINITY
    };
    let margin = if rank_only {
        f32::INFINITY
    } else {
        QUALITY_MARGIN_NLL
    };

    let mut kept: Vec<(f32, Candidate)> = found
        .into_iter()
        .zip(scores)
        .zip(worst)
        .filter_map(|((cand, score), worst)| match score {
            // Unscoreable is not evidence against a candidate; rank it as
            // if it tied the best so it keeps its place.
            None => Some((best, cand)),
            Some(s) => (s <= best + margin
                && s <= ceiling
                // Applies whatever the search's reason: a single character
                // the model flatly disbelieves is junk either way.
                && worst.is_none_or(|w| w <= QUALITY_WORST_TOKEN_NLL))
                .then_some((s, cand)),
        })
        .collect();
    kept.sort_by(|a, b| a.0.total_cmp(&b.0));
    kept.into_iter().map(|(_, c)| c).collect()
}

/// The phrases where two conversions of the same reading disagree, as
/// `(reading, a's text, b's text)` triples.
///
/// Both are aligned to the reading, and only spans the two alignments agree on
/// the boundaries of are compared — a phrase either lines up or is skipped,
/// which keeps a mis-alignment from producing a nonsense pair. Returns nothing
/// when either conversion cannot be aligned.
pub(super) fn differing_spans(reading: &str, a: &str, b: &str) -> Vec<(String, String, String)> {
    let (Some(sa), Some(sb)) = (
        kashiko_engine::segment::align(reading, a),
        kashiko_engine::segment::align(reading, b),
    ) else {
        return Vec::new();
    };
    let r: Vec<char> = reading.chars().collect();
    let ca: Vec<char> = a.chars().collect();
    let cb: Vec<char> = b.chars().collect();
    let slice = |v: &[char], from: usize, to: usize| -> String {
        v.get(from..to).map(|s| s.iter().collect()).unwrap_or_default()
    };
    let mut out = Vec::new();
    for pa in &sa {
        let Some(pb) = sb
            .iter()
            .find(|pb| pb.reading_start == pa.reading_start && pb.reading_end == pa.reading_end)
        else {
            continue;
        };
        let (ta, tb) = (
            slice(&ca, pa.converted_start, pa.converted_end),
            slice(&cb, pb.converted_start, pb.converted_end),
        );
        if ta == tb || ta.is_empty() || tb.is_empty() {
            continue;
        }
        out.push((slice(&r, pa.reading_start, pa.reading_end), ta, tb));
    }
    out
}

/// Sort the scored candidates best-first and drop the ones that read as junk,
/// leaving every other candidate exactly where it was.
///
/// This is the "show fewer, not more" rule: the number of candidates shown
/// follows their quality instead of being padded to a fixed count. Near
/// homophones the model rates far below the best reading (`パケットの約割と…`
/// next to `パケットの役割と…`) are removed rather than ranked lower, while a
/// list where everything scores well is kept in full.
///
/// Split out from the model call so the policy is testable without inference.
/// `scored` pairs an index into `candidates` with its per-character NLL
/// (lower = better); `None` means unscoreable, which is not evidence against a
/// candidate, so those are kept — after the scored ones.
pub(super) fn gate_and_reorder(
    candidates: &[AnnotatedCandidate],
    scored: &[Scored],
) -> Vec<AnnotatedCandidate> {
    let mut order: Vec<Scored> = scored.to_vec();
    // Stable sort: equal or unscoreable candidates keep their original order.
    order.sort_by(|a, b| match (a.score, b.score) {
        (Some(x), Some(y)) => x.total_cmp(&y),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    });

    let best = order
        .iter()
        .filter_map(|e| e.score)
        .fold(f32::INFINITY, f32::min);
    let survivors: Vec<usize> = order
        .iter()
        .enumerate()
        .filter(|(rank, e)| match e.score {
            // Unscoreable: keep.
            None => true,
            Some(s) => {
                e.never_drop
                    || *rank < MIN_KEPT_SCORED
                    || (s <= best + QUALITY_MARGIN_NLL
                        && s <= QUALITY_CEILING_NLL
                        && e.worst.is_none_or(|w| w <= QUALITY_WORST_TOKEN_NLL))
            }
        })
        .map(|(_, e)| e.index)
        .collect();

    // Rebuild the list: unscored candidates stay put (keeping learning on top
    // and the fallbacks at the bottom), and the scored block is replaced by its
    // survivors in ranked order, compacting away the dropped ones.
    let scored_slots: HashSet<usize> = scored.iter().map(|e| e.index).collect();
    let mut survivors = survivors.into_iter();
    let mut out = Vec::with_capacity(candidates.len());
    for (i, cand) in candidates.iter().enumerate() {
        if scored_slots.contains(&i) {
            if let Some(src) = survivors.next() {
                out.push(candidates[src].clone());
            }
        } else {
            out.push(cand.clone());
        }
    }
    out
}

/// Aux line shown while an inference pass is running.
const SEARCHING_AUX: &str = "推論中…";

/// Aux line offering another pass. Rendered as a button by the frontend.
const DEEPEN_AUX: &str = "さらに推論";

/// Aux line shown when a pass came back with nothing the gate would accept.
const NOTHING_MORE_AUX: &str = "候補なし・さらに推論";

/// Minimum length of a passed-through hiragana run (present verbatim in the
/// reading) for a model candidate to count as a low-quality "typo passthrough"
/// and for a full-sentence conversion to be deemed a failure worth splitting.
/// See `kashiko_engine::has_long_unconverted_run`.
///
/// Set to 7 so ordinary Japanese hiragana chains (adverb + particles, e.g.
/// `はとてもいい` = 6) don't trip it, while a genuinely garbled stretch left
/// unconverted (e.g. `できないようらな` = 8) does.
const UNCONVERTED_RUN_MIN: usize = 7;

/// Score added to a latin-word candidate the user has committed before, or
/// that a dictionary lists verbatim. Large enough to outrank any purely
/// heuristic score, since a previous commit is proof rather than a guess.
pub(super) const KNOWN_LATIN_BONUS: u32 = 16;

/// Minimum latin-word score worth showing at all.
const MIN_LATIN_SCORE: u32 = 6;

/// At or above this score the composition is treated as a deliberate latin
/// word, so the typo-recovery machinery (which reads the same "ASCII mixed
/// with kana" signal as a mistyped key) stands down.
const STRONG_LATIN_SCORE: u32 = 10;

/// Mozc-style width/script annotation for a pure-kana candidate, or `None`
/// if the text mixes scripts or contains kanji/punctuation. Used to label
/// `あ` / `ア` / `ｱ` candidates in the conversion list.
fn width_annotation(text: &str) -> Option<&'static str> {
    if kashiko_engine::is_pure_hiragana(text) {
        Some("[全]ひらがな")
    } else if kashiko_engine::is_pure_full_katakana(text) {
        Some("[全]カタカナ")
    } else {
        None
    }
}

/// Helper for building a deduplicated list of conversion candidates.
///
/// Two push paths exist: [`push`] dedups by text (skips duplicates), and
/// [`push_force`] always inserts (used for learning candidates that should
/// appear at the top even if a later source re-emits the same text).
pub(super) struct CandidateBuilder {
    candidates: Vec<AnnotatedCandidate>,
    seen: HashSet<String>,
}

/// `text` with CJK compatibility ideographs replaced by their unified form.
/// The dictionary spells some words both ways (有難う with U+96E3 and with
/// U+FA68), which render identically and so showed as a duplicate row.
fn compatibility_folded(text: &str) -> String {
    text.chars()
        .flat_map(|c| match c {
            '\u{F900}'..='\u{FAFF}' | '\u{2F800}'..='\u{2FA1F}' => {
                kashiko_engine::kana::normalize_nfkc(c.encode_utf8(&mut [0; 4]))
                    .chars()
                    .collect::<Vec<_>>()
            }
            _ => vec![c],
        })
        .collect()
}

impl CandidateBuilder {
    pub(super) fn new() -> Self {
        Self {
            candidates: Vec::new(),
            seen: HashSet::new(),
        }
    }

    /// Push a candidate if its text hasn't been seen yet.
    pub(super) fn push(&mut self, ac: AnnotatedCandidate) {
        let key = compatibility_folded(&ac.text);
        if !self.seen.contains(&key) && self.seen.insert(ac.text.clone()) {
            self.seen.insert(key);
            self.candidates.push(ac);
        }
    }

    /// Push a candidate unconditionally, marking its text as seen so later
    /// dedup'd inserts skip it. Use only for sources that should win over
    /// duplicates from later steps (e.g. learning cache).
    fn push_force(&mut self, ac: AnnotatedCandidate) {
        self.seen.insert(compatibility_folded(&ac.text));
        self.seen.insert(ac.text.clone());
        self.candidates.push(ac);
    }

    fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    pub(super) fn into_candidates(self) -> Vec<AnnotatedCandidate> {
        self.candidates
    }
}

impl InputMethodEngine {
    /// Run kana-kanji conversion for a reading via llama.cpp model.
    ///
    /// Determines the conversion strategy (main model, light model, or parallel beam),
    /// dispatches to the appropriate model(s), measures latency, and records which model was used.
    ///
    /// Skips the model entirely when the reading has no hiragana/katakana — the
    /// model is trained on kana → kanji and hallucinates garbage (e.g. `「` → `w`)
    /// for symbol- or alphabet-only inputs. Rule-based variants from
    /// `SymbolRewriter` cover those cases instead.
    ///
    /// `api_context` is the left context (lctx) fed to the model. Callers pass
    /// `truncate_context_for_api()` for a whole-buffer conversion, or — for
    /// chunked live conversion — the converted text of the preceding chunks.
    pub(super) fn run_kana_kanji_conversion(
        &mut self,
        reading: &str,
        api_context: &str,
        num_candidates: usize,
    ) -> Vec<String> {
        if !kashiko_engine::contains_kana(reading) {
            return vec![];
        }
        let Some(converter) = self.converters.kanji.as_ref() else {
            return vec![];
        };
        let katakana = kashiko_engine::hiragana_to_katakana(reading);
        let main_model_name = converter.model_display_name().to_string();

        let strategy = self.determine_strategy(reading, num_candidates);
        debug!(
            "convert: reading={} chars api_context={} chars candidates={} strategy={:?}",
            reading.chars().count(),
            api_context.chars().count(),
            num_candidates,
            strategy
        );

        let start = Instant::now();

        let candidates = match &strategy {
            ConversionStrategy::ParallelBeam { beam_width } => {
                let Some(light_converter) = self.converters.light_kanji.as_ref() else {
                    return vec![];
                };
                let bw = *beam_width;
                let (default_top1, light_candidates) = std::thread::scope(|s| {
                    let h_default = s.spawn(|| {
                        converter
                            .convert(&katakana, api_context, 1)
                            .unwrap_or_default()
                    });
                    let h_beam = s.spawn(|| {
                        light_converter
                            .convert(&katakana, api_context, bw)
                            .unwrap_or_default()
                    });
                    (
                        h_default.join().unwrap_or_default(),
                        h_beam.join().unwrap_or_default(),
                    )
                });
                Self::merge_candidates_dedup(default_top1, light_candidates, bw)
            }
            ConversionStrategy::LightModelOnly => {
                let Some(light_converter) = self.converters.light_kanji.as_ref() else {
                    return vec![];
                };
                // Cap the beam like the other multi-candidate strategies do:
                // asking the light model for the full `num_candidates` (9)
                // floods long-input conversion with low-probability
                // near-duplicates. Beam output is probability-ordered, so
                // keeping the top `beam_width` drops the unlikely tail.
                let n = if num_candidates > 1 {
                    num_candidates.min(self.config.beam_width.max(1))
                } else {
                    num_candidates
                };
                if n == 1 {
                    // Single-candidate = the live/auto-suggest path: use the
                    // self-checking conversion (see `convert_live`).
                    light_converter
                        .convert_live(reading, api_context)
                        .map(|t| vec![t])
                        .unwrap_or_default()
                } else {
                    light_converter
                        .convert(&katakana, api_context, n)
                        .unwrap_or_default()
                }
            }
            ConversionStrategy::MainModelOnly if num_candidates == 1 => converter
                .convert_live(reading, api_context)
                .map(|t| vec![t])
                .unwrap_or_default(),
            ConversionStrategy::MainModelOnly => converter
                .convert(&katakana, api_context, num_candidates)
                .unwrap_or_default(),
            ConversionStrategy::MainModelBeam { beam_width } => converter
                .convert(&katakana, api_context, *beam_width)
                .unwrap_or_default(),
        };

        self.metrics.conversion_ms = start.elapsed().as_millis() as u64;
        self.update_adaptive_model_flag(&strategy);

        self.metrics.model_name = match &strategy {
            ConversionStrategy::ParallelBeam { .. } => {
                let light_name = self
                    .converters
                    .light_kanji
                    .as_ref()
                    .map(|c| c.model_display_name().to_string())
                    .unwrap_or_default();
                format!("{}+{}", main_model_name, light_name)
            }
            ConversionStrategy::LightModelOnly => self
                .converters
                .light_kanji
                .as_ref()
                .map(|c| c.model_display_name().to_string())
                .unwrap_or(main_model_name),
            ConversionStrategy::MainModelOnly | ConversionStrategy::MainModelBeam { .. } => {
                main_model_name
            }
        };

        candidates
    }

    /// Start kanji conversion for the current input buffer.
    ///
    /// Called when DOWN/TAB/SPACE is pressed: flushes any pending romaji,
    /// resolves the reading, runs `build_conversion_candidates`, and
    /// transitions into the Conversion state. The previous live-conversion
    /// result is preserved as the first model candidate so the user sees
    /// the same text they had been looking at during input.
    ///
    /// `skip_learning` is set by the Tab path to omit learning-cache
    /// candidates (Space/Down keep the default learning-included behavior).
    ///
    /// `initial_selection`: `-1` moves to the previous candidate (Up), `1`
    /// means "advance past a live-conversion duplicate" (Space/Down) — if
    /// the top candidate is exactly the text live conversion was already
    /// showing, the user has already seen it, so land on the second
    /// candidate instead of re-selecting the one they just looked at. `0`
    /// (Tab, and re-entry from `resize_conversion`) always keeps the first
    /// candidate selected.
    pub(super) fn start_conversion(
        &mut self,
        skip_learning: bool,
        split_index: Option<usize>,
        initial_selection: isize,
    ) -> EngineResult {
        // Flush any remaining romaji into composed_hiragana
        self.flush_romaji_to_composed();
        // A fresh conversion of the buffer starts with the focus at the
        // beginning, so any segments settled by a previous focus slide are
        // discarded rather than double-counted.
        self.decided.clear();
        self.tail_text = None;
        self.segment_bounds.clear();

        let full_reading = self.input_buf.text.clone();
        let total_chars = full_reading.chars().count();

        // When the caller didn't pin a split (normal Space) and the input has
        // a leftover-romaji typo, cut the conversion just before it so the
        // clean prefix converts and the rest stays as raw reading.
        // Never in emoji mode: an emoji shortcode is `:` followed by ASCII by
        // definition, so the leftover-romaji split fired on every one of them
        // and cut the reading down to the bare `:`. That is why `:smile` and
        // `:kiniku` offered colon variants (`：`, `︙`) instead of 😄 and 💪 —
        // the rewriter never saw the shortcode.
        let auto_split = if split_index.is_none() && self.input_mode != InputMode::Emoji {
            self.auto_failure_split(&full_reading)
        } else {
            None
        };
        let mut split_idx = split_index.or(auto_split).unwrap_or(total_chars);
        let mut reading: String = full_reading.chars().take(split_idx).collect();

        // For long input (and whenever we auto-split), hand the cached chunk
        // conversion to the candidate builder so the text the user was just
        // looking at is guaranteed to stay the first candidate; the builder
        // still runs a fresh (light-model) pass on top of it for alternatives
        // and typo-corrected guesses.
        let cached_model = if auto_split.is_some() || total_chars > self.config.short_input_threshold
        {
            self.cached_prefix_conversion(split_idx)
        } else {
            None
        };

        // Save auto-suggest/live conversion result before clearing state.
        // This ensures the candidate that was displayed during input is preserved
        // in the conversion candidate list even if the re-inference uses a different strategy.
        let mut prev_suggest_text = std::mem::take(&mut self.live.text);

        self.converters.romaji.reset();
        self.input_buf.cursor_pos = 0;

        if reading.is_empty() {
            return EngineResult::consumed();
        }

        // Stray romaji anywhere in the reading (including position 0, e.g.
        // `wたし` — a missed vowel key): before falling back to the
        // split-at-typo behavior, ask the model what the user *meant*, via
        // BOTH interpretations of the leftover keystroke:
        //   - completion: the vowel was never pressed — finish the romaji
        //     with each vowel (`w`→わ…) and convert (`wたし` → わたし → 私);
        //   - strip: the key was hit by accident — remove the ASCII and
        //     convert the cleaned whole (`wたし` → たし → 足し).
        // Both result sets are shown (completions first); committing either
        // consumes the whole buffer, stray chars included. Only when neither
        // produces a real conversion do we keep the old split behavior
        // (prefix converted, tail left raw).
        //
        // A composition we're confident is a deliberate latin word reads the
        // same way ("ASCII mixed with kana") but must NOT go down this path:
        // stripping or vowel-completing `WordPress` produces nonsense, and
        // rewriting `reading` to the stripped kana would build the candidate
        // list for `おレッ`. Step 10c offers the spelling instead.
        let deliberate_latin = self
            .latin_word_candidate()
            .is_some_and(|(_, score)| score >= STRONG_LATIN_SCORE);
        let has_stray_ascii = !deliberate_latin
            && full_reading.chars().any(|c| c.is_ascii_alphabetic())
            && kashiko_engine::contains_kana(&full_reading);
        let mut candidates: Option<Vec<AnnotatedCandidate>> = None;
        if split_index.is_none() && has_stray_ascii {
            /// Above this reading length the sync completion passes are
            /// skipped (the async deep search covers them) to keep the
            /// Space-key response fast.
            const SYNC_COMPLETION_MAX_CHARS: usize = 15;
            const MAX_SYNC_COMPLETIONS: usize = 3;

            let mut combined: Vec<AnnotatedCandidate> = Vec::new();

            // Completion pattern (one greedy pass per vowel, short input only).
            if total_chars <= SYNC_COMPLETION_MAX_CHARS {
                let api_context = self.truncate_context_for_api();
                for variant in kashiko_engine::typo::romaji_completion_variants(&full_reading) {
                    if combined.len() >= MAX_SYNC_COMPLETIONS {
                        break;
                    }
                    if let Some(out) = self
                        .run_kana_kanji_conversion(&variant, &api_context, 1)
                        .into_iter()
                        .next()
                        && out != variant
                        && !kashiko_engine::has_long_unconverted_run(
                            &out,
                            &variant,
                            UNCONVERTED_RUN_MIN,
                        )
                        && !kashiko_engine::has_unrenderable_char(&out)
                        && !combined.iter().any(|c| c.text == out)
                    {
                        combined.push(
                            AnnotatedCandidate::new(out, CandidateSource::Model)
                                .with_reading(Some(variant)),
                        );
                    }
                }
            }

            // Strip pattern.
            let cleaned: String = full_reading
                .chars()
                .filter(|c| !c.is_ascii_alphabetic())
                .collect();
            let mut has_real_conversion = !combined.is_empty();
            if kashiko_engine::contains_kana(&cleaned) {
                let cleaned_candidates = self.build_conversion_candidates(
                    &cleaned,
                    self.config.num_candidates,
                    skip_learning,
                    None,
                    // Whole-buffer scope: the latin word belongs here.
                    true,
                );
                has_real_conversion |= cleaned_candidates
                    .iter()
                    .any(|c| c.source == CandidateSource::Model && c.text != cleaned);
                for c in cleaned_candidates {
                    if !combined.iter().any(|m| m.text == c.text) {
                        combined.push(c);
                    }
                }
            }

            if has_real_conversion && !combined.is_empty() {
                candidates = Some(combined);
                reading = full_reading
                    .chars()
                    .filter(|c| !c.is_ascii_alphabetic())
                    .collect();
                split_idx = total_chars;
                // The live text shown during typing contained the stray
                // romaji — don't resurface it over the corrected list.
                prev_suggest_text.clear();
            }
        }

        // Get candidates from kanji converter (use full num_candidates for explicit conversion)
        let mut candidates = match candidates {
            Some(c) => c,
            None => self.build_conversion_candidates(
                &reading,
                self.sync_candidate_count(&reading),
                skip_learning,
                cached_model,
                // Latin candidate only when converting the whole buffer.
                split_idx == total_chars,
            ),
        };

        // Conversion-failure fallbacks: we tried to convert the whole reading
        // but the best result still passes a garbled stretch through as raw
        // hiragana (a typo the model couldn't handle). Two recovery attempts,
        // in order:
        //
        // 1. Kana-typo correction search — try small edits (deletions /
        //    adjacent swaps) in the garbled region and convert each; clean
        //    results are shown as "what you probably meant" candidates for
        //    the whole buffer (e.g. `…だったっため…` → `…だったためと思われます`).
        // 2. Dictionary-prefix split — isolate the longest dictionary-word
        //    head (`出来ない` out of `できないようらなけす`) and leave the rest
        //    underlined for manual fixing.
        let mut typo_suspected = has_stray_ascii || auto_split.is_some();
        if split_index.is_none() && split_idx == total_chars {
            let garbled_top = candidates.first().and_then(|c| {
                kashiko_engine::has_long_unconverted_run(&c.text, &reading, UNCONVERTED_RUN_MIN)
                    .then(|| c.text.clone())
            });
            // A long kana run shared with the reading is not proof of failure:
            // `皆さんおはようございます` legitimately keeps nine of them, and
            // it converts at 0.05 NLL. Asking the model before believing the
            // heuristic costs one forward pass and stops the correction search
            // from replacing a good list with edits of the reading
            // (`みなさんお様ございます`). Same guard as the deep search's, which
            // is expressed against `last_quality_best`.
            let confident_top = garbled_top
                .as_deref()
                .and_then(|top| self.score_one(&reading, top))
                .is_some_and(|score| score <= 0.5);
            if let Some(garbled_top) = garbled_top.filter(|_| !confident_top) {
                typo_suspected = true;
                let corrections = self.kana_typo_corrections(&reading, &garbled_top);
                if !corrections.is_empty() {
                    // Corrections lead; keep only the clean survivors of the
                    // original list (fallback hiragana/katakana stay via the
                    // exemption below, the garbled passthroughs drop).
                    let mut merged = corrections;
                    for c in candidates {
                        let is_garbled = kashiko_engine::has_long_unconverted_run(
                            &c.text,
                            &reading,
                            UNCONVERTED_RUN_MIN,
                        ) && c.source != CandidateSource::Fallback;
                        if !is_garbled && !merged.iter().any(|m| m.text == c.text) {
                            merged.push(c);
                        }
                    }
                    candidates = merged;
                    prev_suggest_text.clear();
                } else if let Some(prefix_len) = self.dictionary_prefix_split(&full_reading) {
                    let prefix_reading: String =
                        full_reading.chars().take(prefix_len).collect();
                    let prefix_candidates = self.build_conversion_candidates(
                        &prefix_reading,
                        self.config.num_candidates,
                        skip_learning,
                        None,
                        // Prefix scope: the whole-buffer keystrokes don't apply.
                        false,
                    );
                    let prefix_is_clean = prefix_candidates.first().is_some_and(|c| {
                        c.text != prefix_reading
                            && !kashiko_engine::has_long_unconverted_run(
                                &c.text,
                                &prefix_reading,
                                UNCONVERTED_RUN_MIN,
                            )
                    });
                    if prefix_is_clean {
                        candidates = prefix_candidates;
                        split_idx = prefix_len;
                        reading = prefix_reading;
                        // The saved live-conversion text is for the *whole*
                        // reading, so it must not be reinserted over the
                        // now-shorter prefix.
                        prev_suggest_text.clear();
                    }
                }
            }
        }

        // Offer the dictionary's other readings of each word before ranking, so
        // a word the beam never produced can still win (see
        // `dictionary_substitution_variants`). Added as Dictionary candidates,
        // which puts them in the block the reranker scores and gates.
        if let Some(top) = candidates.first().map(|c| c.text.clone()) {
            let existing: HashSet<String> =
                candidates.iter().map(|c| c.text.clone()).collect();
            // Inserted above the hiragana/katakana fallbacks, which must stay
            // at the very bottom of the list (they are what Up from the top
            // wraps around to).
            let at = candidates
                .iter()
                .position(|c| c.source == CandidateSource::Fallback)
                .unwrap_or(candidates.len());
            let new: Vec<AnnotatedCandidate> = self
                .dictionary_substitution_variants(&reading, &top)
                .into_iter()
                .filter(|v| !existing.contains(v))
                // These are spliced in here rather than inside
                // `build_conversion_candidates`, so the rules applied there
                // have to be repeated. The dictionary holds latin aliases
                // under kana readings (`きぼう` → `KIBOU`), which spliced into
                // a sentence produce `ごKIBOUなら試飲補完を…`.
                .filter(|v| !invents_ascii(&reading, v))
                .filter(|v| {
                    self.learning
                        .as_ref()
                        .is_none_or(|cache| !cache.is_suppressed(&reading, v))
                })
                .map(|v| AnnotatedCandidate::new(v, CandidateSource::Dictionary))
                .collect();
            candidates.splice(at..at, new);
        }

        // Rank the model/dictionary candidates by how natural they read in
        // context before anything is pinned on top, so the ordering reflects
        // the language model rather than the decoder's beam order.
        self.rerank_by_likelihood(&reading, &mut candidates);

        // Words kept in Latin script might have been meant as kana; the
        // all-Japanese reading goes just above the fallbacks.
        if split_idx >= total_chars {
            let alternatives = self.kana_alternatives();
            let at = candidates
                .iter()
                .position(|c| c.source == CandidateSource::Fallback)
                .unwrap_or(candidates.len());
            let existing: HashSet<String> = candidates.iter().map(|c| c.text.clone()).collect();
            let fresh: Vec<AnnotatedCandidate> =
                alternatives.into_iter().filter(|c| !existing.contains(&c.text)).collect();
            candidates.splice(at..at, fresh);
        }

        // The live-conversion text leads the list unconditionally: it is what
        // the user was already looking at, so Space must never appear to throw
        // it away. Pulled to the front when the candidate sources produced it
        // too, inserted when they didn't.
        //
        // Testing `!seen.contains(...)` here instead was the bug behind "the
        // live conversion one is gone": when the sources *did* produce it, it
        // was left wherever reranking had put it, which for a long sentence
        // meant several rows down and out of sight. Placed after reranking and
        // the quality gate, so neither can bury or drop it.
        // A ✕-ed sentence must not sneak back in as the pin: the pin is
        // inserted after the suppression filter, and live conversion works on
        // chunks whose readings don't match a whole-sentence suppression, so
        // nothing upstream catches it.
        let pin_suppressed = self
            .learning
            .as_ref()
            .is_some_and(|cache| cache.is_suppressed(&reading, &prev_suggest_text));
        if !prev_suggest_text.is_empty() && prev_suggest_text != reading && !pin_suppressed {
            let known = candidates.iter().any(|c| c.text == prev_suggest_text);
            candidates.retain(|c| c.text != prev_suggest_text);
            candidates.insert(
                0,
                AnnotatedCandidate::new(prev_suggest_text.clone(), CandidateSource::Model),
            );
            // Being pinned means reranking never scored it, so it contributed
            // nothing to the bar the deep search has to clear — and the pinned
            // text is usually the best reading there is. One extra forward pass
            // buys the correct bar: without it, `皆さんおはようございます`
            // (0.05) was pinned while the recorded best stayed 0.46, doubling
            // the tolerance and letting `皆さんおは様ございます` through.
            if !known
                && let Some(score) = self.score_one(&reading, &prev_suggest_text)
            {
                self.last_quality_best =
                    Some(self.last_quality_best.map_or(score, |b| b.min(score)));
            }
        }

        if candidates.is_empty() {
            // No candidates, stay in hiragana mode
            let preedit = Preedit::with_text_underlined(&reading);
            self.state = InputState::Composing {
                preedit: preedit.clone(),
                romaji_buffer: String::new(),
            };
            return EngineResult::consumed().with_action(EngineAction::UpdatePreedit(preedit));
        }

        // How many "real" conversions (not the always-present hiragana/
        // katakana fallbacks or symbol variants) the quick pass produced.
        // A thin list means the model likely has unexplored alternatives —
        // schedule the deep search to keep inferring in the background.
        let real_candidate_count = candidates
            .iter()
            .filter(|c| {
                !matches!(
                    c.source,
                    CandidateSource::Fallback | CandidateSource::Rewriter
                )
            })
            .count();
        // Whether the best candidate still passes a garbled stretch through
        // (or is just the raw reading) — the "conversion failed" signal used
        // to trigger the deep recovery search below.
        let top_garbled = candidates.first().is_some_and(|c| {
            kashiko_engine::has_long_unconverted_run(&c.text, &reading, UNCONVERTED_RUN_MIN)
        });

        // Map AnnotatedCandidate → public Candidate. The two annotation
        // slots are kept disjoint so descriptions never duplicate between the
        // aux text and the candidate's right-side comment:
        //   - `source_label` ← source.label() only (e.g. `🤖 AI`, `📚 辞書`)
        //   - `description`  ← the per-candidate description only
        //                      (e.g. `三点リーダ`, `[全]英大文字`)
        let mut candidate_list = CandidateList::new(
            candidates
                .into_iter()
                .map(|ac| {
                    let cand_reading = ac.reading.unwrap_or_else(|| reading.clone());
                    let label = ac.source.label();
                    Candidate {
                        text: ac.text,
                        reading: Some(cand_reading),
                        source_label: (!label.is_empty()).then(|| label.to_string()),
                        description: ac.description,
                    }
                })
                .collect(),
        );

        if initial_selection == 1 {
            // Space/Down land on the *second* candidate when the first is the
            // live-conversion text: the user has already seen that reading, so
            // asking to convert means "show me something else". The first row
            // stays available by pressing Up.
            let top_is_live_text = candidate_list
                .candidates()
                .first()
                .is_some_and(|c| !prev_suggest_text.is_empty() && c.text == prev_suggest_text);
            if top_is_live_text {
                candidate_list.move_next();
            }
        } else if initial_selection == -1 {
            candidate_list.move_prev();
        }

        // Schedule the post-response deep search, scoped to whatever is
        // actually being converted — the whole buffer OR the prefix segment
        // of a manual Left/Right resize. It runs *after* this synchronous
        // response is written, and its findings are pushed to the frontend
        // as an `update_candidates` notification, so the quick candidates
        // appear immediately and extra guesses stream in behind the loading
        // aux. Two triggers:
        //   - a typo is suspected → keyboard-edit correction search;
        //   - the quick list is thin → widen the beam / tail alternatives.
        const MIN_QUICK_REAL_CANDIDATES: usize = 4;
        // Signals recomputed against the *scoped* reading, so a resized
        // segment whose conversion failed (only fallback candidates) also
        // gets the recovery search.
        let scoped_raw: String = full_reading.chars().take(split_idx).collect();
        // Emoji mode is excluded for the same reason as the split above: the
        // ASCII in a shortcode is the query, not a mistyped key.
        let scope_has_ascii = !deliberate_latin
            && self.input_mode != InputMode::Emoji
            && scoped_raw.chars().any(|c| c.is_ascii_alphabetic());
        // A thin list is the trigger, whatever the reading's length. The length
        // floor that used to guard this was there because a short reading
        // normally produces few candidates, so the search would have run on
        // nearly every Space press — but short readings now keep their
        // homophones (see `SHORT_READING_CHARS`), so a thin list there means
        // the same thing it means anywhere else: there is more to find.
        // A conversion the model is confident about is not a typo, whatever the
        // heuristics suspected. Both suspicions are heuristic and both misfire
        // on correct input: `みなさんおはようございます` converts to
        // `皆さんおはようございます` at 0.05 NLL, yet `おはようございます` is a
        // nine-character kana run shared with the reading, so `top_garbled`
        // read it as a failed conversion and ran the keyboard-edit search
        // anyway. Its variants come back as faithful conversions of nonsense
        // (`ほあさま` → `ホア様`), which no amount of scoring the correction
        // itself can reject — the only fix is not to ask. Raw keystrokes left
        // in the reading stay an unconditional trigger: that is not a
        // heuristic, the input is provably unfinished.
        const TYPO_SEARCH_MAX_BEST_NLL: f32 = 0.5;
        let converted_confidently = self
            .last_quality_best
            .is_some_and(|best| best <= TYPO_SEARCH_MAX_BEST_NLL);
        let kind = if scope_has_ascii || ((typo_suspected || top_garbled) && !converted_confidently)
        {
            Some(DeepSearchKind::TypoCorrection)
        } else if real_candidate_count < MIN_QUICK_REAL_CANDIDATES
            || self.sync_candidate_count(&reading) < self.config.num_candidates
        {
            // The second condition is the other half of moving beam width off
            // the blocking path: the synchronous pass deliberately asked the
            // model for fewer candidates than configured, so the ones it did not
            // look for have to be found somewhere. Without it, narrowing the
            // sync beam lost `ゆうと` → `優斗` and `かんじ` → `完治` outright,
            // because a reading with a full dictionary never looked thin enough
            // to trigger a search.
            Some(DeepSearchKind::MoreAlternatives)
        } else {
            None
        };
        self.deep_round = 0;
        // A new conversion supersedes any search still running for the old one:
        // its result would be dropped as stale anyway, so stop it rather than
        // let it keep a core busy. `build_deep_job` installs a fresh flag, so
        // this never cancels the search we are about to schedule.
        self.deep_cancel.store(true, Ordering::Relaxed);
        self.deep_search = kind.map(|kind| DeepSearchRequest {
            reading: scoped_raw.clone(),
            buffer: full_reading.clone(),
            scope: split_idx,
            kind,
            round: 0,
        });

        let mut result = self.enter_conversion_state(&full_reading, candidate_list, split_idx);
        // The aux line is either a progress message or the button that asks for
        // another pass. It is offered whenever there is a reading to infer over,
        // including the short ones the automatic search skips — those are
        // exactly where the user is most likely to want more.
        result = result.with_action(if self.deep_search.is_some() {
            EngineAction::UpdateAuxText(SEARCHING_AUX.to_string())
        } else if kashiko_engine::contains_kana(&scoped_raw) {
            EngineAction::UpdateAuxButton(DEEPEN_AUX.to_string())
        } else {
            EngineAction::HideAuxText
        });
        result
    }

    /// Run another, wider inference pass over the conversion on screen — the
    /// aux-line button.
    ///
    /// Each press raises the round, which `build_deep_job` spends on wider beams
    /// and a higher result cap, so pressing it repeatedly keeps finding new
    /// things instead of re-running the same search. Returns immediately with
    /// the progress message; the findings arrive later as an
    /// `update_candidates` notification, exactly like the automatic pass.
    pub fn deepen_inference(&mut self) -> EngineResult {
        let InputState::Conversion { split_index, .. } = self.state else {
            return EngineResult::not_consumed();
        };
        if self.deep_search.is_some() || self.deep_running {
            // Already working — say so rather than queueing a second job.
            return EngineResult::consumed()
                .with_action(EngineAction::UpdateAuxText(SEARCHING_AUX.to_string()));
        }
        let reading: String = self.input_buf.text.chars().take(split_index).collect();
        if !kashiko_engine::contains_kana(&reading) {
            return EngineResult::not_consumed();
        }
        self.deep_round += 1;
        // Typo correction first, then widening: a press after the correction
        // pass has run should look for alternatives rather than repeat it.
        let kind = if self.deep_round == 1 && reading.chars().any(|c| c.is_ascii_alphabetic()) {
            DeepSearchKind::TypoCorrection
        } else {
            DeepSearchKind::MoreAlternatives
        };
        self.deep_search = Some(DeepSearchRequest {
            reading,
            buffer: self.input_buf.text.clone(),
            scope: split_index,
            kind,
            round: self.deep_round,
        });
        EngineResult::consumed().with_action(EngineAction::UpdateAuxText(SEARCHING_AUX.to_string()))
    }

    /// Transition to Conversion state with the given reading and candidate list.
    ///
    /// Sets up the preedit (highlighted selected text), updates the state, and
    /// returns an EngineResult with preedit, candidates, and aux text actions.
    fn enter_conversion_state(&mut self, full_reading: &str, candidates: CandidateList, split_index: usize) -> EngineResult {
        self.enter_conversion_state_at(full_reading, candidates, 0, split_index)
    }

    /// As [`Self::enter_conversion_state`], but for a focused segment that may
    /// start past the beginning: `[segment_start, split_index)` is highlighted,
    /// `self.decided` renders ahead of it, and the reading after it stays raw.
    fn enter_conversion_state_at(
        &mut self,
        full_reading: &str,
        candidates: CandidateList,
        segment_start: usize,
        split_index: usize,
    ) -> EngineResult {
        use crate::core::preedit::AttributeType::Underline;

        let reading: String = full_reading
            .chars()
            .skip(segment_start)
            .take(split_index.saturating_sub(segment_start))
            .collect();
        let selected_text = candidates.selected_text().unwrap_or(&reading).to_string();
        let decided_text: String = self.decided.iter().map(|(_, t)| t.as_str()).collect();

        let mut segments = Vec::new();
        if !decided_text.is_empty() {
            segments.push(PreeditSegment::new(&decided_text, Underline));
        }
        segments.push(PreeditSegment::highlighted(&selected_text));
        if split_index < full_reading.chars().count() {
            // The tail keeps its conversion while the focus is being moved
            // around (`tail_text`); it falls back to the raw reading only for
            // the typo split, where leaving it unconverted is the point.
            let tail = self.tail_text.clone().unwrap_or_else(|| {
                full_reading.chars().skip(split_index).collect()
            });
            segments.push(PreeditSegment::new(&tail, Underline));
        }

        let preedit = Preedit::from_segments(
            segments,
            decided_text.chars().count() + selected_text.chars().count(),
        );

        self.state = InputState::Conversion {
            preedit: preedit.clone(),
            candidates: candidates.clone(),
            split_index,
            segment_start,
        };

        EngineResult::consumed()
            .with_action(EngineAction::UpdatePreedit(preedit))
            .with_action(EngineAction::ShowCandidates(candidates.clone()))
            .with_action(EngineAction::UpdateAuxText(
                self.format_aux_conversion_with_page(full_reading, Some(&candidates)),
            ))
    }

    /// Search user and system dictionaries for candidates matching a reading.
    ///
    /// User dictionary results come first (higher priority), then system dictionary
    /// results sorted by score. Duplicates are removed via HashSet.
    pub(super) fn search_dictionaries(&self, reading: &str, limit: usize) -> Vec<AnnotatedCandidate> {
        let mut candidates = Vec::new();
        let mut seen = HashSet::new();

        // User dictionary (higher priority)
        if let Some(dict) = &self.dicts.user
            && let Some(result) = dict.exact_match_search(reading)
        {
            for cand in result.candidates {
                if candidates.len() >= limit {
                    break;
                }
                if seen.insert(cand.surface.to_string()) {
                    candidates.push(AnnotatedCandidate::new(
                        cand.surface.to_string(),
                        CandidateSource::UserDictionary,
                    ));
                }
            }
        }

        // System dictionary (sorted by score)
        if let Some(dict) = &self.dicts.system
            && let Some(result) = dict.exact_match_search(reading)
        {
            let mut dict_candidates: Vec<_> = result.candidates.to_vec();
            dict_candidates.sort_by(|a, b| a.score.total_cmp(&b.score));
            for cand in dict_candidates {
                if candidates.len() >= limit {
                    break;
                }
                if seen.insert(cand.surface.to_string()) {
                    candidates.push(AnnotatedCandidate::new(
                        cand.surface,
                        CandidateSource::Dictionary,
                    ));
                }
            }
        }

        candidates
    }

    /// The raw keystrokes as a latin-word candidate, with a confidence score,
    /// when the composition looks like a latin word the romaji converter
    /// mangled into kana.
    ///
    /// Typing `WordPress` yields the buffer `WおrdPレッ`: the keystrokes are
    /// the only place the intended spelling still exists, so they are offered
    /// back as a candidate. See [`kashiko_engine::latin::latin_word_confidence`]
    /// for the gates that keep ordinary Japanese input (whose keystrokes are
    /// also pure romaji) out. Having committed the word before, or finding it
    /// verbatim in a dictionary, adds a large bonus — that is proof, not a
    /// guess — and pushes the candidate to the top of the list.
    ///
    /// The score is what ranks the candidate *among* the real conversions
    /// rather than pinning it to either end (see `latin_insert_index`).
    pub(super) fn latin_word_candidate(&self) -> Option<(String, u32)> {
        let raw = self.input_buf.raw_keys.trim();
        let mut score = kashiko_engine::latin::latin_word_confidence(raw, &self.input_buf.text)?;
        let known = self.learning.as_ref().is_some_and(|c| c.has_surface(raw))
            || [self.dicts.user.as_ref(), self.dicts.system.as_ref()]
                .into_iter()
                .flatten()
                .any(|d| d.exact_match_search(raw).is_some());
        if known {
            score += KNOWN_LATIN_BONUS;
        }
        (score >= MIN_LATIN_SCORE).then(|| (raw.to_string(), score))
    }

    /// Where a latin-word candidate scoring `score` belongs in the candidate
    /// list: proven words lead, strong guesses sit just under the top
    /// conversion, weaker ones a little further down — but always *within* the
    /// normal candidates, never appended after the fallbacks.
    fn latin_insert_index(score: u32, len: usize) -> usize {
        let rank = if score >= KNOWN_LATIN_BONUS {
            0
        } else if score >= 12 {
            1
        } else if score >= 9 {
            2
        } else {
            3
        };
        rank.min(len)
    }

    /// Longest dictionary-word reading that is a *proper* prefix of `reading`
    /// (non-empty and strictly shorter than the whole reading), or `None`.
    ///
    /// Used to isolate the cleanly-convertible head of a phrase when the
    /// full-reading conversion failed because of a typo in the tail: converting
    /// just this prefix (e.g. `できない` out of `できないようらなけす`) lets the
    /// good part commit while the garbled remainder stays raw for re-editing.
    /// Checks user then system dictionaries via their trie prefix search.
    fn dictionary_prefix_split(&self, reading: &str) -> Option<usize> {
        let total = reading.chars().count();
        let mut best = 0usize;
        for dict in [self.dicts.user.as_ref(), self.dicts.system.as_ref()]
            .into_iter()
            .flatten()
        {
            for res in dict.common_prefix_search(reading) {
                let len = res.reading.chars().count();
                // Require at least 2 chars so we never split off a lone kana.
                if (2..total).contains(&len) && len > best {
                    best = len;
                }
            }
        }
        (best > 0).then_some(best)
    }

    /// Merge a finished deep search into the live candidate list and build the
    /// notification result.
    ///
    /// The aux line goes back to offering another pass rather than clearing:
    /// whatever this round found, the user can ask again and the next round
    /// searches wider.
    fn finish_deep_job(&mut self, job: DeepSearchOutcome) -> EngineResult {
        let found = job.found;
        let done = || {
            EngineResult::consumed()
                .with_action(EngineAction::UpdateAuxButton(NOTHING_MORE_AUX.to_string()))
        };
        if found.is_empty() {
            return done();
        }
        if !matches!(self.state, InputState::Conversion { .. }) {
            return done();
        }
        // Deleted candidates must not come back through the side door: the
        // quick pass filters suppressed text but the deep search merges its
        // findings straight into the list, so a ✕-ed candidate reappeared a
        // second later when the background search rediscovered it.
        // ...and neither may the model invent characters here, the same rule
        // the quick pass applies: `ごKIBOUなら試飲補完を…` is a beam over an
        // edited reading spelling out keystrokes that were never typed.
        //
        // Both need the learning cache and so stay on this thread; the scoring
        // gate, which needs the model, already ran in the worker.
        let mut found: Vec<Candidate> = found
            .into_iter()
            .filter(|c| !invents_ascii(&job.reading, &c.text))
            .collect();
        if let Some(cache) = self.learning.as_ref() {
            found.retain(|c| !cache.is_suppressed(&job.reading, &c.text));
        }
        if found.is_empty() {
            return done();
        }
        let InputState::Conversion { candidates, .. } = &mut self.state else {
            return done();
        };
        // Insert findings right after the currently selected candidate so they
        // are visible on the *current page* — inserting at the bottom (above
        // the fallbacks) buried them on page 2 whenever the quick pass had
        // already filled page 1, which made the deep results look like they
        // never arrived. Capped at the hiragana/katakana fallback position so
        // those always stay at the very bottom, and the user's currently
        // selected candidate stays selected — the list grows below the
        // selection, it never jumps.
        let selected_text = candidates.selected_text().map(str::to_string);
        let cursor = candidates.cursor();
        let mut all: Vec<Candidate> = candidates.candidates().to_vec();
        let fallback_at = all
            .iter()
            .position(|c| {
                matches!(c.description.as_deref(), Some("[全]ひらがな") | Some("[全]カタカナ"))
            })
            .unwrap_or(all.len());
        let insert_at = (cursor + 1).min(fallback_at);
        all.splice(insert_at..insert_at, found);
        let mut list = CandidateList::new(all);
        if let Some(sel) = selected_text
            && let Some(idx) = list.candidates().iter().position(|c| c.text == sel)
        {
            list.select(idx);
        }
        *candidates = list.clone();
        EngineResult::consumed()
            .with_action(EngineAction::ShowCandidates(list))
            .with_action(EngineAction::UpdateAuxButton(DEEPEN_AUX.to_string()))
    }

    /// Split `reading` into `(prefix, tail)` for tail-alternative generation:
    /// the last live-conversion chunk when the chunk cache still matches the
    /// reading, else the trailing hiragana run (2..=8 chars, strictly shorter
    /// than the whole reading). `None` when no sensible tail exists.
    fn split_tail_for_alternatives(&self, reading: &str) -> Option<(String, String)> {
        // Chunk-aligned split (multi-chunk long input).
        if self.chunks.len() >= 2 {
            let concat: String = self.chunks.iter().map(|c| c.reading.as_str()).collect();
            if concat == reading
                && let Some(last) = self.chunks.last()
            {
                let tail = last.reading.clone();
                let prefix_len = reading.chars().count() - tail.chars().count();
                if prefix_len > 0 && !tail.is_empty() {
                    let prefix: String = reading.chars().take(prefix_len).collect();
                    return Some((prefix, tail));
                }
            }
        }
        // Trailing hiragana run.
        let chars: Vec<char> = reading.chars().collect();
        let is_hira = |c: char| matches!(c, '\u{3041}'..='\u{3096}');
        let mut n = 0usize;
        while n < chars.len() && n < 8 && is_hira(chars[chars.len() - 1 - n]) {
            n += 1;
        }
        (n >= 2 && n < chars.len()).then(|| {
            let prefix: String = chars[..chars.len() - n].iter().collect();
            let tail: String = chars[chars.len() - n..].iter().collect();
            (prefix, tail)
        })
    }

    /// Per-character NLL of one candidate, or `None` when unscoreable.
    fn score_one(&mut self, reading: &str, text: &str) -> Option<f32> {
        let ctx = self.truncate_context_for_api();
        let converter = self.converters.kanji.as_ref()?;
        converter
            .score_candidates(reading, &ctx, std::slice::from_ref(&text.to_string()))
            .ok()?
            .into_iter()
            .next()
            .flatten()
    }

    /// Kana-level typo correction search: when the model can't convert
    /// `reading` cleanly (its best output passes a garbled stretch through as
    /// raw kana), guess what the user *meant* by trying small edits — single
    /// deletions and adjacent swaps inside the garbled region (the classic
    /// double-`っ` and transposition typos) — and converting each edited
    /// reading. Edits whose conversion comes out clean are returned as
    /// candidates carrying their edited reading (so learning records the
    /// corrected kana, not the typo). Costs up to `MAX_TYPO_VARIANTS` greedy
    /// model passes; only runs on the already-rare "conversion failed" path.
    fn kana_typo_corrections(
        &mut self,
        reading: &str,
        garbled_top: &str,
    ) -> Vec<AnnotatedCandidate> {
        const MAX_TYPO_VARIANTS: usize = 12;
        const MAX_CORRECTIONS: usize = 3;

        let chars: Vec<char> = reading.chars().collect();
        // Restrict edits to the garbled region so the variant count stays
        // bounded on long input; fall back to the whole reading if the run
        // can't be located (e.g. the top candidate IS the raw reading).
        let (start, end) = kashiko_engine::longest_passthrough_run(
            garbled_top,
            reading,
            UNCONVERTED_RUN_MIN,
        )
        .and_then(|run| {
            let byte_pos = reading.find(&run)?;
            let start = reading[..byte_pos].chars().count();
            Some((start, start + run.chars().count()))
        })
        .unwrap_or((0, chars.len()));

        let mut variants: Vec<String> = Vec::new();
        // Single-character deletions (stray extra kana, e.g. `だったっため`).
        for i in start..end {
            variants.push(
                chars[..i].iter().chain(&chars[i + 1..]).collect(),
            );
        }
        // Adjacent transpositions (e.g. `ようらな` → `ようなら`).
        for i in start..end.saturating_sub(1) {
            if chars[i] != chars[i + 1] {
                let mut v = chars.clone();
                v.swap(i, i + 1);
                variants.push(v.into_iter().collect());
            }
        }
        variants.dedup();
        variants.truncate(MAX_TYPO_VARIANTS);

        let ctx = self.truncate_context_for_api();
        let mut seen = HashSet::new();
        let mut corrections = Vec::new();
        for variant in variants {
            if corrections.len() >= MAX_CORRECTIONS {
                break;
            }
            let Some(converted) = self
                .run_kana_kanji_conversion(&variant, &ctx, 1)
                .into_iter()
                .next()
            else {
                continue;
            };
            if converted != variant
                && !kashiko_engine::has_long_unconverted_run(
                    &converted,
                    &variant,
                    UNCONVERTED_RUN_MIN,
                )
                // Corrections are merged after `build_conversion_candidates`,
                // so the filters there never see them. Checked against the
                // reading the user actually typed, not the edited variant.
                && !invents_ascii(reading, &converted)
                && seen.insert(converted.clone())
            {
                corrections.push(
                    AnnotatedCandidate::new(converted, CandidateSource::Model)
                        .with_reading(Some(variant)),
                );
            }
        }
        corrections
    }

    /// Undo the last conversion commit (reconversion): take the committed
    /// surface back out of the application and restore the reading in
    /// Conversion state, candidate window and all. The `DeleteSurrounding`
    /// action leads the batch so the frontend replaces the committed text
    /// with the restored marked text.
    ///
    /// Works reliably in standard Cocoa text views (Notes, TextEdit, …);
    /// apps without proper `replacementRange` support (some Electron /
    /// terminal apps) may ignore the deletion.
    pub(super) fn reconvert_last_commit(
        &mut self,
        reading: String,
        surface: String,
    ) -> EngineResult {
        self.converters.romaji.reset();
        self.input_buf.clear();
        self.input_buf.insert(&reading);
        self.live.text.clear();
        self.chunks.clear();
        let mut result = self.start_conversion(false, None, 0);
        result
            .actions
            .insert(0, EngineAction::DeleteSurrounding(surface));
        result
    }

    /// Whether deep-search work is scheduled or in flight. The server main
    /// loop uses this to decide between blocking on stdin and stepping the
    /// search, so key events are served between model calls.
    pub fn deep_search_pending(&self) -> bool {
        self.deep_search.is_some() || self.deep_running
    }

    /// Advance the pending deep search by one task (at most a couple of
    /// model calls). Returns `Some(result)` exactly once, when the search
    /// finishes and the candidate list has been updated — the result
    /// carries the actions to push to the frontend as an
    /// `update_candidates` notification. Returns `None` while idle, still
    /// working, or when the job was canceled.
    ///
    /// The job cancels itself silently whenever the conversion it was
    /// started for is no longer on screen — the user committed a candidate,
    /// kept typing, or dismissed the conversion — so no model time is spent
    /// on a conversion that is already over.
    /// How many candidates the *synchronous* pass asks the model for — the one
    /// the user waits on after pressing Space.
    ///
    /// Beam width is what this costs. Measured against the real model: a 9-wide
    /// beam takes 0.58s on a six-character reading and 4.7s on a thirty-four
    /// character one, while a greedy pass is 20–47ms at any length. That wide
    /// beam was the freeze — the whole of a 6.1s Space press was this one call,
    /// paid on the thread that answers key events.
    ///
    /// So the sync pass stays narrow and the width moves to the background
    /// worker, which cannot block anything: a thin list schedules a deep search
    /// automatically, and its findings arrive as `update_candidates` a moment
    /// later. Short readings keep a small beam because their homophones are
    /// what Space is usually for; long ones take the greedy conversion, which
    /// is also what live conversion was already showing.
    fn sync_candidate_count(&self, reading: &str) -> usize {
        /// Above this length the model gets one greedy pass and nothing more.
        const SHORT_ENOUGH_TO_BEAM: usize = 8;
        /// Beam width for a short reading: 0.2s, against 0.58s for nine.
        const SYNC_BEAM: usize = 3;
        if reading.chars().count() <= SHORT_ENOUGH_TO_BEAM {
            self.config.num_candidates.min(SYNC_BEAM)
        } else {
            1
        }
    }

    /// Hand the scheduled deep search over to be run elsewhere.
    ///
    /// Returns the job once, when one has been scheduled; the caller runs it
    /// (`run_deep_job`) off the main loop and gives the result back through
    /// [`Self::apply_deep_outcome`]. Nothing here calls the model, so this is
    /// always fast — which is the point: a wide beam over a long sentence takes
    /// seconds, and the loop that answers key events must not be inside it.
    pub fn take_deep_job(&mut self) -> Option<DeepSearchJob> {
        if self.deep_running {
            return None;
        }
        let request = self.deep_search.take()?;
        let job = self.build_deep_job(request)?;
        self.deep_running = true;
        Some(job)
    }

    /// Merge a finished deep search into the candidate list.
    ///
    /// `None` when the outcome is stale — the user committed, kept typing, or
    /// resized the segment while the search was running — or when there was
    /// nothing to add.
    pub fn apply_deep_outcome(&mut self, outcome: DeepSearchOutcome) -> Option<EngineResult> {
        self.deep_running = false;
        let stale = !matches!(
            &self.state,
            InputState::Conversion { split_index, .. }
                if *split_index == outcome.request.scope
        ) || self.input_buf.text != outcome.request.buffer;
        if stale {
            return None;
        }
        Some(self.finish_deep_job(outcome))
    }

    /// Stop the running deep search from finishing its work.
    ///
    /// The outcome would be discarded as stale anyway; this just avoids
    /// grinding a wide beam for a conversion that is already over.
    pub fn cancel_deep_search(&mut self) {
        self.deep_search = None;
        self.deep_cancel.store(true, Ordering::Relaxed);
    }

    /// Run the scheduled deep search here and now, blocking until it finishes.
    ///
    /// The frontend never uses this — it would be the freeze this design exists
    /// to remove — but it makes the search testable without threads.
    pub fn step_deep_search(&mut self) -> Option<EngineResult> {
        let job = self.take_deep_job()?;
        self.apply_deep_outcome(run_deep_job(job))
    }

    /// Build the task queue for a scheduled deep search. No model calls —
    /// just cheap variant generation. `None` when the conversion has already
    /// moved on or the reading isn't searchable.
    fn build_deep_job(&mut self, request: DeepSearchRequest) -> Option<DeepSearchJob> {
        let InputState::Conversion { split_index, .. } = &self.state else {
            return None;
        };
        if *split_index != request.scope || self.input_buf.text != request.buffer {
            return None;
        }
        let existing: Vec<String> = match &self.state {
            InputState::Conversion { candidates, .. } => {
                candidates.candidates().iter().map(|c| c.text.clone()).collect()
            }
            _ => return None,
        };
        // The scoped reading under conversion, raw and ASCII-stripped forms.
        let original = request.reading.clone();
        let kind = request.kind;
        let cleaned: String = original
            .chars()
            .filter(|c| !c.is_ascii_alphabetic())
            .collect();
        if !kashiko_engine::contains_kana(&cleaned) {
            return None;
        }
        self.ensure_kanji_converter();

        // Pre-gate cap. Higher than what reaches the window, because
        // `gate_deep_findings` removes the ones that read as nonsense — the
        // search needs room to find the good ones before that selection. Each
        // press of the aux button raises it, so asking again keeps finding new
        // things instead of re-running the same search.
        let max_total_results = 10 + 6 * request.round;
        // Extra beam width per round, spent the same way.
        let widen = request.round;
        let mut queue: VecDeque<DeepTask> = VecDeque::new();
        let mut uniq: HashSet<String> = HashSet::new();
        let mut push_convert = |queue: &mut VecDeque<DeepTask>, variant: String, beam: usize, cap: usize| {
            if kashiko_engine::contains_kana(&variant)
                && !variant.chars().any(|c| c.is_ascii_alphabetic())
                && uniq.insert(variant.clone())
            {
                queue.push_back(DeepTask::Convert { variant, beam, cap });
            }
        };

        match kind {
            DeepSearchKind::TypoCorrection => {
                const MAX_KEYBOARD_VARIANTS: usize = 14;
                /// Chars of context on each side of the typo point included
                /// in the keyboard-edit region.
                const REGION_MARGIN: usize = 4;

                // Typo region *in the original reading* (stray ASCII
                // included): around the first ASCII char when the typo was
                // romaji leftovers, else the garbled run of the current top
                // candidate, else the whole reading.
                let orig_chars: Vec<char> = original.chars().collect();
                let (start, end) =
                    match original.chars().position(|c| c.is_ascii_alphabetic()) {
                        Some(p) => (
                            p.saturating_sub(REGION_MARGIN),
                            (p + 1 + REGION_MARGIN).min(orig_chars.len()),
                        ),
                        None => existing
                            .first()
                            .and_then(|top| {
                                let run = kashiko_engine::longest_passthrough_run(
                                    top,
                                    &original,
                                    UNCONVERTED_RUN_MIN,
                                )?;
                                let byte_pos = original.find(&run)?;
                                let s = original[..byte_pos].chars().count();
                                Some((s, s + run.chars().count()))
                            })
                            .unwrap_or((0, orig_chars.len())),
                    };
                let prefix: String = orig_chars[..start].iter().collect();
                let region: String = orig_chars[start..end].iter().collect();
                let suffix: String = orig_chars[end..].iter().collect();

                // Task order, most → least promising: completions (finish
                // the interrupted keystroke), keyboard edits of the region,
                // coarse kana edits, and finally base-beam extras.
                for v in kashiko_engine::typo::romaji_completion_variants(&original) {
                    push_convert(&mut queue, v, 3 + widen, 2);
                }
                for (i, kb) in
                    kashiko_engine::typo::keyboard_typo_variants(&region, MAX_KEYBOARD_VARIANTS)
                        .into_iter()
                        .enumerate()
                {
                    let beam = if i < 4 { 3 + widen } else { 1 + widen };
                    push_convert(&mut queue, format!("{prefix}{kb}{suffix}"), beam, 2);
                }
                let chars: Vec<char> = cleaned.chars().collect();
                let ascii_before_start = orig_chars[..start]
                    .iter()
                    .filter(|c| c.is_ascii_alphabetic())
                    .count();
                let cstart = start - ascii_before_start;
                let cend = (cstart + (end - start)).min(chars.len());
                for i in cstart..cend.min(chars.len()) {
                    push_convert(
                        &mut queue,
                        chars[..i].iter().chain(&chars[i + 1..]).collect(),
                        1,
                        1,
                    );
                }
                let swap_end = cend.min(chars.len()).saturating_sub(1);
                for i in cstart..swap_end {
                    if chars[i] != chars[i + 1] {
                        let mut v = chars.clone();
                        v.swap(i, i + 1);
                        push_convert(&mut queue, v.into_iter().collect(), 1, 1);
                    }
                }
                push_convert(&mut queue, cleaned, 5 + widen, 2 + widen);
            }
            DeepSearchKind::MoreAlternatives => {
                // Wide main-model beam over the whole reading, then tail
                // alternatives (`かき` → 書き/下記/柿) with the prefix held
                // fixed.
                let tail_split = self.split_tail_for_alternatives(&cleaned);
                // Cap raised to the beam width: the results are already
                // computed, and keeping only four of an eight-wide beam threw
                // away homophones the sync pass no longer looks for.
                push_convert(&mut queue, cleaned.clone(), 8 + 4 * widen, 8 + 4 * widen);
                if let Some((prefix_reading, tail_reading)) = tail_split {
                    queue.push_back(DeepTask::TailBeam {
                        prefix_reading,
                        tail_reading,
                        beam: 6 + 3 * widen,
                    });
                }
            }
        }

        let reading = original
            .chars()
            .filter(|c| !c.is_ascii_alphabetic())
            .collect();
        let converter = self.converters.kanji.clone()?;
        // A fresh flag per job: the previous one may still belong to a job in
        // flight, and cancelling that must not cancel this.
        self.deep_cancel = Arc::new(AtomicBool::new(false));
        // Resolved here rather than in the worker: it comes from the live
        // conversion's chunk cache, which is engine state.
        let cached_prefix = queue
            .iter()
            .find_map(|t| match t {
                DeepTask::TailBeam { prefix_reading, .. } => Some(prefix_reading.chars().count()),
                _ => None,
            })
            .and_then(|n| self.cached_prefix_conversion(n))
            .and_then(|v| v.into_iter().next());
        (!queue.is_empty()).then(|| DeepSearchJob {
            request,
            reading,
            queue,
            found: Vec::new(),
            seen: existing.into_iter().collect(),
            max_results: max_total_results,
            ctx: self.truncate_context_for_api(),
            max_ctx: self.config.max_api_context_len,
            cached_prefix,
            quality_best: self.last_quality_best,
            converter,
            cancel: self.deep_cancel.clone(),
        })
    }


    /// Whether the current input buffer (the not-yet-converted remainder after
    /// a partial commit) is worth converting as its own segment, judged by the
    /// model itself: a single greedy pass must produce something different
    /// from the raw reading and free of long passed-through kana runs.
    ///
    /// Costs one extra greedy inference (~tens of ms) — an accepted trade for
    /// the segment-by-segment conversion flow.
    fn remainder_converts(&mut self) -> bool {
        let reading = self.input_buf.text.clone();
        if reading.is_empty() || !kashiko_engine::contains_kana(&reading) {
            return false;
        }
        self.ensure_kanji_converter();
        let ctx = self.truncate_context_for_api();
        self.run_kana_kanji_conversion(&reading, &ctx, 1)
            .first()
            .is_some_and(|out| {
                out != &reading
                    && !kashiko_engine::has_long_unconverted_run(
                        out,
                        &reading,
                        UNCONVERTED_RUN_MIN,
                    )
            })
    }

    /// Build conversion candidates for a reading from multiple sources.
    ///
    /// Combines learning cache, dictionaries, and model inference results
    /// with deduplication. Uses dynamic candidate count based on input token
    /// count for performance.
    ///
    /// Priority: Learning → User Dictionary → Model → System Dictionary → Fallback
    ///
    /// `skip_learning` suppresses the learning-cache step (1). Used by the Tab
    /// key path so users can escape a noisy learning history without losing
    /// access to dictionary/model candidates.
    /// `allow_latin` enables the raw-keystroke latin-word candidate (step 10c).
    /// Only the whole-buffer calls pass `true`; a prefix-scoped retry must not
    /// offer a candidate covering text outside its scope.
    pub(super) fn build_conversion_candidates(
        &mut self,
        reading: &str,
        num_candidates: usize,
        skip_learning: bool,
        cached_model: Option<Vec<String>>,
        allow_latin: bool,
    ) -> Vec<AnnotatedCandidate> {
        // Try to initialize the kanji converter, but don't bail out if it
        // fails — symbol-only inputs (e.g. `。。。`) don't need the model and
        // we still want to produce dictionary, rewriter, and fallback candidates.
        // run_kana_kanji_conversion handles the converter-missing case.
        if self.converters.kanji.is_none()
            && let Err(e) = self.init_kanji_converter()
        {
            debug!("Failed to initialize kanji converter: {}", e);
        }

        // The cached live-typing conversion (chunked greedy) is kept as the
        // guaranteed-first model candidate, but a fresh whole-reading pass
        // still runs to produce *alternatives* — cache-only conversion left
        // long input with a single AI candidate and nothing to pick from.
        // The strategy layer already downshifts long input to the light
        // model, so the extra pass stays within the Space-key latency budget.
        let candidates = {
            let api_context = self.truncate_context_for_api();
            let fresh = self.run_kana_kanji_conversion(reading, &api_context, num_candidates);
            match cached_model {
                Some(mut cached) => {
                    for c in fresh {
                        if !cached.contains(&c) {
                            cached.push(c);
                        }
                    }
                    cached
                }
                None => fresh,
            }
        };

        let hiragana = reading.to_string();
        let katakana = kashiko_engine::hiragana_to_katakana(reading);

        // Priority: Learning → User Dictionary → Model → System Dictionary → Fallback
        let mut builder = CandidateBuilder::new();

        // 1. Learning cache candidates (highest priority).
        //    Force-inserted so they win against duplicate text from later sources.
        //    Skipped when the caller asks for a learning-free conversion (Tab key).
        if !skip_learning {
            for c in self.lookup_learning_candidates(reading) {
                // Exact matches have reading == input reading; use None to avoid redundancy
                let cand_reading = c.reading.filter(|r| r != reading);
                builder.push_force(
                    AnnotatedCandidate::new(c.text, CandidateSource::Learning)
                        .with_reading(cand_reading),
                );
            }
        }

        // 2. Dictionary candidates (user dict first, then system dict)
        let dict_results = self.search_dictionaries(reading, usize::MAX);
        // Insert user dictionary entries at the top (after learning)
        for ac in &dict_results {
            if ac.source == CandidateSource::UserDictionary {
                builder.push(ac.clone());
            }
        }

        // 3. Model inference results
        if candidates.is_empty() {
            // In emoji mode, defer the literal-fallback decision until
            // after rewriters have run — otherwise `:smile` would be
            // pinned to the top of the candidate list as a Fallback
            // and outrank the 😄 we surface in step 5/6.
            if builder.is_empty() && self.input_mode != InputMode::Emoji {
                builder.push(AnnotatedCandidate::new(
                    hiragana.clone(),
                    CandidateSource::Fallback,
                ));
            }
        } else {
            for text in candidates {
                builder.push(AnnotatedCandidate::new(text, CandidateSource::Model));
            }
        }

        // 4. System dictionary candidates (from search_dictionaries result)
        for ac in dict_results {
            if ac.source == CandidateSource::Dictionary {
                builder.push(ac);
            }
        }

        // 5/6. Hiragana/katakana fallback + rewriter variants.
        //
        // In emoji mode we surface ONLY the rewriter (i.e. EmojiRewriter)
        // candidates — Slack's emoji picker shows emojis and nothing
        // else, and that's the mental model the user wants here.
        // No literal `:smile` / `:xyz` fallback in the candidate list:
        // if nothing matches, the picker is just empty. (Enter on a
        // no-match query in Composing still commits the buffer
        // literal via `commit_composing`; that's the escape hatch.)
        // Non-emoji modes keep the original order so existing IME
        // behavior is untouched.
        let rewriter_variants = self
            .converters
            .rewriters
            .rewrite_all(&[reading.to_string()]);
        if self.input_mode == InputMode::Emoji {
            for (variant, description) in rewriter_variants {
                builder.push(
                    AnnotatedCandidate::new(variant, CandidateSource::Rewriter)
                        .with_description(description),
                );
            }
        } else {
            // Rewriters operate on the user's typed input (the reading
            // itself). Running them on dictionary/model/fallback
            // candidates produces unrelated noise (e.g. a dictionary
            // entry of `,` for some reading would generate `、`/`，`
            // variants the user never asked for; a learning entry `アト`
            // pulled by prefix lookup on `あ` would emit `ｱﾄ`).
            for (variant, description) in rewriter_variants {
                builder.push(
                    AnnotatedCandidate::new(variant, CandidateSource::Rewriter)
                        .with_description(description),
                );
            }
            // Ensure fallback candidates (hiragana / katakana) are strictly at the end.
            // Even if the model or dictionary returned them earlier, move them here.
            builder.candidates.retain(|c| c.text != hiragana && c.text != katakana);
            builder.candidates.push(AnnotatedCandidate::new(hiragana.clone(), CandidateSource::Fallback));
            builder.candidates.push(AnnotatedCandidate::new(katakana.clone(), CandidateSource::Fallback));
            builder.seen.insert(hiragana.to_string());
            builder.seen.insert(katakana.to_string());
        }

        // 7. Enrich Fallback candidates whose text is a known symbol with
        //    its description (mirrors the relevant slice of mozc's
        //    `AddDescForCurrentCandidates`). Restricted to Fallback so the
        //    AI/Dict/Learning paths don't pick up unwanted labels — e.g.
        //    the model returning `金` for `きん` should NOT inherit mozc's
        //    "部首" annotation. Typed-symbol input still gets annotated:
        //    pressing `「` produces a Fallback candidate `「`, which here
        //    picks up "始めかぎ括弧".
        for c in &mut builder.candidates {
            if c.source == CandidateSource::Fallback
                && c.description.is_none()
                && let Some(desc) = kashiko_engine::symbol_description(&c.text)
            {
                c.description = Some(desc.to_string());
            }
        }

        // 8. Attach mozc-style width annotations (`[全]ひらがな`,
        //    `[全]カタカナ`, `[半]カタカナ`) to any pure-kana candidate that
        //    still has no description. This catches `あ`/`ア` candidates that
        //    arrived via the Model or Fallback paths and were deduped against
        //    the rewriter's already-labelled variants.
        for c in &mut builder.candidates {
            if c.description.is_none()
                && let Some(desc) = width_annotation(&c.text)
            {
                c.description = Some(desc.to_string());
            }
        }

        // 9. Filter out disproportionately long candidates when the reading is
        //    short. Fallback candidates (hiragana/katakana) are always preserved.
        {
            use super::input::{LONG_INPUT_THRESHOLD, SHORT_INPUT_MAX_CANDIDATE_LEN};
            let reading_len = reading.chars().count();
            if reading_len < LONG_INPUT_THRESHOLD {
                builder.candidates.retain(|c| {
                    c.source == CandidateSource::Fallback
                        || c.text.chars().count() <= SHORT_INPUT_MAX_CANDIDATE_LEN
                });
            }
        }

        // 10. Drop candidates containing a character virtually no font can
        //     render (rare historical kana from the system dictionary, or a
        //     stray model control token) — regardless of source, so a
        //     dictionary/model result can't surface the same "missing
        //     glyph" box that learning entries are already filtered for
        //     (see `LearningCache::is_learnable`).
        builder
            .candidates
            .retain(|c| !kashiko_engine::has_unrenderable_char(&c.text));

        // 10b. Half-width katakana variants are never shown in the conversion
        //      list (user preference — they only add noise next to the
        //      full-width forms). Matched by the rewriter's annotation and,
        //      as a fallback, by codepoint (U+FF61–FF9F) in case a half-width
        //      form arrives unlabelled from the dictionary or model.
        builder.candidates.retain(|c| {
            c.description.as_deref() != Some("[半]カタカナ")
                && !(!c.text.is_empty()
                    && c.text.chars().all(|ch| ('\u{FF61}'..='\u{FF9F}').contains(&ch)))
        });

        // 10a2. Drop candidates the user deleted with the row's ✕.
        if let Some(cache) = self.learning.as_ref() {
            builder
                .candidates
                .retain(|c| !cache.is_suppressed(reading, &c.text));
        }

        // 10b2. Drop candidates that invent ASCII the reading never contained.
        //       A kana-only reading converting to `な2回慰安ありますか` is the
        //       model hallucinating a digit: nothing in `なにかいいあんありますか`
        //       could produce a `2`. Latin/digit output is only legitimate when
        //       the user actually typed those characters, which the reading
        //       then carries — so comparing against the reading is enough.
        //       Runs *before* the latin-word step below, whose candidate comes
        //       from the keystrokes rather than the reading and is exempt.
        {
            // A counted quantity's digit is not an invention — `にげん` really
            // does convert to `2限`. Without this exemption the rule deleted
            // exactly the candidates step 10c2 exists to supply, and the model's
            // own `2階` for `にかい` only survived because the live-conversion
            // pin put it back afterwards.
            let counted = kashiko_engine::counter::candidates(reading);
            let invents_ascii =
                |text: &str| !counted.iter().any(|c| c == text) && invents_ascii(reading, text);
            // Fallback and Rewriter output is derived from the reading by rule
            // (width/case variants such as `abc` → `ABC`/`ａｂｃ`), so only
            // free-form model/dictionary output is checked.
            builder.candidates.retain(|cand| {
                matches!(
                    cand.source,
                    CandidateSource::Fallback | CandidateSource::Rewriter
                ) || !invents_ascii(&cand.text)
            });
        }

        // 10c. The raw keystrokes as a latin word (`WordPress` out of the
        //      buffer `WおrdPレッ`) — no other source can reconstruct it from
        //      the kana. Inserted *among* the real candidates at a position
        //      set by its confidence score, so a proven word leads while a
        //      weaker guess sits a couple of rows down. Only for whole-buffer
        //      conversions: the keystrokes cover the entire composition, so
        //      injecting them into a prefix-scoped list would commit text
        //      outside that scope.
        if allow_latin
            && let Some((latin, score)) = self.latin_word_candidate()
            && !builder.candidates.iter().any(|c| c.text == latin)
        {
            let at = Self::latin_insert_index(score, builder.candidates.len());
            builder.seen.insert(latin.clone());
            builder.candidates.insert(
                at,
                AnnotatedCandidate::new(latin, CandidateSource::UserDictionary),
            );
        }

        // 10b3. On a short reading, drop pure-kana candidates that mix the two
        //       scripts (`にゲン`, `コバやし`, `カンじ`). Those are the model
        //       giving up half way through, and with nothing deleted on a short
        //       reading (see `SHORT_READING_CHARS`) they would otherwise take
        //       slots from the homophones that reading exists for. The same
        //       shape is legitimate on longer input — `メールする` is katakana
        //       plus a hiragana verb ending — so the rule stops at four
        //       characters, and the two kana fallbacks are exempt.
        if reading.chars().count() <= 4 {
            builder.candidates.retain(|c| {
                c.source == CandidateSource::Fallback
                    || !kashiko_engine::is_all_kana(&c.text)
                    || kashiko_engine::is_pure_hiragana(&c.text)
                    || kashiko_engine::is_pure_full_katakana(&c.text)
            });
        }

        // 10c2. Counted quantities written with a digit (`にげん` → `2限`).
        //       Nothing upstream can produce these: the dictionary is keyed on
        //       whole readings and has no `にげん`, and the model's beam only
        //       sometimes stumbles on the digit form — it finds `2階` for
        //       `にかい` and misses `2限` entirely. Measured on the real model,
        //       `2限` scores 1.03 against 0.42 for the best candidate that *was*
        //       offered and above 2.0 for every other one, so the right answer
        //       simply was not on the list.
        //
        //       Inserted after the invented-character filter, which would drop
        //       them for spelling a digit the kana reading never had, and as
        //       `UserDictionary` so reranking leaves them alone: with no
        //       sentence around it a counted quantity scores no better than the
        //       homophones it is competing with.
        {
            let at = builder
                .candidates
                .iter()
                .position(|c| c.source == CandidateSource::Fallback)
                .unwrap_or(builder.candidates.len())
                .min(1);
            let new: Vec<AnnotatedCandidate> = kashiko_engine::counter::candidates(reading)
                .into_iter()
                // Against the live list, not `seen`: `seen` also remembers text
                // an earlier filter deliberately removed, which is how `2限`
                // ended up silently skipped here after 10b2 dropped the model's
                // copy of it.
                .filter(|t| !builder.candidates.iter().any(|c| c.text == *t))
                .filter(|t| {
                    self.learning
                        .as_ref()
                        .is_none_or(|cache| !cache.is_suppressed(reading, t))
                })
                .map(|t| AnnotatedCandidate::new(t, CandidateSource::UserDictionary))
                .collect();
            for t in &new {
                builder.seen.insert(t.text.clone());
            }
            builder.candidates.splice(at..at, new);
        }

        // 10d. The English spelling of a listed katakana loanword
        //      (`てんぷれーと` → `template`). Nothing upstream can produce it:
        //      the dictionary and the model only ever emit Japanese script, so
        //      a borrowed word's own spelling is unreachable without a table.
        //
        //      Placed second, directly under the katakana form, so it is
        //      visible without displacing whatever the conversion chain
        //      decided, and always above the kana fallbacks. Inserted here —
        //      after the ASCII-invention filter (10b2), which would otherwise
        //      throw it out for spelling letters the kana reading never had —
        //      and as `UserDictionary`, which reranking leaves alone: an
        //      English word scored as if it were Japanese reads badly, and
        //      that is a fact about the scorer, not about the candidate.
        if let Some(english) = kashiko_engine::loanword::english_for(reading) {
            // A matched reading is a single borrowed word, which pins down what
            // the right answers are: the katakana form and the English spelling.
            // That is enough to clear out the half-converted kana the model
            // offers here — `てんプレート`, `テンプレーと`, `リポジとり` — which
            // score close enough to the katakana form to pass the quality gate
            // and are never what anyone meant. Kept narrow on purpose: the same
            // shape is legitimate in general (`メールする` is katakana plus a
            // hiragana verb ending), so only readings the table recognises as
            // one whole word are treated this way.
            let canonical = kashiko_engine::hiragana_to_katakana(reading);
            builder.candidates.retain(|c| {
                c.source != CandidateSource::Model
                    || c.text == canonical
                    || !kashiko_engine::is_all_kana(&c.text)
            });
            // The katakana form leads: for a borrowed word it is the answer, and
            // the model often has it only as the kana fallback at the bottom.
            if let Some(at) = builder.candidates.iter().position(|c| c.text == canonical)
                && at > 0
            {
                let c = builder.candidates.remove(at);
                builder.candidates.insert(0, c);
            }
            // English directly under it, and always above the kana fallbacks.
            if !builder.candidates.iter().any(|c| c.text == english)
                // The ✕ has to work here too, and 10a2 already ran.
                && self
                    .learning
                    .as_ref()
                    .is_none_or(|cache| !cache.is_suppressed(reading, english))
            {
                let fallback_at = builder
                    .candidates
                    .iter()
                    .position(|c| c.source == CandidateSource::Fallback)
                    .unwrap_or(builder.candidates.len());
                let at = fallback_at.min(1);
                builder.seen.insert(english.to_string());
                builder.candidates.insert(
                    at,
                    AnnotatedCandidate::new(english.to_string(), CandidateSource::UserDictionary),
                );
            }
        }

        // 11. Drop low-quality model guesses: when the user mistypes mid-phrase
        //     (e.g. `できないようらな…`), the model often emits a candidate that
        //     just passes the garbled stretch through as raw hiragana
        //     (`できないようらな消す`). Those are meaningless and were the ones
        //     leaking into the list despite the learning filter. Only Model-
        //     source candidates are eligible, and only when a cleaner
        //     alternative survives — never leave the list empty.
        {
            let clean_exists = builder.candidates.iter().any(|c| {
                c.source != CandidateSource::Model
                    || !kashiko_engine::has_long_unconverted_run(
                        &c.text,
                        reading,
                        UNCONVERTED_RUN_MIN,
                    )
            });
            if clean_exists {
                builder.candidates.retain(|c| {
                    c.source != CandidateSource::Model
                        || !kashiko_engine::has_long_unconverted_run(
                            &c.text,
                            reading,
                            UNCONVERTED_RUN_MIN,
                        )
                });
            }
        }

        builder.into_candidates()
    }

    /// Look up learning cache candidates for a reading (exact + prefix match, max 3).
    ///
    /// Returns candidates from the learning cache suitable for auto-suggest display.
    pub(super) fn lookup_learning_candidates(&self, reading: &str) -> Vec<Candidate> {
        let Some(cache) = &self.learning else {
            return vec![];
        };
        let mut candidates: Vec<Candidate> = Vec::new();
        let mut seen = HashSet::new();
        let label = CandidateSource::Learning.label().to_string();

        // Exact match
        for (surface, _score) in cache.lookup(reading) {
            if candidates.len() >= MAX_LEARNING_CANDIDATES {
                break;
            }
            if surface.chars().count() >= LearningCache::MAX_LEARN_SURFACE_CHARS {
                continue;
            }
            if seen.insert(surface.clone()) {
                candidates.push(Candidate {
                    text: surface,
                    reading: Some(reading.to_string()),
                    source_label: Some(label.clone()),
                    description: None,
                });
            }
        }

        // Prefix match (predictive)
        for (full_reading, surface, _score) in cache.prefix_lookup(reading) {
            if candidates.len() >= MAX_LEARNING_CANDIDATES {
                break;
            }
            if full_reading == reading {
                continue;
            }
            if surface.chars().count() >= LearningCache::MAX_LEARN_SURFACE_CHARS {
                continue;
            }
            if seen.insert(surface.clone()) {
                candidates.push(Candidate {
                    text: surface,
                    reading: Some(full_reading),
                    source_label: Some(label.clone()),
                    description: None,
                });
            }
        }

        candidates
    }

    /// Look up dictionary candidates for a reading (1 page, for live conversion display)
    ///
    /// Searches user dictionary first, then system dictionary.
    #[allow(dead_code)]
    pub(super) fn lookup_dict_candidates(&self, reading: &str) -> Vec<Candidate> {
        self.search_dictionaries(reading, CandidateList::DEFAULT_PAGE_SIZE)
            .into_iter()
            .map(|ac| Candidate {
                text: ac.text,
                reading: Some(reading.to_string()),
                source_label: Some(ac.source.label().to_string()),
                description: None,
            })
            .collect()
    }

    /// Build rule-based rewriter variants for the reading itself (e.g. for
    /// symbol input `「` → `『`, `【`, `（`, ...). Used in the auto-suggest path
    /// so users see mozc-style symbol variants without pressing Space first.
    pub(super) fn lookup_rewriter_variants(&self, reading: &str) -> Vec<Candidate> {
        let source_label = CandidateSource::Rewriter.label().to_string();
        self.converters
            .rewriters
            .rewrite_all(&[reading.to_string()])
            .into_iter()
            .map(|(text, description)| Candidate {
                text,
                reading: Some(reading.to_string()),
                source_label: Some(source_label.clone()),
                description,
            })
            .collect()
    }

    /// Merge two candidate lists with deduplication
    /// Primary candidates come first, then secondary candidates that aren't duplicates
    pub(super) fn merge_candidates_dedup(
        primary: Vec<String>,
        secondary: Vec<String>,
        max_candidates: usize,
    ) -> Vec<String> {
        let mut seen = HashSet::new();
        primary
            .into_iter()
            .chain(secondary)
            .filter(|c| seen.insert(c.clone()))
            .take(max_candidates)
            .collect()
    }

    /// Process key in conversion state
    pub(super) fn process_key_conversion(&mut self, key: &KeyEvent) -> EngineResult {
        match key.keysym {
            Keysym::RETURN => self.commit_conversion(),
            Keysym::ESCAPE => self.cancel_conversion(),
            Keysym::SPACE | Keysym::DOWN | Keysym::TAB => self.next_candidate(),
            Keysym::UP => self.prev_candidate(),
            Keysym::PAGE_DOWN => self.next_candidate_page(),
            Keysym::PAGE_UP => self.prev_candidate_page(),
            // Plain Left/Right slide the focused segment through the reading;
            // Shift+Left/Right resize it (fixing a wrong word split).
            Keysym::LEFT if key.modifiers.shift_key => self.resize_conversion(-1),
            Keysym::RIGHT if key.modifiers.shift_key => self.resize_conversion(1),
            Keysym::LEFT => self.move_segment_focus(-1),
            Keysym::RIGHT => self.move_segment_focus(1),
            Keysym::BACKSPACE => self.backspace_conversion(),
            _ => {
                // Ctrl+N / Ctrl+P: emacs-style candidate navigation
                if key.modifiers.control_key && !key.modifiers.alt_key {
                    match key.keysym {
                        Keysym::KEY_N | Keysym::KEY_N_UPPER => return self.next_candidate(),
                        Keysym::KEY_P | Keysym::KEY_P_UPPER => return self.prev_candidate(),
                        _ => {}
                    }
                }

                // Check for digit selection (1-9)
                if let Some(digit) = key.keysym.digit_value() {
                    return self.select_candidate_by_digit(digit);
                }

                // Any printable character: commit current conversion and start new input
                if let Some(ch) = key.to_char()
                    && !key.modifiers.control_key
                    && !key.modifiers.alt_key
                {
                    return self.commit_conversion_and_continue(ch);
                }

                EngineResult::not_consumed()
            }
        }
    }

    /// Get selected text and reading from conversion state, or None if not in conversion
    fn selected_conversion_info(&self) -> Option<(String, Option<String>)> {
        match &self.state {
            InputState::Conversion { candidates, .. } => {
                let text = candidates.selected_text().unwrap_or("").to_string();
                let reading = candidates.selected().and_then(|c| c.reading.clone());
                Some((text, reading))
            }
            _ => None,
        }
    }

    /// Record a conversion selection in the learning cache.
    pub(super) fn record_learning(&mut self, reading: &str, surface: &str) {
        if self.secure_input {
            return;
        }
        if let Some(cache) = &mut self.learning {
            cache.record(reading, surface);
        }
    }

    /// Commit the current conversion
    fn commit_conversion(&mut self) -> EngineResult {
        let Some((text, reading)) = self.selected_conversion_info() else {
            return EngineResult::not_consumed();
        };

        if text.is_empty() {
            return EngineResult::consumed();
        }

        // Text after the focused segment. When the focus was moved with the
        // arrow keys the tail is already converted (`tail_text`) and commits
        // with everything else; otherwise (the typo split) it stays raw and
        // goes back into composing for the user to fix.
        let converted_tail = self.tail_text.take();
        let remaining = if let InputState::Conversion { split_index, .. } = self.state {
            let full_len = self.input_buf.text.chars().count();
            if split_index < full_len && converted_tail.is_none() {
                Some(self.input_buf.text.chars().skip(split_index).collect::<String>())
            } else {
                None
            }
        } else {
            None
        };

        // Record learning for the committed segment. A kana reading of words
        // kept in Latin script is not a conversion of `reading` (which holds
        // the Latin), so it teaches the English detection instead.
        let kana_alternative = self.mixed.kana_alternatives.contains(&text);
        if self.input_mode != InputMode::Emoji && !kana_alternative {
            if let Some(reading) = &reading {
                self.record_learning(reading, &text);
            }
        }
        self.learn_from_mixed_commit(&text);

        // Segments settled by sliding the focus right (Left/Right) commit
        // together with the focused one, ahead of it.
        let decided_text = self.take_decided_text();
        let text = format!(
            "{decided_text}{text}{}",
            converted_tail.as_deref().unwrap_or("")
        );

        // Remember full commits for one key so Ctrl+Backspace can take the
        // text back into conversion mode (undo commit). Partial commits are
        // excluded — their remainder is already back in composition. The
        // reading has to cover everything committed, decided segments
        // included, or the undo would restore only the last segment.
        if remaining.is_none() {
            let covered: String = match self.state {
                InputState::Conversion { split_index, .. } => {
                    self.input_buf.text.chars().take(split_index).collect()
                }
                _ => reading.clone().unwrap_or_default(),
            };
            if !covered.is_empty() {
                self.last_commit = Some((covered, text.clone()));
            }
        }

        if let Some(remaining) = remaining {
            // Partial commit: commit selected segment, keep remainder for further conversion
            self.state = InputState::Empty;
            self.input_buf.set_text(remaining);
            self.live.text.clear();
            self.chunks.clear();

            // Chained segment conversion: ask the model whether the remainder
            // converts cleanly on its own. If it does, jump straight into
            // Conversion state for it (the segment-by-segment flow: Enter
            // commits `テーマに沿って`, and `できている` immediately shows its
            // own candidates). If the model can't make sense of it (a typo
            // tail like `ようらなけす`), fall back to plain composing so the
            // user can edit the raw kana. The extra greedy model pass here is
            // an accepted latency cost.
            let follow_up = if self.remainder_converts() {
                self.start_conversion(false, None, 0)
            } else {
                self.refresh_input_state()
            };

            let mut result = EngineResult::consumed()
                .with_action(EngineAction::Commit(text))
                .with_action(EngineAction::HideCandidates);
            result.actions.extend(follow_up.actions);
            result
        } else {
            // Full commit: no remaining text
            self.state = InputState::Empty;
            self.input_buf.text.clear();
            self.exit_emoji_mode();

            EngineResult::consumed()
                .with_action(EngineAction::UpdatePreedit(Preedit::new()))
                .with_action(EngineAction::HideCandidates)
                .with_action(EngineAction::HideAuxText)
                .with_action(EngineAction::Commit(text))
        }
    }

    /// Commit current conversion and then process a new character as fresh input
    fn commit_conversion_and_continue(&mut self, ch: char) -> EngineResult {
        let Some((mut text, reading)) = self.selected_conversion_info() else {
            return EngineResult::not_consumed();
        };

        if let InputState::Conversion { split_index, .. } = self.state {
            let full_len = self.input_buf.text.chars().count();
            if split_index < full_len {
                let remaining: String = self.input_buf.text.chars().skip(split_index).collect();
                text.push_str(&remaining);
            }
        }

        if self.input_mode != InputMode::Emoji
            && let Some(reading) = &reading
        {
            self.record_learning(reading, &text);
        }

        // Settled segments (from sliding the focus) commit ahead of this text.
        let text = format!("{}{}", self.take_decided_text(), text);

        self.state = InputState::Empty;
        self.input_buf.text.clear();
        self.exit_emoji_mode();

        // Start new input with the character
        let new_input_result = self.start_input(ch);

        // Combine: commit first, then new input actions
        let mut result = EngineResult::consumed()
            .with_action(EngineAction::Commit(text))
            .with_action(EngineAction::HideCandidates);
        result.actions.extend(new_input_result.actions);
        result
    }

    /// Cancel conversion and return to hiragana
    pub(super) fn cancel_conversion(&mut self) -> EngineResult {
        if !matches!(self.state, InputState::Conversion { .. }) {
            return EngineResult::not_consumed();
        }
        let reading = self.input_buf.text.clone();

        if reading.is_empty() {
            self.state = InputState::Empty;
            self.input_buf.clear();
            return EngineResult::consumed()
                .with_action(EngineAction::UpdatePreedit(Preedit::new()))
                .with_action(EngineAction::HideCandidates)
                .with_action(EngineAction::HideAuxText);
        }

        // Reset romaji converter and set output to reading
        self.converters.romaji.reset();
        // We need to push each character to rebuild the state
        for ch in reading.chars() {
            self.converters.romaji.push(ch);
        }

        // Sync input_buf to match the rebuilt romaji output.
        // This ensures uncompleted characters (like 'k') remain in the romaji buffer
        // rather than being permanently committed to input_buf.
        self.input_buf.text = self.converters.romaji.output().to_string();
        self.input_buf.cursor_pos = self.input_buf.text.chars().count();

        let preedit = self.set_composing_state();

        EngineResult::consumed()
            .with_action(EngineAction::UpdatePreedit(preedit))
            .with_action(EngineAction::HideCandidates)
            .with_action(EngineAction::UpdateAuxText(self.format_aux_composing()))
    }

    /// Navigate candidates with the given operation, then update preedit
    fn navigate_candidate(&mut self, op: impl FnOnce(&mut CandidateList) -> bool) -> EngineResult {
        let (selected_text, candidates) = {
            let Some(candidates) = self.state.candidates_mut() else {
                return EngineResult::not_consumed();
            };
            op(candidates);
            let text = candidates.selected_text().unwrap_or("").to_string();
            (text, candidates.clone())
        };
        self.update_conversion_preedit(&selected_text, &candidates)
    }

    /// Select next candidate
    fn next_candidate(&mut self) -> EngineResult {
        self.navigate_candidate(CandidateList::move_next)
    }

    /// Select previous candidate
    fn prev_candidate(&mut self) -> EngineResult {
        self.navigate_candidate(CandidateList::move_prev)
    }

    /// Go to next candidate page
    fn next_candidate_page(&mut self) -> EngineResult {
        self.navigate_candidate(CandidateList::next_page)
    }

    /// Go to previous candidate page
    fn prev_candidate_page(&mut self) -> EngineResult {
        self.navigate_candidate(CandidateList::prev_page)
    }

    /// Word-boundary positions (in chars, ascending; the last one equals the
    /// reading length) from greedy longest-match dictionary segmentation.
    /// Consecutive characters with no dictionary entry (okurigana tails,
    /// interjections, ...) are merged into a single segment so the
    /// boundaries stay word-sized instead of degrading to per-character.
    pub(super) fn word_boundaries(&self, reading: &str) -> Vec<usize> {
        let chars: Vec<char> = reading.chars().collect();
        let longest_at = |pos: usize| -> usize {
            let rest: String = chars[pos..].iter().collect();
            let mut best = 0usize;
            for dict in [self.dicts.user.as_ref(), self.dicts.system.as_ref()]
                .into_iter()
                .flatten()
            {
                for res in dict.common_prefix_search(&rest) {
                    best = best.max(res.reading.chars().count());
                }
            }
            best
        };

        let mut boundaries = Vec::new();
        let mut pos = 0usize;
        while pos < chars.len() {
            let matched = longest_at(pos);
            if matched > 0 {
                pos += matched;
            } else {
                // Swallow the run of unmatched chars into one segment.
                pos += 1;
                while pos < chars.len() && longest_at(pos) == 0 {
                    pos += 1;
                }
            }
            boundaries.push(pos.min(chars.len()));
        }
        boundaries
    }

    /// Forget the learned candidate at `page_index` on the current page and
    /// drop it from the live list, so clicking its ✕ makes it stop coming back.
    ///
    /// Only learned entries can be deleted — dictionary and model candidates
    /// aren't user data. Returns `not_consumed` when the index isn't a learned
    /// candidate, leaving the list untouched.
    pub fn forget_candidate_on_page(&mut self, page_index: usize) -> EngineResult {
        let InputState::Conversion { candidates, .. } = &self.state else {
            return EngineResult::not_consumed();
        };
        let Some(cand) = candidates.page_candidates().get(page_index) else {
            return EngineResult::not_consumed();
        };
        let (text, reading) = (cand.text.clone(), cand.reading.clone());
        let reading = reading.unwrap_or_else(|| self.input_buf.text.clone());
        // The best remaining conversion: what the user is presumably reaching
        // for by deleting this one. Kana fallbacks (the ones carrying a width
        // annotation) are not a choice, so they teach nothing.
        let survivor = candidates
            .candidates()
            .iter()
            .find(|c| c.text != text && c.description.is_none())
            .map(|c| c.text.clone());
        let Some(cache) = self.learning.as_mut() else {
            return EngineResult::not_consumed();
        };
        // Drop any learned entry, and record a suppression so the candidate
        // stays gone even when it came from the model or a dictionary — the
        // point of the ✕ is "stop showing me this", whatever produced it.
        cache.forget(&reading, &text);
        cache.suppress(&reading, &text);
        // Learn from the deletion phrase by phrase, which is what makes the ✕
        // carry over to *other* sentences. Suppressing the whole surface only
        // helps if the identical sentence is typed again, and a long one is
        // never learned at all (`MAX_LEARN_SURFACE_CHARS`) — so deleting
        // `ご希望なら試飲保管を…` on its own would teach nothing. Aligning it
        // against the survivor isolates the phrase actually being rejected
        // (`しいんほかん`: `試飲保管` out, `子音補完` in), which is short enough
        // to learn and applies wherever that reading turns up next, live
        // conversion included.
        if let Some(survivor) = survivor
            && !self.secure_input
        {
            for (span_reading, rejected, wanted) in differing_spans(&reading, &text, &survivor) {
                cache.suppress(&span_reading, &rejected);
                cache.record(&span_reading, &wanted);
            }
        }

        let InputState::Conversion { candidates, .. } = &mut self.state else {
            return EngineResult::not_consumed();
        };
        let selected = candidates.selected_text().map(str::to_string);
        let kept: Vec<Candidate> = candidates
            .candidates()
            .iter()
            .filter(|c| c.text != text)
            .cloned()
            .collect();
        if kept.is_empty() {
            return EngineResult::not_consumed();
        }
        let mut list = CandidateList::new(kept);
        if let Some(sel) = selected
            && let Some(idx) = list.candidates().iter().position(|c| c.text == sel)
        {
            list.select(idx);
        }
        *candidates = list.clone();
        // Written through immediately: a deletion is a deliberate instruction
        // that must survive a crash or a force-quit, unlike ordinary learning
        // which can wait for the next focus change.
        self.save_learning();
        EngineResult::consumed().with_action(EngineAction::ShowCandidates(list))
    }

    /// Whole-sentence variants built by swapping one word of `converted` for a
    /// dictionary homophone, so the reranker actually has the right answer to
    /// choose from.
    ///
    /// Reranking can only reorder what exists, and the candidate sources can't
    /// produce these: the dictionary is consulted by exact match on the *whole*
    /// reading (`ざっしにのります` matches nothing), so the only sentence-level
    /// candidates come from the model's beam. When the beam misses the intended
    /// word — `雑誌に乗ります` where `載ります` was meant — nothing downstream
    /// can recover it. This fills that recall gap the way a lattice would:
    /// align the conversion to the reading, then for each phrase offer the
    /// dictionary's other readings of the same kana, spliced back into the
    /// sentence. Scoring in context then picks between them (`雑誌に` selects
    /// `載ります`, `車に` selects `乗ります`).
    ///
    /// Each phrase is split at the kana it shares with its reading, so only the
    /// stem is substituted and inflection survives: `のります`/`乗ります` share
    /// the suffix `ります`, leaving the stem `の`/`乗` to swap.
    /// Split a converted stem the dictionary doesn't hold as a whole
    /// (`しこうちゅう`/`施工中`) into the dictionary words it is made of
    /// (`しこう`/`施工` + `ちゅう`/`中`). `None` unless it splits into two or more.
    fn split_compound_stem(&self, reading: &str, converted: &str) -> Option<Vec<(String, String)>> {
        const MAX_PIECES: usize = 4;
        fn go(
            e: &InputMethodEngine,
            r: &[char],
            c: &str,
            out: &mut Vec<(String, String)>,
        ) -> bool {
            if r.is_empty() {
                return c.is_empty();
            }
            if out.len() >= MAX_PIECES {
                return false;
            }
            let longest = if out.is_empty() { r.len() - 1 } else { r.len() };
            for len in (1..=longest).rev() {
                let piece: String = r[..len].iter().collect();
                for cand in e.search_dictionaries(&piece, usize::MAX) {
                    let Some(rest) = c.strip_prefix(cand.text.as_str()) else {
                        continue;
                    };
                    if cand.text.is_empty() {
                        continue;
                    }
                    out.push((piece.clone(), cand.text));
                    if go(e, &r[len..], rest, out) {
                        return true;
                    }
                    out.pop();
                }
            }
            false
        }
        let r: Vec<char> = reading.chars().collect();
        let mut out = Vec::new();
        (r.len() >= 2 && go(self, &r, converted, &mut out)).then_some(out)
    }

    /// Phrase boundaries for segment navigation: the conversion's own phrases,
    /// with any compound the dictionary lacks split into its words, so
    /// `しこうちゅうの` (施工中の) can be focused as `しこう|ちゅうの`.
    fn phrase_boundaries(&self, reading: &str, converted: &str) -> Option<Vec<usize>> {
        let spans = kashiko_engine::segment::align(reading, converted)?;
        let r: Vec<char> = reading.chars().collect();
        let c: Vec<char> = converted.chars().collect();
        let mut bounds = Vec::new();
        for span in &spans {
            let seg_reading: String = r[span.reading_start..span.reading_end].iter().collect();
            let seg_converted: String = c[span.converted_start..span.converted_end].iter().collect();
            let shared = seg_reading
                .chars()
                .rev()
                .zip(seg_converted.chars().rev())
                .take_while(|(a, b)| a == b)
                .count();
            let stem_reading: String =
                seg_reading.chars().take(seg_reading.chars().count() - shared).collect();
            let stem_converted: String =
                seg_converted.chars().take(seg_converted.chars().count() - shared).collect();
            if !stem_reading.is_empty()
                && self.search_dictionaries(&stem_reading, 1).is_empty()
                && let Some(pieces) = self.split_compound_stem(&stem_reading, &stem_converted)
            {
                let mut at = span.reading_start;
                for (piece, _) in &pieces[..pieces.len() - 1] {
                    at += piece.chars().count();
                    bounds.push(at);
                }
            }
            bounds.push(span.reading_end);
        }
        Some(bounds)
    }

    pub(super) fn dictionary_substitution_variants(&self, reading: &str, converted: &str) -> Vec<String> {
        /// Cap on generated variants: each one costs a scoring pass later.
        const MAX_VARIANTS: usize = 6;

        let Some(spans) = kashiko_engine::segment::align(reading, converted) else {
            return Vec::new();
        };
        let r: Vec<char> = reading.chars().collect();
        let c: Vec<char> = converted.chars().collect();
        let take = |src: &[char], a: usize, b: usize| -> String {
            src[a.min(src.len())..b.min(src.len())].iter().collect()
        };

        let mut out = Vec::new();
        for span in &spans {
            if out.len() >= MAX_VARIANTS {
                break;
            }
            let seg_reading = take(&r, span.reading_start, span.reading_end);
            let seg_converted = take(&c, span.converted_start, span.converted_end);

            // Trim the shared tail (okurigana / auxiliary verb) so only the
            // stem is looked up and replaced.
            let shared: usize = seg_reading
                .chars()
                .rev()
                .zip(seg_converted.chars().rev())
                .take_while(|(a, b)| a == b)
                .count();
            let stem_reading: String = seg_reading
                .chars()
                .take(seg_reading.chars().count().saturating_sub(shared))
                .collect();
            let stem_converted: String = seg_converted
                .chars()
                .take(seg_converted.chars().count().saturating_sub(shared))
                .collect();
            // Nothing was converted here (pure kana), so there is no homophone
            // choice to offer.
            if stem_reading.is_empty() || stem_converted == stem_reading {
                continue;
            }

            // Nouns are in the dictionary under the stem itself; verbs are
            // only under their dictionary form, so an inflected verb needs
            // that form rebuilt before it can be looked up. `のります` leaves
            // the stem `の`, which the dictionary knows only as 野/乃/之 — the
            // homophone we want lives under `のる` (乗る, 載る).
            //
            // The 連用形 ending that begins the shared suffix says which row
            // the verb belongs to, and therefore how its dictionary form ends:
            // `り` is the い-段 of ら行, so the form is stem + `る`.
            let inflected: Option<(String, char)> = seg_reading
                .chars()
                .nth(seg_reading.chars().count().saturating_sub(shared))
                .and_then(dictionary_form_ending)
                .map(|end| (format!("{stem_reading}{end}"), end));

            // In a verb slot the plain stem resolves to unrelated nouns
            // (`の` → 野/乃/ゝ), so the dictionary form wins when it matches
            // anything; the plain stem is the fallback for nouns, where no
            // dictionary form exists (`ざっしす` finds nothing).
            let mut lookups: Vec<(String, Option<char>)> = Vec::new();
            if let Some((form, end)) = inflected
                && !self.search_dictionaries(&form, 1).is_empty()
            {
                lookups.push((form, Some(end)));
            }
            if lookups.is_empty() {
                // A compound the dictionary lacks as a whole (`しこうちゅう` →
                // 施工中) is swapped word by word: `しこう` → 思考/試行, `中` kept.
                if self.search_dictionaries(&stem_reading, 1).is_empty()
                    && let Some(pieces) = self.split_compound_stem(&stem_reading, &stem_converted)
                {
                    let head = take(&c, 0, span.converted_start);
                    let tail = take(&c, span.converted_end - shared.min(span.converted_end), c.len());
                    for (i, (piece_reading, piece_converted)) in pieces.iter().enumerate() {
                        for cand in self.search_dictionaries(piece_reading, usize::MAX) {
                            if out.len() >= MAX_VARIANTS {
                                break;
                            }
                            if &cand.text == piece_converted
                                || kashiko_engine::kana::is_all_kana(&cand.text)
                                || cand.text.chars().any(|ch| ch.is_ascii_alphabetic())
                            {
                                continue;
                            }
                            let stem: String = pieces
                                .iter()
                                .enumerate()
                                .map(|(j, (_, pc))| if j == i { cand.text.as_str() } else { pc.as_str() })
                                .collect();
                            let variant = format!("{head}{stem}{tail}");
                            if variant != converted && !out.contains(&variant) {
                                out.push(variant);
                            }
                        }
                    }
                    continue;
                }
                lookups.push((stem_reading.clone(), None));
            }

            for (lookup_reading, strip) in lookups {
                if out.len() >= MAX_VARIANTS {
                    break;
                }
                for cand in self.search_dictionaries(&lookup_reading, usize::MAX) {
                    if out.len() >= MAX_VARIANTS {
                        break;
                    }
                    // Drop the dictionary-form ending so the sentence's own
                    // inflection is what gets attached (`載る` → `載` + `ります`).
                    let surface = match strip {
                        Some(end) => match cand.text.strip_suffix(end) {
                            Some(stem) if !stem.is_empty() => stem.to_string(),
                            _ => continue,
                        },
                        None => cand.text.clone(),
                    };
                    if surface == stem_converted {
                        continue;
                    }
                    let cand = AnnotatedCandidate { text: surface, ..cand };
                    let variant = format!(
                        "{}{}{}{}",
                        take(&c, 0, span.converted_start),
                        cand.text,
                        take(
                            &c,
                            span.converted_end - shared.min(span.converted_end),
                            span.converted_end
                        ),
                        take(&c, span.converted_end, c.len()),
                    );
                    if variant != converted && !out.contains(&variant) {
                        out.push(variant);
                    }
                }
            }
        }
        out
    }

    /// Reorder the model/dictionary candidates by how natural the language
    /// model finds them in context (per-character NLL, lower is better).
    ///
    /// This is what puts `今日の天気は晴れです` above `今日の転機は晴れです`:
    /// both are valid readings of the kana, and only a language model can say
    /// which one is the sentence. Beam order alone can't — it ranks by the
    /// decoder's own path probability, which is why near-homophone junk kept
    /// surfacing high.
    ///
    /// Deliberately narrow:
    /// - Learning and user-dictionary candidates keep the top; they are the
    ///   user's own choices and outrank any model opinion.
    /// - Rewriter variants and the hiragana/katakana fallbacks keep the bottom.
    /// - Only the Model/Dictionary block in between is reordered.
    ///
    /// Scoring costs one forward pass per candidate (sharing the prompt's KV
    /// cache), so it is capped and skipped for short readings, where the
    /// candidate list is short and already good.
    pub(super) fn rerank_by_likelihood(&mut self, reading: &str, candidates: &mut Vec<AnnotatedCandidate>) {
        /// Readings shorter than this are left alone: a one- or two-character
        /// reading has few candidates and little context to judge them by, so
        /// the per-candidate forward pass buys nothing. Kept low because short
        /// inflected verbs (`のります`, 4 chars) are exactly where homophones
        /// need separating — a higher bound skipped that case entirely.
        const MIN_RERANK_READING_CHARS: usize = 3;
        /// Upper bound on forward passes per conversion. One pass is ~12ms, so
        /// this is what the user waits for after Space; the tail beyond it is
        /// dropped rather than shown unjudged (see below).
        const MAX_RERANK_CANDIDATES: usize = 12;

        // Cleared first: every early return below leaves no score for this
        // conversion, and a stale one from the previous conversion would set
        // the deep search's bar from unrelated text.
        self.last_quality_best = None;
        /// Readings up to this length are only reordered, never thinned.
        ///
        /// A single word gives the model no sentence to judge a candidate
        /// against, so a bad score there means "unusual", not "wrong". `ゆうと`
        /// offered `言うと` and nothing else for exactly that reason: the
        /// dictionary lists 96 readings and the model's beam adds 優斗, 悠と,
        /// 佑都 — and every one of them scored badly enough to be deleted,
        /// leaving a verb form to win a reading that is nearly always a name.
        /// Homophone variety is the point of a short reading, and the scoring
        /// budget already bounds how many are shown.
        ///
        /// Long readings keep the strict gate: there the score is measured
        /// against a real sentence, and a homophone that reads badly in one is
        /// genuinely wrong.
        const SHORT_READING_CHARS: usize = 4;

        /// A single phrase up to this length (`しこうちゅうの`, one word plus its
        /// particle) is judged like a short reading: with no sentence around
        /// it the model just prefers the commonest homophone (施工中の 0.05 vs
        /// 思考中の 1.10), which says nothing about which one the user means.
        /// Only candidates with a genuinely broken character are dropped, by a
        /// looser bar than a sentence gets: jinen-v2 puts 思考中の's worst
        /// character at 6.5, against 10.0 for 志向中の.
        const SINGLE_PHRASE_MAX_CHARS: usize = 8;
        const SINGLE_PHRASE_WORST_TOKEN_NLL: f32 = 8.0;

        let short_reading = reading.chars().count() <= SHORT_READING_CHARS;
        let single_phrase = reading.chars().count() <= SINGLE_PHRASE_MAX_CHARS
            && candidates
                .first()
                .and_then(|top| kashiko_engine::segment::align(reading, &top.text))
                .is_some_and(|spans| spans.len() == 1);
        if reading.chars().count() < MIN_RERANK_READING_CHARS || self.converters.kanji.is_none() {
            return;
        }
        // Learned candidates take part in ranking so context can overrule a
        // past choice made in a different sentence, but they are never dropped
        // (see `Scored::never_drop`) and get a head start
        // (`LEARNED_SCORE_BONUS`). User-dictionary entries stay out of this
        // entirely: those are explicit configuration, not an inference.
        let rerankable = |c: &AnnotatedCandidate| {
            matches!(
                c.source,
                CandidateSource::Model | CandidateSource::Dictionary | CandidateSource::Learning
            )
        };
        let mut positions: Vec<usize> = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| rerankable(c))
            .map(|(i, _)| i)
            .collect();
        if positions.len() < 2 {
            // One candidate cannot be reordered, but its score is still the
            // answer to "did this convert well?" — which the typo-correction
            // heuristics need. Narrowing the synchronous beam made this the
            // common case, and without the score those heuristics fall back to
            // their false-positive behaviour: `皆さんおはようございます` has a
            // nine-character kana run shared with the reading, so it reads as a
            // failed conversion until something says otherwise.
            if let Some(&i) = positions.first() {
                let text = candidates[i].text.clone();
                self.last_quality_best = self.score_one(reading, &text);
            }
            return;
        }
        // Anything past the scoring budget is removed, not shown unjudged.
        // Passing it through was the larger half of "why are there so many
        // candidates": for `かけているぶんしょう` the budget was spent on the
        // first twelve, of which the gate kept four — and then
        // `かけている分廠`, `分省`, `分章` (2.5, 2.1, 2.7 NLL against 0.10 for
        // the best) walked straight into the window because nothing had looked
        // at them. Candidate sources emit roughly best-first, so the tail is
        // where the rare readings live; the deep search adds more later if the
        // remaining list turns out thin, and those *are* gated.
        let overflow: HashSet<usize> = positions
            .split_off(positions.len().min(MAX_RERANK_CANDIDATES))
            .into_iter()
            // The user's own past choices are never silently discarded.
            .filter(|&i| candidates[i].source != CandidateSource::Learning)
            .collect();
        if !overflow.is_empty() {
            let mut kept = 0usize;
            candidates.retain(|_| {
                let keep = !overflow.contains(&kept);
                kept += 1;
                keep
            });
            positions = candidates
                .iter()
                .enumerate()
                .filter(|(_, c)| rerankable(c))
                .map(|(i, _)| i)
                .take(MAX_RERANK_CANDIDATES)
                .collect();
        }

        let texts: Vec<String> = positions
            .iter()
            .map(|&i| candidates[i].text.clone())
            .collect();
        let ctx = self.truncate_context_for_api();
        let Some(converter) = self.converters.kanji.as_ref() else {
            return;
        };
        let Ok(scores) = converter.score_candidates_detailed(reading, &ctx, &texts) else {
            return;
        };

        let scored: Vec<Scored> = positions
            .iter()
            .copied()
            .zip(scores)
            .map(|(index, detail)| {
                let learned = candidates[index].source == CandidateSource::Learning;
                Scored {
                    index,
                    score: detail
                        .map(|d| if learned { d.mean - LEARNED_SCORE_BONUS } else { d.mean }),
                    worst: detail.map(|d| d.worst),
                    never_drop: learned
                        || short_reading
                        || (single_phrase
                            && detail.is_some_and(|d| d.worst <= SINGLE_PHRASE_WORST_TOKEN_NLL)),
                }
            })
            .collect();
        // Candidates still carrying raw keystrokes are excluded from the
        // recorded best: they score well precisely *because* they copy the
        // input, and letting that stand in for "the conversion went fine" is
        // how a genuinely garbled sentence looked confident —
        // `サイビルド前だったt前だと思われます` scored 0.18 while every real
        // correction of it sat above 1.9. Kana passthroughs need no such rule;
        // the model rates those badly on their own.
        self.last_quality_best = scored
            .iter()
            .filter(|e| !candidates[e.index].text.chars().any(|c| c.is_ascii_alphabetic()))
            .filter_map(|e| e.score)
            .reduce(f32::min);
        let plain: HashMap<String, f32> = scored
            .iter()
            .filter_map(|e| Some((candidates[e.index].text.clone(), e.score?)))
            .collect();
        *candidates = gate_and_reorder(candidates, &scored);
        if let Some(right) = self.scope_right_context(reading) {
            self.reorder_by_right_context(reading, &right, candidates, &plain);
        }
    }

    /// What follows `reading` when it is converted: the unconverted rest of
    /// the composition, then the document text after the cursor.
    fn scope_right_context(&self, reading: &str) -> Option<String> {
        let rest = self.input_buf.text.strip_prefix(reading).unwrap_or("");
        let right = format!("{rest}{}", self.right_context().unwrap_or_default());
        (!right.trim().is_empty()).then_some(right)
    }

    /// Reorder the leading plausible model/dictionary candidates by how they
    /// read before `right` (see `KanaKanjiConverter::score_with_right_context`).
    /// Nothing is added or removed; `plain` holds each one's score without
    /// the right context, which bounds which ones may move up.
    fn reorder_by_right_context(
        &self,
        reading: &str,
        right: &str,
        candidates: &mut [AnnotatedCandidate],
        plain: &HashMap<String, f32>,
    ) {
        /// Scoring budget: two passes per candidate.
        const MAX_REORDERED: usize = 6;
        const PLAUSIBLE_MARGIN: f32 = 1.0;
        let Some(converter) = self.converters.kanji.as_ref() else {
            return;
        };
        let floor = plain.values().copied().fold(f32::INFINITY, f32::min);
        let slots: Vec<usize> = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| matches!(c.source, CandidateSource::Model | CandidateSource::Dictionary))
            .filter(|(_, c)| plain.get(&c.text).is_some_and(|&p| p <= floor + PLAUSIBLE_MARGIN))
            .map(|(i, _)| i)
            .take(MAX_REORDERED)
            .collect();
        if slots.len() < 2 {
            return;
        }
        let texts: Vec<String> = slots.iter().map(|&i| candidates[i].text.clone()).collect();
        let ctx = self.truncate_context_for_api();
        let Ok(scores) = converter.score_with_right_context(reading, &ctx, &texts, right) else {
            return;
        };
        let mut order: Vec<(usize, f32)> = slots
            .iter()
            .zip(scores)
            .map(|(&i, s)| (i, s.map_or(f32::INFINITY, |(_, total)| total)))
            .collect();
        order.sort_by(|a, b| a.1.total_cmp(&b.1));
        let moved: Vec<AnnotatedCandidate> =
            order.iter().map(|&(i, _)| candidates[i].clone()).collect();
        for (&slot, cand) in slots.iter().zip(moved) {
            candidates[slot] = cand;
        }
    }

    /// Take the settled segments' text (left of the focus), learning each
    /// against its own reading on the way out, and clear them.
    fn take_decided_text(&mut self) -> String {
        let decided = std::mem::take(&mut self.decided);
        if decided.is_empty() {
            return String::new();
        }
        let mut at = 0usize;
        let mut out = String::new();
        for (seg_len, seg_text) in &decided {
            if self.input_mode != InputMode::Emoji {
                let seg_reading: String =
                    self.input_buf.text.chars().skip(at).take(*seg_len).collect();
                self.record_learning(&seg_reading, seg_text);
            }
            out.push_str(seg_text);
            at += seg_len;
        }
        out
    }

    /// Slide the focused conversion segment one word left or right.
    ///
    /// Left/Right move the highlighted window *through* the reading — the
    /// natural reading of "move the selection" — rather than growing it from
    /// the start. Moving right settles the current segment's choice into
    /// `self.decided` and focuses the next word; moving left pops the previous
    /// choice back into focus. Shift+Left/Right still resize the segment for
    /// when the word split itself is wrong.
    fn move_segment_focus(&mut self, delta: isize) -> EngineResult {
        let InputState::Conversion {
            split_index,
            segment_start,
            ref candidates,
            ..
        } = self.state
        else {
            return EngineResult::consumed();
        };
        let selected = candidates.selected_text().unwrap_or_default().to_string();
        let full_reading = self.input_buf.text.clone();
        let total = full_reading.chars().count();

        // Phrase boundaries: prefer aligning the whole-sentence conversion
        // against the reading, which is the only source that knows where the
        // words actually are. Dictionary longest-match read
        // `きょうのてんきははれです` as `…てんき|はは|れです` (母!), because it
        // can't see that the two `は` belong to different words — the
        // conversion `今日の天気は晴れです` makes that obvious. Computed once,
        // on the first arrow press, and reused for the rest of the navigation.
        if self.segment_bounds.is_empty() {
            self.segment_bounds =
                self.phrase_boundaries(&full_reading, &selected)
                    // A conversion left in kana is one opaque phrase; the
                    // dictionary can still find the words in it.
                    .filter(|b| b.len() > 1)
                    .unwrap_or_else(|| self.word_boundaries(&full_reading));
        }
        let boundaries = self.segment_bounds.clone();

        // Right after Space the whole reading is one focused span (so the user
        // can pick a whole-sentence candidate). The first arrow press splits it
        // into words; from then on the focus slides, wrapping back to the whole
        // span at either end so both keys always do something.
        let whole_focused = segment_start == 0 && split_index >= total;

        if delta > 0 {
            if whole_focused {
                let first_end = boundaries.first().copied().unwrap_or(total);
                if first_end >= total {
                    // Single word: nothing to move through.
                    return EngineResult::consumed();
                }
                return self.convert_focused_segment(&full_reading, 0, first_end);
            }
            if split_index >= total {
                // Past the last word: back to the whole span.
                self.decided.clear();
                return self.convert_focused_segment(&full_reading, 0, total);
            }
            // Settle this segment and focus the next word.
            self.decided
                .push((split_index - segment_start, selected));
            let new_start = split_index;
            let new_end = boundaries
                .iter()
                .find(|&&b| b > new_start)
                .copied()
                .unwrap_or(total);
            self.convert_focused_segment(&full_reading, new_start, new_end)
        } else {
            if whole_focused {
                // Already covering everything — the leftmost position.
                return EngineResult::consumed();
            }
            let Some((prev_len, _)) = self.decided.pop() else {
                // At the first word: widen back out to the whole span.
                return self.convert_focused_segment(&full_reading, 0, total);
            };
            let new_end = segment_start;
            let new_start = new_end.saturating_sub(prev_len);
            self.convert_focused_segment(&full_reading, new_start, new_end)
        }
    }

    /// Convert `full_reading[start..end]` and focus it, leaving already-decided
    /// segments and the raw tail around it untouched.
    fn convert_focused_segment(
        &mut self,
        full_reading: &str,
        start: usize,
        end: usize,
    ) -> EngineResult {
        let reading: String = full_reading
            .chars()
            .skip(start)
            .take(end.saturating_sub(start))
            .collect();
        if reading.is_empty() {
            return EngineResult::consumed();
        }
        // The latin-word candidate covers the whole composition, so it is only
        // valid when the focus spans everything (the wrap-around case).
        let allow_latin = start == 0 && end >= full_reading.chars().count();
        let candidates = self.build_conversion_candidates(
            &reading,
            self.config.num_candidates,
            false,
            None,
            allow_latin,
        );
        let list = CandidateList::new(
            candidates
                .into_iter()
                .map(|ac| {
                    let cand_reading = ac.reading.unwrap_or_else(|| reading.clone());
                    let label = ac.source.label();
                    Candidate {
                        text: ac.text,
                        reading: Some(cand_reading),
                        source_label: (!label.is_empty()).then(|| label.to_string()),
                        description: ac.description,
                    }
                })
                .collect(),
        );
        if list.is_empty() {
            return EngineResult::consumed();
        }
        // Keep the tail converted rather than reverting it to hiragana: moving
        // the focus is navigation within an already-converted sentence, so only
        // the focused segment should look unsettled.
        let tail_reading: String = full_reading.chars().skip(end).collect();
        self.tail_text = if tail_reading.is_empty() {
            None
        } else {
            let ctx = self.truncate_context_for_api();
            self.run_kana_kanji_conversion(&tail_reading, &ctx, 1)
                .into_iter()
                .next()
                .filter(|t| t != &tail_reading)
        };
        // Focus moves are user-driven navigation, not a fresh conversion, so
        // no deep search is scheduled for them.
        self.deep_search = None;
        self.enter_conversion_state_at(full_reading, list, start, end)
    }

    /// Resize the focused conversion segment (Shift+Left/Right), jumping
    /// between dictionary word boundaries (`すべての|あぷりで|…`) instead of
    /// one character at a time. Wraps around at either end.
    fn resize_conversion(&mut self, delta: isize) -> EngineResult {
        if let InputState::Conversion { split_index, .. } = self.state {
            let full_reading = self.input_buf.text.clone();
            let full_len = full_reading.chars().count();
            let boundaries = self.word_boundaries(&full_reading);
            let new_split = if delta < 0 {
                // Nearest boundary left of the current split; wrap to the
                // full reading when already at the first word.
                boundaries
                    .iter()
                    .rev()
                    .find(|&&b| b < split_index)
                    .copied()
                    .unwrap_or(full_len)
            } else {
                // Nearest boundary right of the current split; wrap to the
                // first word when already covering the whole reading.
                boundaries
                    .iter()
                    .find(|&&b| b > split_index)
                    .copied()
                    .unwrap_or_else(|| boundaries.first().copied().unwrap_or(full_len))
            };

            if new_split != split_index && new_split > 0 {
                // Restart conversion with the new split index
                return self.start_conversion(false, Some(new_split), 0);
            }
        }
        EngineResult::consumed()
    }

    /// Select and commit the candidate at `page_index` (0-based) within the
    /// current page, like pressing the digit key `page_index + 1`. Not
    /// consumed unless a candidate list is active (Conversion state).
    pub fn select_candidate_on_page(&mut self, page_index: usize) -> EngineResult {
        let start = std::time::Instant::now();
        self.metrics.conversion_ms = 0;
        let result = self.select_candidate_by_digit(page_index + 1);
        self.metrics.process_key_ms = start.elapsed().as_millis() as u64;
        result
    }

    /// Select candidate by digit (1-9)
    fn select_candidate_by_digit(&mut self, digit: usize) -> EngineResult {
        let (selected_text, reading) = {
            let candidates = match self.state.candidates_mut() {
                Some(c) => c,
                None => return EngineResult::not_consumed(),
            };

            if candidates.select_on_page(digit).is_none() {
                return EngineResult::consumed();
            }

            let text = candidates.selected_text().unwrap_or("").to_string();
            let reading = candidates.selected().and_then(|c| c.reading.clone());
            (text, reading)
        };

        // Record learning before committing
        if let Some(reading) = &reading {
            self.record_learning(reading, &selected_text);
        }

        // Settled segments (from sliding the focus) commit ahead of this text.
        let selected_text = format!("{}{}", self.take_decided_text(), selected_text);

        // Remember the commit for one key (Ctrl+Backspace undo).
        if let Some(reading) = &reading {
            self.last_commit = Some((reading.clone(), selected_text.clone()));
        }

        // Commit immediately after digit selection

        self.state = InputState::Empty;

        EngineResult::consumed()
            .with_action(EngineAction::UpdatePreedit(Preedit::new()))
            .with_action(EngineAction::HideCandidates)
            .with_action(EngineAction::HideAuxText)
            .with_action(EngineAction::Commit(selected_text))
    }

    /// Update preedit after candidate selection change
    fn update_conversion_preedit(
        &mut self,
        selected_text: &str,
        candidates: &CandidateList,
    ) -> EngineResult {
        let mut preedit = Preedit::with_text(selected_text);
        preedit.set_attributes(vec![PreeditAttribute::new(
            0,
            selected_text.chars().count(),
            AttributeType::Highlight,
        )]);

        if let Some(p) = self.state.preedit_mut() {
            *p = preedit.clone();
        }

        let reading = candidates
            .selected()
            .and_then(|c| c.reading.as_deref())
            .unwrap_or("");

        EngineResult::consumed()
            .with_action(EngineAction::UpdatePreedit(preedit))
            .with_action(EngineAction::ShowCandidates(candidates.clone()))
            .with_action(EngineAction::UpdateAuxText(
                self.format_aux_conversion_with_page(reading, Some(candidates)),
            ))
    }

    /// Handle backspace in conversion mode
    fn backspace_conversion(&mut self) -> EngineResult {
        // Return to hiragana mode with the reading
        self.cancel_conversion()
    }
}
