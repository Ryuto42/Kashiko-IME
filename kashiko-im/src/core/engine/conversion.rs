//! Conversion state handling (candidates, commit). The live-conversion
//! chunking lives in the sibling `chunk` module.

use std::collections::{HashSet, VecDeque};
use std::time::Instant;

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

/// An in-flight deep search, advanced one task per `step_deep_search` call.
pub(super) struct DeepSearchJob {
    /// The request the job was started for; the job cancels itself when the
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
}

/// How much worse than the best candidate a candidate may score and still be
/// shown, in per-character NLL. Relative rather than absolute so it adapts to
/// how hard the sentence is: what matters is whether a candidate is close to
/// the best reading, not its raw likelihood.
const QUALITY_MARGIN_NLL: f32 = 0.55;

/// Absolute per-character NLL ceiling. Catches the case where even the best
/// candidate is poor, so "within the margin of the best" would still let
/// nonsense through.
const QUALITY_CEILING_NLL: f32 = 1.6;

/// Candidates always kept, however badly they score. The list must never be
/// gutted — the user needs something to pick even when the model is lost.
pub(super) const MIN_KEPT_SCORED: usize = 2;

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
    scored: &[(usize, Option<f32>)],
) -> Vec<AnnotatedCandidate> {
    let mut order = scored.to_vec();
    // Stable sort: equal or unscoreable candidates keep their original order.
    order.sort_by(|a, b| match (a.1, b.1) {
        (Some(x), Some(y)) => x.total_cmp(&y),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    });

    let best = order.iter().filter_map(|(_, s)| *s).fold(f32::INFINITY, f32::min);
    let survivors: Vec<usize> = order
        .iter()
        .enumerate()
        .filter(|(rank, (_, score))| match score {
            // Unscoreable: keep.
            None => true,
            Some(s) => {
                *rank < MIN_KEPT_SCORED
                    || (*s <= best + QUALITY_MARGIN_NLL && *s <= QUALITY_CEILING_NLL)
            }
        })
        .map(|(_, (idx, _))| *idx)
        .collect();

    // Rebuild the list: unscored candidates stay put (keeping learning on top
    // and the fallbacks at the bottom), and the scored block is replaced by its
    // survivors in ranked order, compacting away the dropped ones.
    let scored_slots: HashSet<usize> = scored.iter().map(|(i, _)| *i).collect();
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
struct CandidateBuilder {
    candidates: Vec<AnnotatedCandidate>,
    seen: HashSet<String>,
}

impl CandidateBuilder {
    fn new() -> Self {
        Self {
            candidates: Vec::new(),
            seen: HashSet::new(),
        }
    }

    /// Push a candidate if its text hasn't been seen yet.
    fn push(&mut self, ac: AnnotatedCandidate) {
        if self.seen.insert(ac.text.clone()) {
            self.candidates.push(ac);
        }
    }

    /// Push a candidate unconditionally, marking its text as seen so later
    /// dedup'd inserts skip it. Use only for sources that should win over
    /// duplicates from later steps (e.g. learning cache).
    fn push_force(&mut self, ac: AnnotatedCandidate) {
        self.seen.insert(ac.text.clone());
        self.candidates.push(ac);
    }

    fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    fn into_candidates(self) -> Vec<AnnotatedCandidate> {
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
            "convert: reading=\"{}\" api_context=\"{}\" candidates={} strategy={:?}",
            reading, api_context, num_candidates, strategy
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
        let auto_split = if split_index.is_none() {
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
                self.config.num_candidates,
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
            if let Some(garbled_top) = garbled_top {
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

        // Rank the model/dictionary candidates by how natural they read in
        // context before anything is pinned on top, so the ordering reflects
        // the language model rather than the decoder's beam order.
        self.rerank_by_likelihood(&reading, &mut candidates);

        // If the previous auto-suggest result is not in the new candidates, insert it at the top
        // so it doesn't disappear when the conversion strategy changes.
        let seen: HashSet<&str> = candidates.iter().map(|c| c.text.as_str()).collect();
        if !prev_suggest_text.is_empty()
            && prev_suggest_text != reading
            && !seen.contains(prev_suggest_text.as_str())
        {
            candidates.insert(
                0,
                AnnotatedCandidate::new(prev_suggest_text.clone(), CandidateSource::Model),
            );
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
            let top_is_live_duplicate = candidate_list
                .candidates()
                .first()
                .is_some_and(|c| !prev_suggest_text.is_empty() && c.text == prev_suggest_text);
            if top_is_live_duplicate {
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
        let scope_has_ascii =
            !deliberate_latin && scoped_raw.chars().any(|c| c.is_ascii_alphabetic());
        // Alternatives are only worth a background pass on input long enough
        // to have competing readings; short input would otherwise schedule a
        // model run on nearly every Space press, since a thin candidate list
        // is normal there.
        const MIN_ALTERNATIVES_READING_CHARS: usize = 6;
        let kind = if typo_suspected || scope_has_ascii || top_garbled {
            Some(DeepSearchKind::TypoCorrection)
        } else if real_candidate_count < MIN_QUICK_REAL_CANDIDATES
            && reading.chars().count() >= MIN_ALTERNATIVES_READING_CHARS
        {
            Some(DeepSearchKind::MoreAlternatives)
        } else {
            None
        };
        self.deep_search = kind.map(|kind| DeepSearchRequest {
            reading: scoped_raw,
            buffer: full_reading.clone(),
            scope: split_idx,
            kind,
        });

        let mut result = self.enter_conversion_state(&full_reading, candidate_list, split_idx);
        if self.deep_search.is_some() {
            result = result.with_action(EngineAction::UpdateAuxText(
                "候補を検索中…".to_string(),
            ));
        }
        result
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
    fn search_dictionaries(&self, reading: &str, limit: usize) -> Vec<AnnotatedCandidate> {
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
                if seen.insert(cand.surface.clone()) {
                    candidates.push(AnnotatedCandidate::new(
                        cand.surface.clone(),
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
                if seen.insert(cand.surface.clone()) {
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
        self.deep_search.is_some() || self.deep_job.is_some()
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
    pub fn step_deep_search(&mut self) -> Option<EngineResult> {
        // Lazily build the job from the scheduled request.
        if self.deep_job.is_none()
            && let Some(request) = self.deep_search.take()
        {
            self.deep_job = self.build_deep_job(request);
        }
        let job = self.deep_job.as_ref()?;

        // Cancel when the conversion has moved on (committed / re-edited /
        // resized to a different segment).
        let still_valid = matches!(
            &self.state,
            InputState::Conversion { split_index, .. }
                if *split_index == job.request.scope
        ) && self.input_buf.text == job.request.buffer;
        if !still_valid {
            self.deep_job = None;
            return None;
        }

        let mut job = self.deep_job.take()?;
        let done = match job.queue.pop_front() {
            None => true,
            Some(task) => {
                self.run_deep_task(task, &mut job);
                job.queue.is_empty() || job.found.len() >= job.max_results
            }
        };
        if done {
            Some(self.finish_deep_job(job))
        } else {
            self.deep_job = Some(job);
            None
        }
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

        const MAX_TOTAL_RESULTS: usize = 6;
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
                    push_convert(&mut queue, v, 3, 2);
                }
                for (i, kb) in
                    kashiko_engine::typo::keyboard_typo_variants(&region, MAX_KEYBOARD_VARIANTS)
                        .into_iter()
                        .enumerate()
                {
                    let beam = if i < 4 { 3 } else { 1 };
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
                push_convert(&mut queue, cleaned, 5, 2);
            }
            DeepSearchKind::MoreAlternatives => {
                // Wide main-model beam over the whole reading, then tail
                // alternatives (`かき` → 書き/下記/柿) with the prefix held
                // fixed.
                let tail_split = self.split_tail_for_alternatives(&cleaned);
                push_convert(&mut queue, cleaned.clone(), 8, 4);
                if let Some((prefix_reading, tail_reading)) = tail_split {
                    queue.push_back(DeepTask::TailBeam {
                        prefix_reading,
                        tail_reading,
                        beam: 6,
                    });
                }
            }
        }

        let reading = original
            .chars()
            .filter(|c| !c.is_ascii_alphabetic())
            .collect();
        (!queue.is_empty()).then(|| DeepSearchJob {
            request,
            reading,
            queue,
            found: Vec::new(),
            seen: existing.into_iter().collect(),
            max_results: MAX_TOTAL_RESULTS,
        })
    }

    /// Execute one deep-search task, appending clean findings to the job.
    fn run_deep_task(&mut self, task: DeepTask, job: &mut DeepSearchJob) {
        let ctx = self.truncate_context_for_api();
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
                for out in self.deep_convert(&variant, &ctx, beam) {
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
                let prefix_conv = self
                    .cached_prefix_conversion(prefix_reading.chars().count())
                    .and_then(|v| v.into_iter().next());
                let prefix_conv = match prefix_conv {
                    Some(p) => Some(p),
                    None => self
                        .deep_convert(&prefix_reading, &ctx, 1)
                        .into_iter()
                        .next(),
                };
                let Some(prefix_conv) = prefix_conv else {
                    return;
                };
                let reading = job.reading.clone();
                let tail_ctx = self.lctx_for(&ctx, &prefix_conv);
                for tail_out in self.deep_convert(&tail_reading, &tail_ctx, beam) {
                    if job.found.len() >= job.max_results {
                        break;
                    }
                    accept(job, format!("{prefix_conv}{tail_out}"), &reading);
                }
            }
        }
    }

    /// Merge the finished job's findings into the live candidate list and
    /// build the notification result (always at least `HideAuxText`, to
    /// clear the loading indicator).
    fn finish_deep_job(&mut self, job: DeepSearchJob) -> EngineResult {
        let found = job.found;
        let InputState::Conversion { candidates, .. } = &mut self.state else {
            return EngineResult::consumed().with_action(EngineAction::HideAuxText);
        };
        if found.is_empty() {
            // Nothing new — just clear the loading indicator.
            return EngineResult::consumed().with_action(EngineAction::HideAuxText);
        }
        // Insert findings right after the currently selected candidate so
        // they are visible on the *current page* — inserting at the bottom
        // (above the fallbacks) buried them on page 2 whenever the quick
        // pass had already filled page 1, which made the deep results look
        // like they never arrived. Capped at the hiragana/katakana fallback
        // position so those always stay at the very bottom, and the user's
        // currently selected candidate stays selected — the list grows
        // below the selection, it never jumps.
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
            .with_action(EngineAction::HideAuxText)
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

    /// Direct main-model conversion for the deep search, bypassing the
    /// adaptive strategy (which would downshift long input to the light
    /// model — the deep pass explicitly wants the strong model).
    fn deep_convert(&mut self, reading: &str, ctx: &str, num_candidates: usize) -> Vec<String> {
        if !kashiko_engine::contains_kana(reading) {
            return vec![];
        }
        let Some(converter) = self.converters.kanji.as_ref() else {
            return vec![];
        };
        let katakana = kashiko_engine::hiragana_to_katakana(reading);
        converter.convert(&katakana, ctx, num_candidates).unwrap_or_default()
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
            let reading_lower = reading.to_ascii_lowercase();
            let invents_ascii = |text: &str| {
                text.chars().any(|c| {
                    c.is_ascii_alphanumeric()
                        && !reading_lower.contains(c.to_ascii_lowercase())
                })
            };
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

        // Record learning for the committed segment
        if self.input_mode != InputMode::Emoji {
            if let Some(reading) = &reading {
                self.record_learning(reading, &text);
            }
        }

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
    fn word_boundaries(&self, reading: &str) -> Vec<usize> {
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
        let Some(cache) = self.learning.as_mut() else {
            return EngineResult::not_consumed();
        };
        // Drop any learned entry, and record a suppression so the candidate
        // stays gone even when it came from the model or a dictionary — the
        // point of the ✕ is "stop showing me this", whatever produced it.
        cache.forget(&reading, &text);
        cache.suppress(&reading, &text);

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
        EngineResult::consumed().with_action(EngineAction::ShowCandidates(list))
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
        /// Readings shorter than this are left alone: few candidates, and the
        /// per-candidate forward pass isn't worth the Space-key latency.
        const MIN_RERANK_READING_CHARS: usize = 5;
        /// Upper bound on forward passes per conversion. Set above the usual
        /// candidate count so the whole list gets judged — a candidate left
        /// unscored can't be gated, which is how junk used to survive at the
        /// bottom of the list.
        const MAX_RERANK_CANDIDATES: usize = 12;

        if reading.chars().count() < MIN_RERANK_READING_CHARS || self.converters.kanji.is_none() {
            return;
        }
        let rerankable = |c: &AnnotatedCandidate| {
            matches!(
                c.source,
                CandidateSource::Model | CandidateSource::Dictionary
            )
        };
        let positions: Vec<usize> = candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| rerankable(c))
            .map(|(i, _)| i)
            .take(MAX_RERANK_CANDIDATES)
            .collect();
        if positions.len() < 2 {
            return;
        }

        let texts: Vec<String> = positions
            .iter()
            .map(|&i| candidates[i].text.clone())
            .collect();
        let ctx = self.truncate_context_for_api();
        let Some(converter) = self.converters.kanji.as_ref() else {
            return;
        };
        let Ok(scores) = converter.score_candidates(reading, &ctx, &texts) else {
            return;
        };

        let scored: Vec<(usize, Option<f32>)> =
            positions.iter().copied().zip(scores).collect();
        *candidates = gate_and_reorder(candidates, &scored);
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
                kashiko_engine::segment::boundaries_from_conversion(&full_reading, &selected)
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
