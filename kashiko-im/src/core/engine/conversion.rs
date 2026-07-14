//! Conversion state handling (candidates, commit). The live-conversion
//! chunking lives in the sibling `chunk` module.

use std::collections::HashSet;
use std::time::Instant;

use tracing::debug;

use super::*;

/// Maximum number of learning candidates to show
const MAX_LEARNING_CANDIDATES: usize = 3;

/// Minimum length of a passed-through hiragana run (present verbatim in the
/// reading) for a model candidate to count as a low-quality "typo passthrough"
/// and for a full-sentence conversion to be deemed a failure worth splitting.
/// See `kashiko_engine::has_long_unconverted_run`.
///
/// Set to 7 so ordinary Japanese hiragana chains (adverb + particles, e.g.
/// `はとてもいい` = 6) don't trip it, while a genuinely garbled stretch left
/// unconverted (e.g. `できないようらな` = 8) does.
const UNCONVERTED_RUN_MIN: usize = 7;

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
                light_converter
                    .convert(&katakana, api_context, n)
                    .unwrap_or_default()
            }
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

        // Mid-text romaji typo (auto_split fired): before falling back to the
        // split-at-typo behavior, ask the model what the user *meant* — strip
        // the stray ASCII letters out of the reading and convert the cleaned
        // whole. If the model produces a real conversion, present that as the
        // candidate list for the full buffer (committing it drops the stray
        // chars); only when the model can't make sense of it do we keep the
        // old split behavior (prefix converted, tail left raw).
        let mut candidates: Option<Vec<AnnotatedCandidate>> = None;
        if auto_split.is_some() {
            let cleaned: String = full_reading
                .chars()
                .filter(|c| !c.is_ascii_alphabetic())
                .collect();
            if kashiko_engine::contains_kana(&cleaned) {
                let cleaned_candidates = self.build_conversion_candidates(
                    &cleaned,
                    self.config.num_candidates,
                    skip_learning,
                    None,
                );
                let has_real_conversion = cleaned_candidates
                    .iter()
                    .any(|c| c.source == CandidateSource::Model && c.text != cleaned);
                if has_real_conversion {
                    candidates = Some(cleaned_candidates);
                    reading = cleaned;
                    split_idx = total_chars;
                    // The live text shown during typing contained the stray
                    // romaji — don't resurface it over the corrected list.
                    prev_suggest_text.clear();
                }
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
        let mut typo_suspected = auto_split.is_some();
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

        // Typo suspected on a whole-buffer conversion: schedule the deep
        // correction search. It runs *after* this synchronous response is
        // written, and its findings are pushed to the frontend as an
        // `update_candidates` notification, so the quick candidates appear
        // immediately and better guesses stream in behind the loading aux.
        // Split conversions are excluded — appending whole-reading candidates
        // to a prefix-scoped list would corrupt the commit scope.
        let deep_pending = typo_suspected && split_idx == total_chars;
        self.deep_search_reading = deep_pending.then(|| full_reading.clone());

        let mut result = self.enter_conversion_state(&full_reading, candidate_list, split_idx);
        if deep_pending {
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
        let reading: String = full_reading.chars().take(split_index).collect();
        let selected_text = candidates.selected_text().unwrap_or(&reading).to_string();

        let mut segments = vec![PreeditSegment::highlighted(&selected_text)];
        if split_index < full_reading.chars().count() {
            let remaining_reading: String = full_reading.chars().skip(split_index).collect();
            segments.push(PreeditSegment::new(&remaining_reading, crate::core::preedit::AttributeType::Underline));
        }

        let preedit = Preedit::from_segments(
            segments,
            selected_text.chars().count(),
        );

        self.state = InputState::Conversion {
            preedit: preedit.clone(),
            candidates: candidates.clone(),
            split_index,
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

    /// Deep typo-correction search, run by the server *after* the synchronous
    /// key response has been written. Explores a wider edit space than the
    /// quick in-response pass (base reading with a wide main-model beam, plus
    /// single deletions / adjacent swaps / adjacent-pair deletions around the
    /// typo region) and merges any clean findings into the live candidate
    /// list. The server pushes the returned actions to the frontend as an
    /// `update_candidates` notification, so candidates "stream in" while the
    /// user is looking at the quick results.
    ///
    /// Returns `None` when no search was scheduled or the conversion state
    /// has moved on; returns a `HideAuxText`-only result when the search ran
    /// but found nothing (to clear the loading indicator).
    pub fn run_deep_typo_search(&mut self) -> Option<EngineResult> {
        const MAX_KEYBOARD_VARIANTS: usize = 14;
        /// Chars of context on each side of the typo point included in the
        /// keyboard-edit region.
        const REGION_MARGIN: usize = 4;

        let original = self.deep_search_reading.take()?;
        // Bail out unless we're still on the same whole-buffer conversion.
        let InputState::Conversion { split_index, .. } = &self.state else {
            return None;
        };
        if *split_index != self.input_buf.text.chars().count() || self.input_buf.text != original
        {
            return None;
        }
        let existing: Vec<String> = match &self.state {
            InputState::Conversion { candidates, .. } => {
                candidates.candidates().iter().map(|c| c.text.clone()).collect()
            }
            _ => return None,
        };

        let cleaned: String = original
            .chars()
            .filter(|c| !c.is_ascii_alphabetic())
            .collect();
        if !kashiko_engine::contains_kana(&cleaned) {
            return None;
        }

        // Typo region *in the original reading* (stray ASCII included — the
        // leftover keystrokes are evidence for the keyboard edits): around
        // the first ASCII char when the typo was romaji leftovers, else the
        // garbled run of the current top candidate, else the whole reading.
        let orig_chars: Vec<char> = original.chars().collect();
        let (start, end) = match original.chars().position(|c| c.is_ascii_alphabetic()) {
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

        // Phased search with reserved budgets, most → least promising. The
        // keyboard edits get first claim on the result slots: they model how
        // the typo physically happened, and letting the base beam fill the
        // list first was exactly how the intended correction got crowded out.
        //   Phase 1: keyboard-aware romaji-space edits of the typo region
        //            (transpositions / neighbor keys / deletions / vowel
        //            insertions); the most likely edits get a small beam.
        //   Phase 2: plain kana-space deletions and swaps as a coarse net.
        //   Phase 3: the cleaned base with a wide main-model beam — capped
        //            hard, since the quick pass already showed its top picks.
        const MAX_TOTAL_RESULTS: usize = 6;
        const KB_BEAM_VARIANTS: usize = 4;
        const MAX_BASE_RESULTS: usize = 2;

        let kb_variants: Vec<String> =
            kashiko_engine::typo::keyboard_typo_variants(&region, MAX_KEYBOARD_VARIANTS)
                .into_iter()
                .map(|kb| format!("{prefix}{kb}{suffix}"))
                .collect();

        let chars: Vec<char> = cleaned.chars().collect();
        let ascii_before_start = orig_chars[..start]
            .iter()
            .filter(|c| c.is_ascii_alphabetic())
            .count();
        let cstart = start - ascii_before_start;
        let cend = (cstart + (end - start)).min(chars.len());
        let mut kana_variants: Vec<String> = Vec::new();
        for i in cstart..cend.min(chars.len()) {
            kana_variants.push(chars[..i].iter().chain(&chars[i + 1..]).collect());
        }
        let swap_end = cend.min(chars.len()).saturating_sub(1);
        for i in cstart..swap_end {
            if chars[i] != chars[i + 1] {
                let mut v = chars.clone();
                v.swap(i, i + 1);
                kana_variants.push(v.into_iter().collect());
            }
        }

        self.ensure_kanji_converter();
        let ctx = self.truncate_context_for_api();
        let mut seen: HashSet<String> = existing.into_iter().collect();
        let mut uniq: HashSet<String> = HashSet::new();
        let mut found: Vec<Candidate> = Vec::new();

        let run_phase = |engine: &mut Self,
                             variants: &[String],
                             beam_first: usize,
                             beam: usize,
                             phase_cap: usize,
                             found: &mut Vec<Candidate>,
                             seen: &mut HashSet<String>,
                             uniq: &mut HashSet<String>| {
            let mut phase_found = 0usize;
            for (i, variant) in variants.iter().enumerate() {
                if phase_found >= phase_cap || found.len() >= MAX_TOTAL_RESULTS {
                    break;
                }
                if !kashiko_engine::contains_kana(variant)
                    || variant.chars().any(|c| c.is_ascii_alphabetic())
                    || !uniq.insert(variant.clone())
                {
                    continue;
                }
                let n = if i < KB_BEAM_VARIANTS { beam_first } else { beam };
                for out in engine.deep_convert(variant, &ctx, n) {
                    if out != *variant
                        && !kashiko_engine::has_long_unconverted_run(
                            &out,
                            variant,
                            UNCONVERTED_RUN_MIN,
                        )
                        && !kashiko_engine::has_unrenderable_char(&out)
                        && seen.insert(out.clone())
                    {
                        found.push(Candidate {
                            text: out,
                            reading: Some(variant.clone()),
                            source_label: Some(CandidateSource::Model.label().to_string()),
                            description: None,
                        });
                        phase_found += 1;
                        if phase_found >= phase_cap || found.len() >= MAX_TOTAL_RESULTS {
                            break;
                        }
                    }
                }
            }
        };

        // Phase 1: keyboard edits — first claim on the budget.
        run_phase(
            self, &kb_variants, 3, 1, MAX_TOTAL_RESULTS, &mut found, &mut seen, &mut uniq,
        );
        // Phase 2: coarse kana edits fill what's left.
        let remaining = MAX_TOTAL_RESULTS.saturating_sub(found.len());
        run_phase(
            self, &kana_variants, 1, 1, remaining, &mut found, &mut seen, &mut uniq,
        );
        // Phase 3: base beam extras, hard-capped.
        let base = [cleaned.clone()];
        run_phase(
            self, &base, 5, 5, MAX_BASE_RESULTS.min(MAX_TOTAL_RESULTS.saturating_sub(found.len())),
            &mut found, &mut seen, &mut uniq,
        );

        let InputState::Conversion { candidates, .. } = &mut self.state else {
            return None;
        };
        if found.is_empty() {
            // Nothing new — just clear the loading indicator.
            return Some(EngineResult::consumed().with_action(EngineAction::HideAuxText));
        }
        // Insert findings *above* the hiragana/katakana fallback entries so
        // those always stay at the bottom, and keep the user's currently
        // selected candidate selected — the list grows, it never jumps.
        let selected_text = candidates.selected_text().map(str::to_string);
        let mut all: Vec<Candidate> = candidates.candidates().to_vec();
        let insert_at = all
            .iter()
            .position(|c| {
                matches!(c.description.as_deref(), Some("[全]ひらがな") | Some("[全]カタカナ"))
            })
            .unwrap_or(all.len());
        all.splice(insert_at..insert_at, found);
        let mut list = CandidateList::new(all);
        if let Some(sel) = selected_text
            && let Some(idx) = list.candidates().iter().position(|c| c.text == sel)
        {
            list.select(idx);
        }
        *candidates = list.clone();
        Some(
            EngineResult::consumed()
                .with_action(EngineAction::ShowCandidates(list))
                .with_action(EngineAction::HideAuxText),
        )
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
    pub(super) fn build_conversion_candidates(
        &mut self,
        reading: &str,
        num_candidates: usize,
        skip_learning: bool,
        cached_model: Option<Vec<String>>,
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
            Keysym::LEFT => self.resize_conversion(-1),
            Keysym::RIGHT => self.resize_conversion(1),
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

        // Check if there is remaining unconverted text after the split
        let remaining = if let InputState::Conversion { split_index, .. } = self.state {
            let full_len = self.input_buf.text.chars().count();
            if split_index < full_len {
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

        if let Some(remaining) = remaining {
            // Partial commit: commit selected segment, keep remainder for further conversion
            self.state = InputState::Empty;
            self.input_buf.text = remaining;
            self.input_buf.cursor_pos = self.input_buf.text.chars().count();
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

    /// Resize the conversion segment using Left/Right arrow keys
    fn resize_conversion(&mut self, delta: isize) -> EngineResult {
        if let InputState::Conversion { split_index, .. } = self.state {
            let full_len = self.input_buf.text.chars().count();
            let mut new_split = if delta < 0 {
                split_index.saturating_sub(delta.unsigned_abs())
            } else {
                split_index.saturating_add(delta.unsigned_abs())
            };

            if new_split > full_len {
                new_split = 1;
            } else if new_split < 1 {
                new_split = full_len;
            }

            if new_split != split_index {
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
