//! English words typed in the middle of romaji stay in Latin script
//! (`kyouhagoogledekensaku` → きょうはgoogleでけんさく). The decision is made by
//! `kashiko_engine::mixed`; this rebuilds the composition from the keystrokes
//! accordingly and handles what follows from it.

use kashiko_engine::mixed::{self, Learned, Options, Segment};

use super::*;

/// Per-composition state of the English detection.
#[derive(Debug, Default)]
pub(super) struct MixedInput {
    /// English words (as typed) in the current composition.
    pub words: Vec<String>,
    /// Candidates that read the composition as all-Japanese; picking one
    /// tells us those words were not English.
    pub kana_alternatives: Vec<String>,
}

impl InputMethodEngine {
    /// Rebuild the composition from the keystrokes with English words kept as
    /// typed. Called after each key typed in hiragana mode; does nothing
    /// unless the buffer is still exactly what the keystrokes produced.
    pub(super) fn apply_auto_english(&mut self) {
        let len = self.input_buf.text.chars().count();
        if self.input_mode != InputMode::Hiragana
            || !self.input_buf.raw_synced
            || self.input_buf.cursor_pos != len
            || !self.input_buf.raw_keys.chars().any(|c| c.is_ascii_alphabetic())
        {
            return;
        }
        let (chosen, rejected) = self
            .learning
            .as_ref()
            .map(|c| c.latin_word_choices())
            .unwrap_or_default();
        let learned = |w: &str| {
            if rejected.contains(w) {
                Some(Learned::Japanese)
            } else if chosen.contains_key(w) {
                Some(Learned::English)
            } else {
                None
            }
        };
        let is_japanese_word = |kana: &str| {
            [self.dicts.user.as_ref(), self.dicts.system.as_ref()]
                .into_iter()
                .flatten()
                .any(|d| d.exact_match_search(kana).is_some())
        };
        let segments = mixed::split(
            &self.input_buf.raw_keys,
            &Options {
                english_context: self.english_context(),
                learned: &learned,
                is_japanese_word: &is_japanese_word,
            },
        );
        let words: Vec<String> = segments
            .iter()
            .filter_map(|s| match s {
                Segment::English(w) => Some(w.clone()),
                Segment::Japanese(_) => None,
            })
            .collect();
        // Plain Japanese that never had an English word: leave the buffer
        // exactly as the romaji converter built it.
        if words.is_empty() && self.mixed.words.is_empty() {
            return;
        }
        // Only a spelling with capitals counts: a lowercase one is what the
        // word looked like before usual spellings existed, not a preference.
        let learned_spelling = |w: &str| chosen.get(w).filter(|s| s.as_str() != w).cloned();
        let (text, pending) = mixed::render(&segments, &learned_spelling);
        self.input_buf.set_rendered(text);
        self.converters.romaji.reset();
        for c in pending.chars() {
            self.converters.romaji.push(c);
        }
        self.mixed.words = words;
    }

    /// The text before the cursor is English (`I want `): two or more latin
    /// words ending in a space.
    fn english_context(&self) -> bool {
        let ctx = self.truncate_context_for_api();
        if !ctx.ends_with(' ') {
            return false;
        }
        let words: Vec<&str> = ctx.split_whitespace().rev().take(2).collect();
        words.len() == 2
            && words
                .iter()
                .all(|w| w.chars().all(|c| c.is_ascii_alphabetic() || matches!(c, '\'' | ',')))
    }

    /// The composition read as all-Japanese, and its conversion: offered
    /// when English words were kept, in case one of them was meant as kana.
    pub(super) fn kana_alternatives(&mut self) -> Vec<AnnotatedCandidate> {
        self.mixed.kana_alternatives.clear();
        if self.mixed.words.is_empty() || !self.input_buf.raw_synced {
            return Vec::new();
        }
        let kana = mixed::all_kana(&self.input_buf.raw_keys);
        let ctx = self.truncate_context_for_api();
        let mut out = Vec::new();
        if let Some(converted) = self.run_kana_kanji_conversion(&kana, &ctx, 1).into_iter().next()
            && converted != kana
        {
            out.push(converted);
        }
        out.push(kana);
        self.mixed.kana_alternatives = out.clone();
        out.into_iter()
            .map(|t| AnnotatedCandidate::new(t, CandidateSource::Model))
            .collect()
    }

    /// Learn from a committed candidate: picking a kana reading means the
    /// English words in this composition were Japanese after all.
    pub(super) fn learn_from_mixed_commit(&mut self, committed: &str) {
        if !self.secure_input
            && self.mixed.kana_alternatives.iter().any(|k| k == committed)
            && let Some(cache) = self.learning.as_mut()
        {
            for word in &self.mixed.words {
                cache.suppress(&mixed::all_kana(word), word);
            }
        }
        self.mixed = MixedInput::default();
    }
}
