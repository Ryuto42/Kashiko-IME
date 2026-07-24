//! Tests for the latin-word candidate: recovering `WordPress` from the buffer
//! `WおrdPレッ` that the romaji converter produced from those keystrokes.
//!
//! The ranking matters as much as the detection — the candidate has to land
//! *among* the real conversions in confidence order, not pinned to either end.

use kashiko_engine::LearningCache;

use super::*;

/// Engine with the keystroke history and buffer set directly, no kanji model.
/// Driving real key events would depend on the romaji tables producing exactly
/// the mangled buffer; these tests assert the candidate/ranking logic.
fn engine_with_keys(raw: &str, buffer: &str) -> InputMethodEngine {
    let mut engine = InputMethodEngine::new();
    engine.converters.kanji = None;
    engine.input_buf.insert(buffer);
    engine.input_buf.raw_keys = raw.to_string();
    engine
}

#[test]
fn mangled_latin_word_is_offered() {
    let engine = engine_with_keys("WordPress", "WおrdPレッ");
    let (text, score) = engine
        .latin_word_candidate()
        .expect("WordPress should be recovered from the keystrokes");
    assert_eq!(text, "WordPress");
    assert!(score >= 10, "expected a strong score, got {score}");
}

#[test]
fn ordinary_japanese_input_offers_nothing() {
    // The regression this guards: ordinary Japanese keystrokes are pure ASCII
    // romaji too, so a naive check offered `watasi` on every conversion — and
    // once learned it would have hijacked live conversion from `私`.
    let engine = engine_with_keys("watasi", "わたし");
    assert!(engine.latin_word_candidate().is_none());
}

#[test]
fn single_key_typo_offers_nothing() {
    // `wたし` is a missed vowel; the keyboard-typo search owns it.
    let engine = engine_with_keys("wtasi", "wたし");
    assert!(engine.latin_word_candidate().is_none());
}

#[test]
fn mixed_sentence_with_intact_latin_offers_nothing() {
    let engine = engine_with_keys("watasihaLinuxga", "わたしはLinuxが");
    assert!(engine.latin_word_candidate().is_none());
}

#[test]
fn previously_committed_word_outranks_a_fresh_guess() {
    let fresh = engine_with_keys("WordPress", "WおrdPレッ")
        .latin_word_candidate()
        .unwrap()
        .1;

    let mut engine = engine_with_keys("WordPress", "WおrdPレッ");
    let mut cache = LearningCache::new(100);
    cache.record("WおrdPレッ", "WordPress");
    engine.learning = Some(cache);
    let learned = engine.latin_word_candidate().unwrap().1;

    assert!(
        learned > fresh,
        "a committed word must outrank a guess: learned={learned} fresh={fresh}"
    );
}

#[test]
fn candidate_lands_among_normal_candidates_by_confidence() {
    // Known word → top of the list.
    let mut engine = engine_with_keys("WordPress", "WおrdPレッ");
    let mut cache = LearningCache::new(100);
    cache.record("WおrdPレッ", "WordPress");
    engine.learning = Some(cache);
    let texts: Vec<String> = engine
        .build_conversion_candidates("おレッ", 9, false, None, true)
        .into_iter()
        .map(|c| c.text)
        .collect();
    assert_eq!(
        texts.first().map(String::as_str),
        Some("WordPress"),
        "a proven latin word should lead, got {texts:?}"
    );

    // Unproven guess → present, but neither first nor dead last (the
    // hiragana/katakana fallbacks stay at the bottom).
    let mut engine = engine_with_keys("WordPress", "WおrdPレッ");
    let texts: Vec<String> = engine
        .build_conversion_candidates("おレッ", 9, false, None, true)
        .into_iter()
        .map(|c| c.text)
        .collect();
    let at = texts.iter().position(|t| t == "WordPress");
    assert!(at.is_some(), "guess should still be offered, got {texts:?}");
    assert_eq!(
        texts.last().map(String::as_str) == Some("WordPress"),
        false,
        "guess must not be pinned to the bottom, got {texts:?}"
    );
}

#[test]
fn prefix_scoped_conversion_never_offers_the_whole_buffer_word() {
    // A Left/Right-resized segment converts only a prefix; offering the
    // whole-buffer keystrokes there would commit text outside that scope.
    let mut engine = engine_with_keys("WordPress", "WおrdPレッ");
    let texts: Vec<String> = engine
        .build_conversion_candidates("おレッ", 9, false, None, false)
        .into_iter()
        .map(|c| c.text)
        .collect();
    assert!(
        !texts.iter().any(|t| t == "WordPress"),
        "prefix scope must not include the latin word, got {texts:?}"
    );
}

#[test]
fn partial_commit_drops_stale_keystrokes() {
    // After a partial commit the buffer holds only the remainder, so the
    // original keystrokes no longer describe it and must be discarded.
    let mut engine = engine_with_keys("WordPress", "WおrdPレッ");
    engine.input_buf.set_text("レッ".to_string());
    assert!(engine.input_buf.raw_keys.is_empty());
    assert!(engine.latin_word_candidate().is_none());
}
