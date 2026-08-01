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

// --- segment focus movement (Left/Right slide, Shift+Left/Right resize) ---

/// Shift + a non-printable key (used for the resize gesture).
fn press_shift_key(keysym: Keysym) -> KeyEvent {
    KeyEvent::new(keysym, KeyModifiers::new().with_shift(true), true)
}

/// Engine in Conversion state over `reading`, with a user dictionary so the
/// word boundaries are real ones rather than one big span.
fn engine_in_conversion(reading: &str, words: &[(&str, &str)]) -> InputMethodEngine {
    use std::io::Write;
    let mut engine = InputMethodEngine::new();
    engine.converters.kanji = None;
    let entries: Vec<String> = words
        .iter()
        .map(|(r, s)| {
            format!(
                r#"{{"reading":"{r}","candidates":[{{"surface":"{s}","score":1.0}}]}}"#
            )
        })
        .collect();
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.write_all(format!("[{}]", entries.join(",")).as_bytes()).unwrap();
    tmp.flush().unwrap();
    engine.dicts.user = Some(kashiko_engine::dict::Dictionary::build_from_json(tmp.path()).unwrap());
    engine.input_buf.insert(reading);
    engine.start_conversion(false, None, 0);
    engine
}

#[test]
fn right_settles_the_segment_and_focuses_the_next_word() {
    // `あい` + `うえ` are dictionary words, so they form two segments.
    let mut engine = engine_in_conversion("あいうえ", &[("あい", "藍"), ("うえ", "上")]);

    // Initial conversion covers the whole reading.
    let InputState::Conversion {
        split_index,
        segment_start,
        ..
    } = engine.state
    else {
        panic!("expected Conversion state");
    };
    assert_eq!((segment_start, split_index), (0, 4));

    // Shrink to the first word, then slide right: the focus window MOVES
    // rather than growing from position 0.
    engine.process_key(&press_shift_key(Keysym::LEFT));
    let InputState::Conversion {
        segment_start,
        split_index,
        ..
    } = engine.state
    else {
        panic!("expected Conversion state");
    };
    assert_eq!(segment_start, 0, "resize keeps the window anchored at 0");
    let first_end = split_index;
    assert!(first_end < 4, "Shift+Left should shrink the segment");

    engine.process_key(&press_key(Keysym::RIGHT));
    let InputState::Conversion {
        segment_start,
        split_index,
        ..
    } = engine.state
    else {
        panic!("expected Conversion state");
    };
    assert_eq!(
        segment_start, first_end,
        "Right must move the window's start, not keep it at 0"
    );
    assert!(split_index > segment_start);
    assert_eq!(engine.decided.len(), 1, "the passed segment is settled");
}

#[test]
fn left_returns_focus_to_the_previous_segment() {
    let mut engine = engine_in_conversion("あいうえ", &[("あい", "藍"), ("うえ", "上")]);
    engine.process_key(&press_shift_key(Keysym::LEFT));
    engine.process_key(&press_key(Keysym::RIGHT));
    assert_eq!(engine.decided.len(), 1);

    engine.process_key(&press_key(Keysym::LEFT));
    let InputState::Conversion { segment_start, .. } = engine.state else {
        panic!("expected Conversion state");
    };
    assert_eq!(segment_start, 0, "Left brings the focus back to the start");
    assert!(engine.decided.is_empty(), "the settled choice is popped");
}

#[test]
fn commit_includes_segments_settled_by_sliding() {
    let mut engine = engine_in_conversion("あいうえ", &[("あい", "藍"), ("うえ", "上")]);
    engine.process_key(&press_shift_key(Keysym::LEFT));
    engine.process_key(&press_key(Keysym::RIGHT));

    let result = engine.process_key(&press_key(Keysym::RETURN));
    let committed: String = result
        .actions
        .iter()
        .find_map(|a| match a {
            EngineAction::Commit(text) => Some(text.clone()),
            _ => None,
        })
        .expect("Enter should commit");
    // Whatever the candidates were, the settled segment must not be lost:
    // the commit has to cover the whole reading, not just the focused part.
    assert!(
        committed.chars().count() >= 2,
        "commit dropped the settled segment: {committed:?}"
    );
    assert!(engine.decided.is_empty());
}

#[test]
fn arrows_work_straight_after_space() {
    // The reported bug: right after Space the whole reading is focused
    // (split_index == total), and both arrow handlers bailed out early, so
    // neither key did anything. The first Right must split into words.
    let mut engine = engine_in_conversion("あいうえ", &[("あい", "藍"), ("うえ", "上")]);
    engine.process_key(&press_key(Keysym::RIGHT));
    let InputState::Conversion {
        segment_start,
        split_index,
        ..
    } = engine.state
    else {
        panic!("expected Conversion state");
    };
    assert_eq!(segment_start, 0);
    assert!(
        split_index < 4,
        "the first Right should focus the first word, got split_index={split_index}"
    );

    // Left from the first word widens back out to the whole reading.
    engine.process_key(&press_key(Keysym::LEFT));
    let InputState::Conversion {
        segment_start,
        split_index,
        ..
    } = engine.state
    else {
        panic!("expected Conversion state");
    };
    assert_eq!((segment_start, split_index), (0, 4));
}

#[test]
fn focus_uses_conversion_derived_phrase_boundaries() {
    // The reported bug: for `今日の天気は晴れです` the dictionary read the
    // reading as `…てんき|はは|れです` (母 is a real word), so the first Right
    // focused the nonsense span `はは`. Aligning against the conversion gives
    // the real phrases: 今日の | 天気は | 晴れです.
    let reading = "きょうのてんきははれです";
    let mut engine = InputMethodEngine::new();
    engine.converters.kanji = None;
    engine.input_buf.insert(reading);
    // Enter Conversion with the sentence conversion as the selected candidate,
    // which is what the model produces on Space.
    engine.state = InputState::Conversion {
        preedit: Preedit::new(),
        candidates: CandidateList::new(vec![Candidate {
            text: "今日の天気は晴れです".to_string(),
            reading: Some(reading.to_string()),
            source_label: None,
            description: None,
        }]),
        split_index: reading.chars().count(),
        segment_start: 0,
    };

    engine.process_key(&press_key(Keysym::RIGHT));
    let InputState::Conversion {
        segment_start,
        split_index,
        ..
    } = engine.state
    else {
        panic!("expected Conversion state");
    };
    assert_eq!(
        (segment_start, split_index),
        (0, 4),
        "first phrase should be きょうの (今日の), not a dictionary mis-split"
    );
    assert_eq!(engine.segment_bounds, vec![4, 8, 12]);
}
