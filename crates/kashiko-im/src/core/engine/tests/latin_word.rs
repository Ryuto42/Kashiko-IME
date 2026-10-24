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
    assert!(
        !(texts.last().map(String::as_str) == Some("WordPress")),
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
            format!(r#"{{"reading":"{r}","candidates":[{{"surface":"{s}","score":1.0}}]}}"#)
        })
        .collect();
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.write_all(format!("[{}]", entries.join(",")).as_bytes())
        .unwrap();
    tmp.flush().unwrap();
    engine.dicts.user =
        Some(kashiko_engine::dict::Dictionary::build_from_json(tmp.path()).unwrap());
    engine.input_buf.insert(reading);
    engine.start_conversion(false, None, 0);
    engine
}

/// As `engine_in_conversion`, with `selected` as the conversion on screen.
fn engine_showing(reading: &str, words: &[(&str, &str)], selected: &str) -> InputMethodEngine {
    let mut engine = engine_in_conversion(reading, words);
    engine.state = InputState::Conversion {
        preedit: Preedit::new(),
        candidates: CandidateList::new(vec![Candidate {
            text: selected.to_string(),
            reading: Some(reading.to_string()),
            source_label: None,
            description: None,
        }]),
        split_index: reading.chars().count(),
        segment_start: 0,
    };
    engine
}

fn focus_span(engine: &InputMethodEngine) -> (usize, usize) {
    let InputState::Conversion {
        segment_start,
        split_index,
        ..
    } = engine.state
    else {
        panic!("expected Conversion state");
    };
    (segment_start, split_index)
}

fn preedit_text(engine: &InputMethodEngine) -> String {
    engine
        .state
        .preedit()
        .map(|p| p.text().to_string())
        .unwrap_or_default()
}

#[test]
fn arrows_move_the_focus_through_phrases_without_changing_them() {
    // `あい` + `うえ` are dictionary words, so they form two phrases.
    let mut engine = engine_showing("あいうえ", &[("あい", "藍"), ("うえ", "上")], "藍上");
    assert_eq!(
        focus_span(&engine),
        (0, 4),
        "Space converts the whole reading"
    );
    let shown = "藍上";

    engine.process_key(&press_key(Keysym::RIGHT));
    assert_eq!(
        focus_span(&engine),
        (0, 2),
        "the first Right focuses the first phrase"
    );
    assert_eq!(engine.phrases.len(), 2);
    assert_eq!(
        preedit_text(&engine),
        shown,
        "opening into phrases changes nothing"
    );

    engine.process_key(&press_key(Keysym::RIGHT));
    assert_eq!(focus_span(&engine), (2, 4));
    assert_eq!(
        preedit_text(&engine),
        shown,
        "moving the focus changes nothing"
    );

    engine.process_key(&press_key(Keysym::RIGHT));
    assert_eq!(
        focus_span(&engine),
        (2, 4),
        "Right stops at the last phrase"
    );

    engine.process_key(&press_key(Keysym::LEFT));
    assert_eq!(focus_span(&engine), (0, 2));
    engine.process_key(&press_key(Keysym::LEFT));
    assert_eq!(
        focus_span(&engine),
        (0, 2),
        "Left stops at the first phrase"
    );
}

#[test]
fn a_phrase_choice_survives_moving_away_and_commits_with_the_rest() {
    let mut engine = engine_showing("あいうえ", &[("あい", "藍"), ("うえ", "上")], "あいうえ");
    engine.process_key(&press_key(Keysym::RIGHT));
    // Pick 藍 for the first phrase, then move on.
    let InputState::Conversion { candidates, .. } = &engine.state else {
        panic!("expected Conversion state");
    };
    let at = candidates
        .candidates()
        .iter()
        .position(|c| c.text == "藍")
        .expect("藍 offered");
    for _ in 0..at {
        engine.process_key(&press_key(Keysym::SPACE));
    }
    engine.process_key(&press_key(Keysym::RIGHT));
    assert!(
        preedit_text(&engine).starts_with("藍"),
        "{}",
        preedit_text(&engine)
    );
    engine.process_key(&press_key(Keysym::LEFT));
    let InputState::Conversion { candidates, .. } = &engine.state else {
        panic!("expected Conversion state");
    };
    assert_eq!(
        candidates.selected_text(),
        Some("藍"),
        "the choice is kept on top"
    );

    let result = engine.process_key(&press_key(Keysym::RETURN));
    let committed: String = result
        .actions
        .iter()
        .find_map(|a| match a {
            EngineAction::Commit(text) => Some(text.clone()),
            _ => None,
        })
        .expect("Enter should commit");
    assert_eq!(committed, "藍うえ");
    assert!(engine.phrases.is_empty());
}

#[test]
fn shift_arrows_move_a_phrase_boundary_by_one_character() {
    let mut engine = engine_in_conversion("あいうえ", &[("あい", "藍"), ("うえ", "上")]);
    engine.process_key(&press_key(Keysym::RIGHT));
    engine.process_key(&press_shift_key(Keysym::RIGHT));
    assert_eq!(focus_span(&engine), (0, 3));
    let lens: Vec<usize> = engine.phrases.iter().map(|p| p.0).collect();
    assert_eq!(lens, vec![3, 1]);

    engine.process_key(&press_shift_key(Keysym::RIGHT));
    let lens: Vec<usize> = engine.phrases.iter().map(|p| p.0).collect();
    assert_eq!(lens, vec![4], "the emptied phrase is dropped");

    engine.process_key(&press_shift_key(Keysym::LEFT));
    engine.process_key(&press_shift_key(Keysym::LEFT));
    let lens: Vec<usize> = engine.phrases.iter().map(|p| p.0).collect();
    assert_eq!(
        lens,
        vec![2, 2],
        "shrinking the last phrase starts a new one"
    );
    assert_eq!(engine.phrases.iter().map(|p| p.0).sum::<usize>(), 4);
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
    assert_eq!(
        focus_span(&engine),
        (0, 4),
        "first phrase should be きょうの (今日の), not a dictionary mis-split"
    );
    let phrases: Vec<(usize, &str)> = engine
        .phrases
        .iter()
        .map(|(l, t)| (*l, t.as_str()))
        .collect();
    assert_eq!(phrases, vec![(4, "今日の"), (4, "天気は"), (4, "晴れです")]);
    assert_eq!(preedit_text(&engine), "今日の天気は晴れです");
}

#[test]
fn katakana_after_hiragana_starts_a_phrase_and_punctuation_ends_one() {
    let engine = InputMethodEngine::new();
    let phrases = engine
        .sentence_phrases(
            "たまにくろーどがなにもさわって",
            "たまにクロードがなにも触って",
        )
        .unwrap();
    assert_eq!(
        phrases,
        vec![
            (3, "たまに".to_string()),
            (8, "クロードがなにも".to_string()),
            (4, "触って".to_string()),
        ]
    );
    let phrases = engine
        .sentence_phrases("はい、そうです。", "はい、そうです。")
        .unwrap();
    let texts: Vec<&str> = phrases.iter().map(|p| p.1.as_str()).collect();
    assert!(
        texts.iter().all(|t| !t.starts_with(['、', '。'])),
        "{texts:?}"
    );
    assert_eq!(texts.concat(), "はい、そうです。");
}

// --- dictionary substitution: filling the recall gap ---

#[test]
fn homophone_of_an_inflected_verb_is_offered_as_a_whole_sentence() {
    // The reported case: `雑誌にのります` converts to `雑誌に乗ります` and
    // `載ります` never appears, so no amount of reranking can pick it. The
    // dictionary is only consulted by exact match on the whole reading, which
    // matches nothing here — this fills that gap.
    //
    // `のります`/`乗ります` share the suffix `ります`, so only the stem `の` is
    // looked up and swapped, keeping the inflection intact.
    let engine = engine_in_conversion_dict(&[("の", "載"), ("ざっし", "雑誌")]);
    let variants = engine.dictionary_substitution_variants("ざっしにのります", "雑誌に乗ります");
    assert!(
        variants.contains(&"雑誌に載ります".to_string()),
        "expected 雑誌に載ります among variants, got {variants:?}"
    );
}

#[test]
fn substitution_leaves_pure_kana_phrases_alone() {
    // A phrase that was never converted has no homophone to offer.
    let engine = engine_in_conversion_dict(&[("の", "載")]);
    let variants = engine.dictionary_substitution_variants("あいうえ", "あいうえ");
    assert!(variants.is_empty(), "got {variants:?}");
}

#[test]
fn substitution_never_repeats_the_original() {
    let engine = engine_in_conversion_dict(&[("ざっし", "雑誌")]);
    let variants = engine.dictionary_substitution_variants("ざっしに", "雑誌に");
    assert!(
        !variants.contains(&"雑誌に".to_string()),
        "the input conversion must not be offered back: {variants:?}"
    );
}

/// Engine with a user dictionary of `(reading, surface)` entries, no model.
fn engine_in_conversion_dict(words: &[(&str, &str)]) -> InputMethodEngine {
    use std::io::Write;
    let mut engine = InputMethodEngine::new();
    engine.converters.kanji = None;
    let entries: Vec<String> = words
        .iter()
        .map(|(r, s)| {
            format!(r#"{{"reading":"{r}","candidates":[{{"surface":"{s}","score":1.0}}]}}"#)
        })
        .collect();
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.write_all(format!("[{}]", entries.join(",")).as_bytes())
        .unwrap();
    tmp.flush().unwrap();
    engine.dicts.user =
        Some(kashiko_engine::dict::Dictionary::build_from_json(tmp.path()).unwrap());
    engine
}
