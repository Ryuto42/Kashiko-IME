//! F6–F10 character-form conversion while composing and converting.

use kashiko_engine::LearningCache;

use super::*;

fn type_str(engine: &mut InputMethodEngine, s: &str) {
    for ch in s.chars() {
        engine.process_key(&press(ch));
    }
}

fn preedit(engine: &InputMethodEngine) -> String {
    engine.preedit().unwrap().text().to_string()
}

fn committed(result: &EngineResult) -> Option<&str> {
    result.actions.iter().find_map(|a| match a {
        EngineAction::Commit(text) => Some(text.as_str()),
        _ => None,
    })
}

fn engine_with_cache() -> InputMethodEngine {
    let mut engine = InputMethodEngine::new();
    engine.learning = Some(LearningCache::new(100));
    engine
}

#[test]
fn f6_f7_f8_change_the_form_and_enter_commits_it() {
    let cases = [
        (Keysym::F6, "かんじ"),
        (Keysym::F7, "カンジ"),
        (Keysym::F8, "ｶﾝｼﾞ"),
    ];
    for (key, expected) in cases {
        let mut engine = InputMethodEngine::new();
        type_str(&mut engine, "kanji");
        let result = engine.process_key(&press_key(key));
        assert!(result.consumed);
        assert!(matches!(engine.state(), InputState::Composing { .. }));
        assert_eq!(preedit(&engine), expected);

        let result = engine.process_key(&press_key(Keysym::RETURN));
        assert_eq!(committed(&result), Some(expected));
        assert!(matches!(engine.state(), InputState::Empty));
    }
}

#[test]
fn pending_romaji_is_flushed_before_converting() {
    let mut engine = InputMethodEngine::new();
    type_str(&mut engine, "kan");
    engine.process_key(&press_key(Keysym::F7));
    assert_eq!(preedit(&engine), "カン");
}

#[test]
fn f9_cycles_lower_upper_capitalized() {
    let mut engine = InputMethodEngine::new();
    type_str(&mut engine, "kanji");
    for expected in ["ｋａｎｊｉ", "ＫＡＮＪＩ", "Ｋａｎｊｉ", "ｋａｎｊｉ"] {
        engine.process_key(&press_key(Keysym::F9));
        assert_eq!(preedit(&engine), expected);
    }
}

#[test]
fn f10_cycles_and_restarts_when_the_key_changes() {
    let mut engine = InputMethodEngine::new();
    type_str(&mut engine, "kanji");
    engine.process_key(&press_key(Keysym::F10));
    assert_eq!(preedit(&engine), "kanji");
    engine.process_key(&press_key(Keysym::F10));
    assert_eq!(preedit(&engine), "KANJI");
    engine.process_key(&press_key(Keysym::F10));
    assert_eq!(preedit(&engine), "Kanji");

    engine.process_key(&press_key(Keysym::F9));
    assert_eq!(preedit(&engine), "ｋａｎｊｉ");
    engine.process_key(&press_key(Keysym::F10));
    assert_eq!(preedit(&engine), "kanji");
}

#[test]
fn f9_commits_the_full_width_text() {
    let mut engine = InputMethodEngine::new();
    type_str(&mut engine, "ab1");
    engine.process_key(&press_key(Keysym::F9));
    let result = engine.process_key(&press_key(Keysym::RETURN));
    assert_eq!(committed(&result), Some("ａｂ１"));
}

#[test]
fn f9_does_nothing_when_keystrokes_no_longer_match_the_kana() {
    let mut engine = InputMethodEngine::new();
    type_str(&mut engine, "aii");
    engine.process_key(&press_key(Keysym::BACKSPACE));
    assert_eq!(preedit(&engine), "あい");
    assert!(!engine.input_buf.raw_synced);

    let result = engine.process_key(&press_key(Keysym::F9));
    assert!(result.consumed);
    assert_eq!(preedit(&engine), "あい");
    let result = engine.process_key(&press_key(Keysym::RETURN));
    assert_eq!(committed(&result), Some("あい"));
}

#[test]
fn f9_converts_the_width_of_what_is_there_without_keystrokes() {
    let mut engine = InputMethodEngine::new();
    type_str(&mut engine, "aii");
    engine.process_key(&press_key(Keysym::BACKSPACE));
    engine.input_buf.set_text("あ1b".to_string());
    engine.process_key(&press_key(Keysym::F9));
    assert_eq!(preedit(&engine), "あ１ｂ");
    engine.process_key(&press_key(Keysym::F10));
    assert_eq!(preedit(&engine), "あ1b");
}

#[test]
fn escape_returns_to_the_reading() {
    let mut engine = InputMethodEngine::new();
    type_str(&mut engine, "kanji");
    engine.process_key(&press_key(Keysym::F7));
    let result = engine.process_key(&press_key(Keysym::ESCAPE));
    assert!(result.consumed);
    assert_eq!(preedit(&engine), "かんじ");
    assert!(matches!(engine.state(), InputState::Composing { .. }));
}

#[test]
fn typing_after_a_form_resumes_from_the_reading() {
    let mut engine = InputMethodEngine::new();
    type_str(&mut engine, "kanji");
    engine.process_key(&press_key(Keysym::F7));
    engine.process_key(&press('a'));
    assert_eq!(preedit(&engine), "かんじあ");
    let result = engine.process_key(&press_key(Keysym::RETURN));
    assert_eq!(committed(&result), Some("かんじあ"));
}

#[test]
fn f6_in_katakana_mode_commits_hiragana() {
    let mut engine = InputMethodEngine::new();
    type_str(&mut engine, "kanji");
    engine.process_key(&press_ctrl(Keysym::KEY_K));
    assert_eq!(preedit(&engine), "カンジ");
    engine.process_key(&press_key(Keysym::F6));
    assert_eq!(preedit(&engine), "かんじ");
    let result = engine.process_key(&press_key(Keysym::RETURN));
    assert_eq!(committed(&result), Some("かんじ"));
}

#[test]
fn conversion_state_takes_f_keys_on_the_whole_reading() {
    let mut engine = InputMethodEngine::new();
    type_str(&mut engine, "kanji");
    engine.process_key(&press_key(Keysym::SPACE));
    assert!(matches!(engine.state(), InputState::Conversion { .. }));

    let result = engine.process_key(&press_key(Keysym::F7));
    assert!(result.consumed);
    assert!(
        result
            .actions
            .iter()
            .any(|a| matches!(a, EngineAction::HideCandidates))
    );
    assert!(matches!(engine.state(), InputState::Composing { .. }));
    assert_eq!(preedit(&engine), "カンジ");

    let result = engine.process_key(&press_key(Keysym::RETURN));
    assert_eq!(committed(&result), Some("カンジ"));
}

#[test]
fn conversion_state_f9_uses_the_keystrokes() {
    let mut engine = InputMethodEngine::new();
    type_str(&mut engine, "kanji");
    engine.process_key(&press_key(Keysym::SPACE));
    engine.process_key(&press_key(Keysym::F9));
    assert_eq!(preedit(&engine), "ｋａｎｊｉ");
    engine.process_key(&press_key(Keysym::F9));
    assert_eq!(preedit(&engine), "ＫＡＮＪＩ");
}

#[test]
fn committed_form_is_learned() {
    let mut engine = engine_with_cache();
    type_str(&mut engine, "kanji");
    engine.process_key(&press_key(Keysym::F7));
    engine.process_key(&press_key(Keysym::RETURN));

    let learned = engine.learning.as_ref().unwrap().lookup("かんじ");
    assert!(learned.iter().any(|(surface, _)| surface == "カンジ"));
}

#[test]
fn f_keys_are_ignored_in_emoji_mode() {
    let mut engine = InputMethodEngine::new();
    engine.process_key(&press(':'));
    type_str(&mut engine, "smile");
    let before = preedit(&engine);
    let result = engine.process_key(&press_key(Keysym::F7));
    assert!(result.consumed);
    assert_eq!(preedit(&engine), before);
}
