//! `.` `,` `-` right after a half-width digit stay half-width.

use super::*;

fn typed(s: &str) -> String {
    let mut engine = InputMethodEngine::new();
    for ch in s.chars() {
        engine.process_key(&press(ch));
    }
    engine.preedit().unwrap().text().to_string()
}

#[test]
fn punctuation_after_digit_is_half_width() {
    assert_eq!(typed("3.14"), "3.14");
    assert_eq!(typed("1,000"), "1,000");
    assert_eq!(typed("03-1234"), "03-1234");
}

#[test]
fn punctuation_after_kana_is_full_width() {
    assert_eq!(typed("a."), "あ。");
    assert_eq!(typed("a,"), "あ、");
    assert_eq!(typed("ka-"), "かー");
}

#[test]
fn only_the_character_before_the_cursor_counts() {
    let mut engine = InputMethodEngine::new();
    for ch in "a3".chars() {
        engine.process_key(&press(ch));
    }
    engine.process_key(&press_key(Keysym::HOME));
    engine.process_key(&press('.'));
    assert_eq!(engine.preedit().unwrap().text(), "。あ3");

    engine.process_key(&press_key(Keysym::END));
    engine.process_key(&press('.'));
    assert_eq!(engine.preedit().unwrap().text(), "。あ3.");
}

#[test]
fn backspace_removes_the_half_width_mark() {
    let mut engine = InputMethodEngine::new();
    for ch in "3.1".chars() {
        engine.process_key(&press(ch));
    }
    engine.process_key(&press_key(Keysym::BACKSPACE));
    engine.process_key(&press_key(Keysym::BACKSPACE));
    assert_eq!(engine.preedit().unwrap().text(), "3");
    engine.process_key(&press('.'));
    assert_eq!(engine.preedit().unwrap().text(), "3.");
}
