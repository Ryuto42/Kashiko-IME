//! English words typed inside romaji stay in Latin script.

use super::*;

fn type_str(engine: &mut InputMethodEngine, s: &str) {
    for c in s.chars() {
        engine.process_key(&press(c));
    }
}

#[test]
fn english_words_inside_romaji_stay_latin() {
    let mut engine = InputMethodEngine::new();
    type_str(&mut engine, "kyouhagoogledekensaku");
    assert_eq!(engine.input_buf.text, "きょうはGoogleでけんさく");
}

#[test]
fn plain_romaji_is_untouched() {
    let mut engine = InputMethodEngine::new();
    type_str(&mut engine, "watashihanihonjindesu");
    assert_eq!(engine.input_buf.text, "わたしはにほんじんです");
}

#[test]
fn space_after_an_english_word_converts() {
    let mut engine = InputMethodEngine::new();
    type_str(&mut engine, "github");
    engine.process_key(&press_key(Keysym::SPACE));
    assert!(matches!(engine.state(), InputState::Conversion { .. }));
}

#[test]
fn a_word_rejected_before_stays_kana() {
    let mut engine = InputMethodEngine::new();
    let mut cache = kashiko_engine::LearningCache::new(100);
    cache.suppress("ごおｇぇ", "google");
    engine.learning = Some(cache);
    type_str(&mut engine, "google");
    assert!(
        !engine.input_buf.text.contains("google"),
        "{}",
        engine.input_buf.text
    );
}

#[test]
fn the_kana_reading_is_offered_and_picking_it_is_learned() {
    let mut engine = InputMethodEngine::new();
    engine.learning = Some(kashiko_engine::LearningCache::new(100));
    type_str(&mut engine, "googlede");
    let kana = kashiko_engine::mixed::all_kana("googlede");
    engine.process_key(&press_key(Keysym::SPACE));
    let InputState::Conversion { candidates, .. } = engine.state() else {
        panic!("expected conversion");
    };
    assert!(
        candidates.candidates().iter().any(|c| c.text == kana),
        "{kana} not offered: {:?}",
        candidates.candidates()
    );
    for _ in 0..60 {
        match engine.state() {
            InputState::Conversion { candidates, .. }
                if candidates.selected_text() != Some(kana.as_str()) =>
            {
                engine.process_key(&press_key(Keysym::DOWN));
            }
            _ => break,
        }
    }
    engine.process_key(&press_key(Keysym::RETURN));
    let (_, rejected) = engine.learning.as_ref().unwrap().latin_word_choices();
    assert!(rejected.contains("google"), "{rejected:?}");
}

#[test]
fn a_latin_word_chosen_for_other_kana_does_not_count() {
    // `You` picked as the conversion of ゆー (katakana-style reading) once
    // turned `younakousei` into `Youなこうせい`.
    let mut engine = InputMethodEngine::new();
    let mut cache = kashiko_engine::LearningCache::new(100);
    cache.record("ゆー", "You");
    cache.record("いん", "in");
    engine.learning = Some(cache);
    type_str(&mut engine, "younakousei");
    assert_eq!(engine.input_buf.text, "ようなこうせい");
    engine.reset();
    type_str(&mut engine, "kaiinkai");
    assert_eq!(engine.input_buf.text, "かいいんかい");
}

#[test]
fn a_latin_word_chosen_for_its_keystrokes_is_learned() {
    let mut engine = InputMethodEngine::new();
    let mut cache = kashiko_engine::LearningCache::new(100);
    cache.record("あぴ", "API");
    engine.learning = Some(cache);
    type_str(&mut engine, "apinokagi");
    assert_eq!(engine.input_buf.text, "APIのかぎ");
}
