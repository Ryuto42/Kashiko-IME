//! Verification against the real dictionary and model.
//!
//! Ignored by default (they download/load a GGUF and the system dictionary).
//! Run serially — each test loads its own model and they share the on-disk
//! learning cache, so in parallel they interfere:
//!
//!     cargo test -p kashiko-im --lib real_model -- --ignored --nocapture --test-threads=1

use super::*;
use crate::config::Settings;

fn real_engine() -> InputMethodEngine {
    let settings = Settings::load().unwrap_or_default();
    let mut e = InputMethodEngine::with_config(EngineConfig::from_settings(&settings));
    e.init_from_settings(&settings).expect("engine init");
    e
}

/// Candidate texts the engine offers for `reading` in `context`.
fn candidates_for(e: &mut InputMethodEngine, context: &str, reading: &str) -> Vec<String> {
    e.reset();
    if !context.is_empty() {
        e.set_surrounding_context(context, "");
    }
    e.input_buf.insert(reading);
    e.start_conversion(false, None, 0);
    match e.state() {
        InputState::Conversion { candidates, .. } => {
            candidates.candidates().iter().map(|c| c.text.clone()).collect()
        }
        other => panic!("expected Conversion state, got {other:?}"),
    }
}

#[test]
#[ignore]
fn context_selects_the_right_homophone() {
    let mut e = real_engine();
    let mut failures = Vec::new();
    for (ctx, want) in [("車に", "乗ります"), ("雑誌に", "載ります")] {
        let cands = candidates_for(&mut e, ctx, "のります");
        println!("ctx={ctx:?}\n  candidates={cands:?}");
        if !cands.iter().any(|c| c == want) {
            failures.push(format!("{ctx}: {want} not offered at all"));
        } else if cands.first().map(String::as_str) != Some(want) {
            failures.push(format!("{ctx}: wanted {want} on top, got {:?}", cands.first()));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
#[ignore]
fn kana_fallbacks_stay_at_the_bottom() {
    // Up from the top wraps to katakana/hiragana, so they must remain last.
    let mut e = real_engine();
    for reading in ["のります", "きょうのてんき", "かんじ"] {
        let cands = candidates_for(&mut e, "", reading);
        println!("{reading}: {cands:?}");
        let n = cands.len();
        assert!(n >= 2, "{reading}: too few candidates {cands:?}");
        assert_eq!(
            cands[n - 2],
            reading,
            "{reading}: hiragana must be second to last, got {cands:?}"
        );
        assert_eq!(
            cands[n - 1],
            kashiko_engine::hiragana_to_katakana(reading),
            "{reading}: katakana must be last, got {cands:?}"
        );
    }
}

/// Live-conversion text for `reading` in `context`, with `learned` already in
/// the learning cache (as it would be after committing it once).
fn live_text_for(context: &str, reading: &str, learned: Option<(&str, &str)>) -> String {
    let mut e = real_engine();
    e.reset();
    if let Some((r, s)) = learned {
        let cache = e.learning.get_or_insert_with(|| kashiko_engine::LearningCache::new(100));
        cache.record(r, s);
    }
    e.set_surrounding_context(context, "");
    e.input_buf.insert(reading);
    e.chunked_auto_suggest().unwrap_or_else(|| reading.to_string())
}

#[test]
#[ignore]
fn live_conversion_follows_context_over_a_learned_form() {
    // A learned form used to short-circuit the model, so live conversion
    // ignored context entirely: after committing `車に乗ります` once, `のります`
    // stayed `乗ります` even under `雑誌に`.
    let learned = Some(("のります", "乗ります"));
    let with_car = live_text_for("車に", "のります", learned);
    let with_magazine = live_text_for("雑誌に", "のります", learned);
    println!("車に → {with_car:?}\n雑誌に → {with_magazine:?}");
    assert_eq!(with_car, "乗ります", "learning should hold where context agrees");
    assert_eq!(
        with_magazine, "載ります",
        "context should overrule the learned form"
    );
}

/// Candidates after the background deep search has run to completion, as the
/// user sees them once the loading indicator clears.
fn candidates_after_deep_search(e: &mut InputMethodEngine, context: &str, reading: &str) -> Vec<String> {
    e.reset();
    if !context.is_empty() {
        e.set_surrounding_context(context, "");
    }
    e.input_buf.insert(reading);
    // Live conversion first, as the real input path does: its text is pinned
    // to the top of the candidate list.
    if let Some(live) = e.chunked_auto_suggest() {
        e.live.text = live;
    }
    e.start_conversion(false, None, 0);
    let mut guard = 0;
    while e.deep_search_pending() && guard < 200 {
        e.step_deep_search();
        guard += 1;
    }
    match e.state() {
        InputState::Conversion { candidates, .. } => {
            candidates.candidates().iter().map(|c| c.text.clone()).collect()
        }
        other => panic!("expected Conversion state, got {other:?}"),
    }
}

#[test]
#[ignore]
fn every_offered_candidate_is_worth_offering() {
    // The reported list: nine candidates for `みなさんおはようございます`, of
    // which only #1, #2 and #9 were ever picked. Two leaks fed it — the deep
    // search merged its beams unscored, and the quick pass showed the tail its
    // scoring budget never reached.
    let mut e = real_engine();
    let cases: &[(&str, &[&str], &[&str])] = &[
        (
            "みなさんおはようございます",
            &["皆さんおはようございます", "皆さんおはよう御座います"],
            &["皆さんホア様ございます", "皆さんor兵庫ざいます", "皆さんおはお湯ございます", "皆さんお様ございます"],
        ),
        (
            "かけているぶんしょう",
            &["書けている文章", "欠けている文章"],
            &["かけている分廠", "かけている分省", "かけている分章"],
        ),
        // 80+ rare name readings used to reach the window here, because only
        // the first twelve fit the scoring budget and the rest went unjudged.
        ("かんじ", &["漢字", "感じ", "幹事"], &["侃二", "冠仁", "寛滋"]),
    ];
    let mut failures = Vec::new();
    for (reading, want, junk) in cases {
        let cands = candidates_after_deep_search(&mut e, "", reading);
        println!("{reading}\n  ({}) {cands:?}", cands.len());
        for w in *want {
            if !cands.iter().any(|c| c == w) {
                failures.push(format!("{reading}: lost {w}"));
            }
        }
        for j in *junk {
            if cands.iter().any(|c| c == j) {
                failures.push(format!("{reading}: still offers {j}"));
            }
        }
        // Two kana fallbacks plus a handful of real conversions.
        if cands.len() > 8 {
            failures.push(format!("{reading}: {} candidates is too many", cands.len()));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
#[ignore]
fn a_failed_conversion_still_gets_its_correction_search() {
    // The gate must not cost us the search's reason for existing: input with
    // raw keystrokes left in it always gets the correction pass, and its
    // findings are ranked against each other rather than against the literal
    // passthrough (which scores 0.18 there while every real correction sits
    // above 1.9 — an absolute bar would delete all of them).
    let mut e = real_engine();
    let reading = "さいびるどまえだったtまえだとおもわれます";
    let cands = candidates_after_deep_search(&mut e, "", reading);
    println!("({}) {cands:?}", cands.len());
    assert!(
        cands.iter().any(|c| !c.contains('t') && c.contains("ビルド")),
        "no ASCII-free correction offered, got {cands:?}"
    );
}

#[test]
#[ignore]
fn a_deleted_candidate_never_comes_back() {
    // The ✕ has to mean "stop showing me this" for good: the quick pass, the
    // background search and live conversion all have to honour it, and the
    // record has to be on disk before the next keystroke.
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let mut e = real_engine();
    e.learning_path = Some(tmp.path().to_path_buf());
    e.learning = Some(kashiko_engine::LearningCache::new(1000));

    let reading = "ごきぼうならしいんほかんをふくすうもじにひろげるかたちでたいおうできます";
    let before = candidates_after_deep_search(&mut e, "", reading);
    println!("before ({}) {before:?}", before.len());
    let doomed = before[0].clone();
    // Delete row 1, then convert the same reading again from scratch.
    e.forget_candidate_on_page(0);
    let after = candidates_after_deep_search(&mut e, "", reading);
    println!("deleted {doomed:?}\nafter  ({}) {after:?}", after.len());

    assert!(
        !after.iter().any(|c| *c == doomed),
        "{doomed} came back: {after:?}"
    );
    assert!(after.len() >= 3, "list should survive the deletion: {after:?}");
    // The rejected phrase was learned, so it is written through immediately
    // rather than waiting for the next focus change.
    let saved = std::fs::read_to_string(tmp.path()).unwrap();
    assert!(!saved.is_empty(), "deletion was not persisted");
}

#[test]
#[ignore]
fn long_sentence_candidates_are_all_plausible() {
    // A wrong character inside a long sentence barely moves the per-character
    // average, so `…形で対欧できます` used to sit right next to
    // `…形で対応できます`. Its worst character scores 13.8.
    let mut e = real_engine();
    let cands = candidates_after_deep_search(
        &mut e,
        "",
        "ごきぼうならしいんほかんをふくすうもじにひろげるかたちでたいおうできます",
    );
    println!("({}) {cands:#?}", cands.len());
    // `ご希望慣らしイン保管を…` is not caught and is expected to stay: it
    // averages 0.24 with a worst character of 4.2, so the model does not
    // strongly disbelieve any part of it — only a tighter average margin would
    // reach it, and that would start costing real candidates.
    for junk in ["対欧", "KIBOU"] {
        assert!(
            !cands.iter().any(|c| c.contains(junk)),
            "{junk} should have been gated out, got {cands:?}"
        );
    }
    assert!(
        cands.iter().any(|c| c.contains("子音補完")),
        "the intended reading must be offered, got {cands:?}"
    );
}
