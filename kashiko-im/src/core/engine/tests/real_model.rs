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
        // The budget still bounds this list, but nothing inside it is deleted:
        // see `short_readings_keep_their_homophones` for why a single word is
        // judged differently from a sentence.
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
        // Two kana fallbacks plus a handful of real conversions — but only
        // where the gate is allowed to delete, which is readings long enough
        // for the score to mean something.
        if reading.chars().count() > 4 && cands.len() > 8 {
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
    // rather than waiting for the next focus change — and written encrypted,
    // so the surface is not readable on disk.
    let raw = std::fs::read(tmp.path()).unwrap();
    assert!(!raw.is_empty(), "deletion was not persisted");
    assert!(
        !String::from_utf8_lossy(&raw).contains(&doomed),
        "the deleted surface is readable in the cache file"
    );
    let saved = kashiko_engine::learning_file::read_maybe_encrypted(tmp.path()).unwrap();
    assert!(saved.contains(&doomed), "the suppression was not recorded");
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

#[test]
#[ignore]
fn loanwords_offer_their_english_spelling() {
    // Borrowed words have no Japanese spelling to convert to, so the dictionary
    // and the model can only ever return katakana — the English form is
    // unreachable without a table.
    let mut e = real_engine();
    let mut failures = Vec::new();
    for (reading, want) in [
        ("てんぷれーと", "template"),
        ("わーくふろー", "workflow"),
        ("りぽじとり", "repository"),
        ("ぷるりくえすと", "pull request"),
        ("でぷろい", "deploy"),
    ] {
        let cands = candidates_after_deep_search(&mut e, "", reading);
        println!("{reading} → {cands:?}");
        let kana_at = cands.iter().position(|c| kashiko_engine::is_all_kana(c));
        match cands.iter().position(|c| c == want) {
            None => failures.push(format!("{reading}: {want} not offered, got {cands:?}")),
            // The katakana form leads and English follows it, so the English
            // spelling is on the first page and above every kana row but that
            // one.
            Some(at) if at > 1 => {
                failures.push(format!("{reading}: {want} sits at row {}, got {cands:?}", at + 1))
            }
            Some(at) => {
                if kana_at.is_some_and(|k| k > at && k > 1) {
                    failures.push(format!("{reading}: a kana row precedes {want}, got {cands:?}"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
#[ignore]
fn counted_quantities_are_offered() {
    // `にげん` had no `2限`: the dictionary is keyed on whole readings and has
    // no entry, and the model's beam finds the digit form only sometimes (`2階`
    // for `にかい`, nothing for `にげん`). Measured on the real model `2限`
    // scores 1.03 where the best candidate offered scored 0.42 — the right
    // answer was simply never on the list.
    let mut e = real_engine();
    let mut failures = Vec::new();
    for (reading, want) in [
        ("にげん", "2限"),
        ("にかい", "2回"),
        ("さんじ", "3時"),
        ("ごねん", "5年"),
        ("じゅうにじ", "12時"),
    ] {
        let cands = candidates_after_deep_search(&mut e, "", reading);
        println!("{reading} ({}) {cands:?}", cands.len());
        if !cands.iter().any(|c| c == want) {
            failures.push(format!("{reading}: {want} not offered, got {cands:?}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
#[ignore]
fn short_readings_keep_their_homophones() {
    // A single word gives the model no sentence to judge a candidate against,
    // so a bad score there means "unusual", not "wrong". `ゆうと` used to offer
    // `言うと` and nothing else, with all 96 dictionary readings and the beam's
    // own 優斗 deleted for scoring badly — on a reading that is nearly always a
    // name.
    let mut e = real_engine();
    let mut failures = Vec::new();
    for (reading, want) in [
        // `優斗` and `寛治` come from the *background* pass now: the
        // synchronous beam is narrow (see `sync_candidate_count`) and the width
        // moved to the worker, so these arrive a moment after Space rather than
        // during it.
        ("ゆうと", &["優斗", "侑斗", "佑都"][..]),
        ("たなか", &["田中", "田仲", "多仲"][..]),
        ("こばやし", &["小林", "子林", "児林"][..]),
        ("かんじ", &["漢字", "感じ", "幹事", "監事", "寛治"][..]),
    ] {
        let cands = candidates_after_deep_search(&mut e, "", reading);
        println!("{reading} ({}) {cands:?}", cands.len());
        for w in want {
            if !cands.iter().any(|c| c == w) {
                failures.push(format!("{reading}: lost {w}, got {cands:?}"));
            }
        }
        // Half-converted kana (`にゲン`, `コバやし`, `ゴネん`) is the model
        // giving up part way and takes slots from the homophones.
        for c in &cands {
            let artifact = kashiko_engine::is_all_kana(c)
                && !kashiko_engine::is_pure_hiragana(c)
                && !kashiko_engine::is_pure_full_katakana(c);
            if artifact {
                failures.push(format!("{reading}: half-converted {c}, got {cands:?}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
#[ignore]
fn bracketed_and_decorated_entries_are_not_offered() {
    // The dictionary carries nicknames and glosses (`田中☆`, `任意(コピー)`)
    // that are not conversions of the reading.
    let mut e = real_engine();
    for reading in ["たなか", "にんい", "こばやし"] {
        let cands = candidates_after_deep_search(&mut e, "", reading);
        for c in &cands {
            assert!(
                !c.contains('(') && !c.contains('（') && !c.contains('☆') && !c.contains('★'),
                "{reading}: {c} should not be offered, got {cands:?}"
            );
        }
    }
}

#[test]
#[ignore]
fn the_inference_button_finds_more_each_press() {
    // The aux line offers another pass, and each press has to search wider —
    // pressing it twice must not re-run the identical search.
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let mut e = real_engine();
    e.learning_path = Some(tmp.path().to_path_buf());
    e.learning = Some(kashiko_engine::LearningCache::new(1000));

    let reading = "ごきぼうならしいんほかんをふくすうもじにひろげるかたちでたいおうできます";
    let before = candidates_after_deep_search(&mut e, "", reading);
    println!("auto   ({}) {before:?}", before.len());

    let mut seen = before.len();
    let mut grew = false;
    for round in 1..=2 {
        let r = e.deepen_inference();
        assert!(
            r.consumed,
            "round {round}: the button did nothing"
        );
        let mut guard = 0;
        while e.deep_search_pending() && guard < 400 {
            e.step_deep_search();
            guard += 1;
        }
        let after = match e.state() {
            InputState::Conversion { candidates, .. } => {
                candidates.candidates().iter().map(|c| c.text.clone()).collect::<Vec<_>>()
            }
            other => panic!("expected Conversion state, got {other:?}"),
        };
        println!("round {round} ({}) {after:?}", after.len());
        assert!(
            after.len() >= seen,
            "round {round} lost candidates: {after:?}"
        );
        grew |= after.len() > seen;
        seen = after.len();
    }
    // Later rounds may legitimately find nothing — a short reading's beam runs
    // out — but the first press has to earn itself on a reading this long.
    assert!(grew, "the button never added anything");
}

#[test]
#[ignore]
fn a_sentence_with_a_number_converts() {
    // The reported failure: `こうほがよんけんみまんのときはおさなくても` came
    // back as `ウホが4件未満の時は幼くても`, with `候補` missing two characters.
    //
    // One deletion caused all of it. `候補が4件未満の時は幼くても` (NLL 0.05,
    // and what live conversion was already showing) was thrown out for spelling
    // a `4` the kana reading does not contain — leaving only the kana
    // fallbacks. A fallback is by definition a long unconverted run, so the
    // typo-correction path fired, and its single-character deletions of the
    // reading (`こほが…`, `うほが…`) became the only candidates left.
    let mut e = real_engine();
    let mut failures = Vec::new();
    for (reading, want) in [
        (
            "こうほがよんけんみまんのときはおさなくても",
            &["候補が4件未満の時は幼くても", "候補が4件未満の時は押さなくても"][..],
        ),
        ("よんけんみまん", &["4件未満"][..]),
    ] {
        let cands = candidates_after_deep_search(&mut e, "", reading);
        println!("{reading} ({}) {cands:?}", cands.len());
        for w in want {
            if !cands.iter().any(|c| c == w) {
                failures.push(format!("{reading}: {w} missing, got {cands:?}"));
            }
        }
        // `ウホ`/`コホ` are the correction search deleting a character from the
        // reading; nothing that mangles the opening phrase should be offered.
        for c in &cands {
            if c.starts_with("ウホ") || c.starts_with("コホ") {
                failures.push(format!("{reading}: still offers {c}"));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
#[ignore]
fn selection_stays_responsive_while_inference_runs() {
    // The deep search runs on a worker thread, so scheduling one must not stop
    // the engine from answering anything. Measured through the real server, a
    // key event during inference now returns in ~3ms; before this it waited for
    // whichever model call was in flight, which for a wide beam is seconds.
    let mut e = real_engine();
    let reading = "ごきぼうならしいんほかんをふくすうもじにひろげるかたちでたいおうできます";
    let _ = candidates_for(&mut e, "", reading);

    let job = e.take_deep_job().expect("a search should have been scheduled");
    // With the job out of the engine's hands, every key is served immediately.
    let t = std::time::Instant::now();
    for _ in 0..20 {
        e.process_key(&press_key(Keysym::DOWN));
    }
    let per_key = t.elapsed() / 20;
    println!("per key while a search is out: {per_key:?}");
    assert!(
        per_key < std::time::Duration::from_millis(20),
        "keys are still waiting on inference: {per_key:?} each"
    );

    // And the findings still land when the work comes back.
    let outcome = crate::run_deep_job(job);
    let applied = e.apply_deep_outcome(outcome);
    println!("applied: {}", applied.is_some());
}

#[test]
#[ignore]
fn a_compound_offers_each_words_homophones() {
    // `しこうちゅう` isn't in the dictionary as a whole, so `施工中の` was
    // treated as one opaque word: no 思考中の, and Right had nothing to split.
    let mut e = real_engine();
    let cands = candidates_for(&mut e, "", "しこうちゅうの");
    println!("candidates={cands:?}");
    assert!(cands.iter().any(|c| c == "思考中の"), "{cands:?}");

    e.process_key(&press_key(Keysym::RIGHT));
    let focused = match e.state() {
        InputState::Conversion { candidates, .. } => candidates
            .candidates()
            .iter()
            .map(|c| c.text.clone())
            .collect::<Vec<_>>(),
        other => panic!("expected Conversion state, got {other:?}"),
    };
    println!("focused={focused:?}");
    assert!(focused.iter().any(|c| c == "思考"), "{focused:?}");
}
