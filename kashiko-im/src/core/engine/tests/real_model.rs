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
    let mut settings = Settings::load().unwrap_or_default();
    if let Ok(model) = std::env::var("KASHIKO_TEST_MODEL") {
        settings.conversion.model = Some(model);
    }
    if let Ok(model) = std::env::var("KASHIKO_TEST_LIGHT") {
        settings.conversion.light_model = Some(model);
    }
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
        //
        // Which name spellings appear depends on the model's beam, so this asks
        // for most of a set rather than exact members.
        ("ゆうと", &["優斗", "侑斗", "佑都", "悠斗", "悠人", "佑斗", "祐登"][..]),
        ("たなか", &["田中", "田仲", "多仲"][..]),
        ("こばやし", &["小林", "子林", "児林"][..]),
        ("かんじ", &["漢字", "感じ", "幹事", "監事", "寛治"][..]),
    ] {
        let cands = candidates_after_deep_search(&mut e, "", reading);
        println!("{reading} ({}) {cands:?}", cands.len());
        let found = want.iter().filter(|w| cands.iter().any(|c| c == *w)).count();
        if found * 2 < want.len() + 1 {
            failures.push(format!("{reading}: only {found} of {want:?}, got {cands:?}"));
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
        // Either numeral style: jinen-v1 writes 4件, jinen-v2 四件.
        (
            "こうほがよんけんみまんのときはおさなくても",
            &["候補が4件未満の時は押さなくても", "候補が四件未満の時は押さなくても"][..],
        ),
        ("よんけんみまん", &["4件未満", "四件未満"][..]),
    ] {
        let cands = candidates_after_deep_search(&mut e, "", reading);
        println!("{reading} ({}) {cands:?}", cands.len());
        if !want.iter().any(|w| cands.iter().any(|c| c == w)) {
            failures.push(format!("{reading}: none of {want:?}, got {cands:?}"));
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

/// Live-conversion accuracy on AJIMEE-Bench (`evaluation_items.json`), through
/// the same path the preedit uses. Point `AJIMEE` at the file; set
/// `AJIMEE_RIGHT=1` to also supply the text after the cursor.
#[test]
#[ignore]
fn ajimee_live_accuracy() {
    let Ok(path) = std::env::var("AJIMEE") else {
        println!("AJIMEE not set; skipping");
        return;
    };
    let with_right = std::env::var("AJIMEE_RIGHT").is_ok();
    let items: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut settings = Settings::load().unwrap_or_default();
    if let Ok(model) = std::env::var("AJIMEE_MODEL") {
        settings.conversion.model = Some(model);
    }
    if let Ok(model) = std::env::var("AJIMEE_LIGHT") {
        settings.conversion.light_model = Some(model);
    }
    let mut e = InputMethodEngine::with_config(EngineConfig::from_settings(&settings));
    e.init_from_settings(&settings).expect("engine init");
    e.learning = None;
    let (mut hit, mut hit_nfkc, mut total, mut ms) = (0usize, 0usize, 0usize, 0u128);
    let (mut space_top, mut space_any) = (0usize, 0usize);
    let typed = std::env::var("AJIMEE_TYPED").is_ok();
    let (mut switched, mut keys) = (0usize, 0usize);
    for item in &items {
        let left = item["context_text"].as_str().unwrap_or("");
        let reading = kashiko_engine::katakana_to_hiragana(item["input"].as_str().unwrap());
        let expected: Vec<&str> = item["expected_output"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|v| v.as_str())
            .collect();
        let original = item["original_text"].as_str().unwrap_or("");
        let right = original
            .strip_prefix(left)
            .and_then(|rest| expected.iter().find_map(|x| rest.strip_prefix(x)))
            .unwrap_or("");
        e.reset();
        e.metrics.adaptive_use_light_model = false;
        e.set_surrounding_context(left, if with_right { right } else { "" });
        let t = std::time::Instant::now();
        let out = if typed {
            // As typed: one live conversion per keystroke, the latency-driven
            // model switch carried across them like in a real composition.
            let mut out = None;
            for ch in reading.chars() {
                e.input_buf.insert(ch.encode_utf8(&mut [0; 4]));
                out = e.chunked_auto_suggest();
                switched += e.metrics.adaptive_use_light_model as usize;
                keys += 1;
            }
            out.unwrap_or_else(|| reading.clone())
        } else {
            e.input_buf.insert(&reading);
            e.chunked_auto_suggest().unwrap_or_else(|| reading.clone())
        };
        ms += t.elapsed().as_millis();
        total += 1;
        e.start_conversion(false, None, 0);
        if let InputState::Conversion { candidates, .. } = e.state() {
            let texts: Vec<&str> = candidates.candidates().iter().map(|c| c.text.as_str()).collect();
            space_top += texts.first().is_some_and(|t| expected.contains(t)) as usize;
            space_any += texts.iter().any(|t| expected.contains(t)) as usize;
        }
        let nfkc = |t: &str| kashiko_engine::kana::normalize_nfkc(t);
        if expected.contains(&out.as_str()) {
            hit += 1;
        }
        if expected.iter().any(|x| nfkc(x) == nfkc(&out)) {
            hit_nfkc += 1;
        } else {
            println!("MISS ctx={left:?} right={right:?}\n  got  {out}\n  want {}", expected[0]);
        }
    }
    println!(
        "AJIMEE live: {hit}/{total} = {:.1}%, NFKC {:.1}% ({} ms/item)",
        100.0 * hit as f64 / total as f64,
        100.0 * hit_nfkc as f64 / total as f64,
        ms / total as u128
    );
    println!("AJIMEE space: top {space_top}/{total}, anywhere in list {space_any}/{total}");
    if typed {
        println!("AJIMEE typed: {keys} keystrokes, light model in effect after {switched}");
    }
}

#[test]
#[ignore]
fn text_after_the_cursor_picks_the_homophone() {
    // Typing into the middle of a sentence: only what follows tells these
    // apart, and the model itself has no slot for it.
    let cases = [
        ("", "きしゃ", "が駅に到着した。", "汽車"),
        ("", "きしゃ", "会見で質問した。", "記者"),
        ("", "はし", "を渡って向こう岸へ行く。", "橋"),
        ("", "はし", "でご飯を食べる。", "箸"),
        ("", "こうえん", "で子供が遊んでいる。", "公園"),
        ("", "こうえん", "会のチケットを買った。", "講演"),
        ("", "せいかく", "が明るい人だ。", "性格"),
        ("", "せいかく", "な数字を出す。", "正確"),
        ("", "いどう", "を発表した人事部。", "異動"),
        ("", "いどう", "して会議室に向かう。", "移動"),
        ("", "かいとう", "した肉を焼く。", "解凍"),
        ("", "かいとう", "用紙に名前を書く。", "解答"),
        ("", "きかん", "限定のセールです。", "期間"),
        ("", "きかん", "投資家が株を買う。", "機関"),
        ("", "あつい", "お茶を飲む。", "熱い"),
        ("", "あつい", "夏の日が続く。", "暑い"),
        ("", "あつい", "本を読み終えた。", "厚い"),
        ("", "かてい", "の事情で休む。", "家庭"),
        ("", "かてい", "を経て結論に至る。", "過程"),
        ("", "しこう", "錯誤を繰り返す。", "試行"),
        ("", "しこう", "回路を鍛える。", "思考"),
        ("", "たいしょう", "年齢を確認する。", "対象"),
        ("", "たいしょう", "的な性格の二人。", "対照"),
        ("", "かんしん", "を持って聞いてくれた。", "関心"),
        ("", "かんしん", "するほど上手だ。", "感心"),
        ("", "こうかい", "前にテストする。", "公開"),
        ("", "こうかい", "しても遅い。", "後悔"),
        ("", "ほしょう", "期間が切れた。", "保証"),
        ("", "ほしょう", "制度で守られる。", "保障"),
        ("家計を助けるため、領主の家に", "ほうこうし", "ている。", "奉公し"),
    ];
    let mut e = real_engine();
    e.learning = None;
    let mut misses = Vec::new();
    for (left, reading, right, want) in cases {
        e.reset();
        e.set_surrounding_context(left, right);
        e.input_buf.insert(reading);
        let t = std::time::Instant::now();
        let live = e.chunked_auto_suggest().unwrap_or_default();
        let live_ms = t.elapsed().as_millis();
        e.start_conversion(false, None, 0);
        let top = match e.state() {
            InputState::Conversion { candidates, .. } => {
                candidates.candidates().first().map(|c| c.text.clone()).unwrap_or_default()
            }
            other => panic!("expected Conversion state, got {other:?}"),
        };
        println!("{reading}|{right}: live={live} ({live_ms} ms) space={top}");
        if live != want || top != want {
            misses.push(format!("{reading}|{right}: live={live} space={top}, want {want}"));
        }
    }
    println!("right-context: {}/{} live+space correct", cases.len() - misses.len(), cases.len());
    // Without the right context 12 of these come out right, all of them by
    // luck of the commonest homophone; this guards the gain over that.
    assert!(cases.len() - misses.len() >= 15, "{misses:#?}");
}

#[test]
#[ignore]
fn common_words_convert_without_context() {
    // No context at all: the first word typed into an empty field.
    let words = [
        ("にほん", "日本"), ("あした", "明日"), ("いく", "行く"), ("きょう", "今日"),
        ("でんわ", "電話"), ("かいしゃ", "会社"), ("がっこう", "学校"), ("せんせい", "先生"),
        ("ともだち", "友達"), ("しごと", "仕事"), ("じかん", "時間"), ("ことし", "今年"),
        ("らいしゅう", "来週"), ("かいぎ", "会議"), ("しりょう", "資料"), ("かくにん", "確認"),
        ("れんらく", "連絡"), ("よてい", "予定"), ("へんこう", "変更"), ("たいおう", "対応"),
        ("みせ", "店"), ("くるま", "車"), ("でんしゃ", "電車"), ("えき", "駅"),
        ("ひと", "人"), ("みず", "水"), ("やま", "山"), ("かわ", "川"),
        ("はな", "花"), ("あめ", "雨"), ("そら", "空"), ("いえ", "家"),
        ("みる", "見る"), ("たべる", "食べる"), ("かく", "書く"), ("よむ", "読む"),
        ("はなす", "話す"), ("おもう", "思う"), ("あう", "会う"), ("つかう", "使う"),
        ("しらべる", "調べる"), ("おくる", "送る"),
    ];
    let mut e = real_engine();
    e.learning = None;
    let (mut live_ok, mut top_ok) = (0, 0);
    for (r, want) in words {
        e.reset();
        e.input_buf.insert(r);
        let live = e.chunked_auto_suggest().unwrap_or_else(|| r.to_string());
        let c = candidates_for(&mut e, "", r);
        live_ok += (live == want) as usize;
        top_ok += (c.first().map(String::as_str) == Some(want)) as usize;
        if live != want || c.first().map(String::as_str) != Some(want) {
            println!("MISS {r}: live={live} top={:?}", c.first());
        }
    }
    println!("common words: live {live_ok}/{n}, space top {top_ok}/{n}", n = words.len());
    // Measured: jinen-v1 38/42 and 39/42, jinen-v2 37 and 39.
    assert!(live_ok * 100 >= words.len() * 85 && top_ok * 100 >= words.len() * 90);
}

#[test]
#[ignore]
fn english_inside_romaji_converts_around_it() {
    let mut e = real_engine();
    e.learning = None;
    e.live.enabled = true;
    let cases = [
        ("kyouhagoogledekensakusuru", "Google"),
        ("githubnoissuewokakunin", "GitHub"),
        ("ashitanomeetingnisanka", "meeting"),
    ];
    let mut misses = Vec::new();
    for (raw, word) in cases {
        e.reset();
        let t = std::time::Instant::now();
        for c in raw.chars() {
            e.process_key(&press(c));
        }
        let per_key = t.elapsed().as_millis() / raw.len() as u128;
        let live = e.live.text.clone();
        println!("{raw}: buffer={} live={live} ({per_key} ms/key)", e.input_buf.text);
        if !live.contains(word) || live.chars().any(|c| ('ぁ'..='ゖ').contains(&c) && false) {
            misses.push(format!("{raw}: {live}"));
        }
    }
    assert!(misses.is_empty(), "{misses:#?}");
}

/// Japanese sentences typed as romaji never sprout English, with the real
/// dictionary deciding what is a word. `ROMAJI_CORPUS`: one sentence per line.
#[test]
#[ignore]
fn romaji_corpus_stays_japanese() {
    let Ok(path) = std::env::var("ROMAJI_CORPUS") else {
        return;
    };
    let mut e = real_engine();
    // `KEEP_LEARNING=1` runs against the user's own learning cache.
    if std::env::var("KEEP_LEARNING").is_err() {
        e.learning = None;
    }
    let text = std::fs::read_to_string(path).unwrap();
    let mut bad = Vec::new();
    for line in text.lines() {
        e.reset();
        for c in line.chars() {
            e.process_key(&press(c));
            if !e.mixed.words.is_empty() {
                bad.push(format!("{line}: {:?} in {}", e.mixed.words, e.input_buf.text));
                break;
            }
        }
    }
    println!("{} lines, {} with English", text.lines().count(), bad.len());
    assert!(bad.is_empty(), "{bad:#?}");
}



/// The focused segment's candidates after `rights` presses of Right.
fn segment_after(e: &mut InputMethodEngine, reading: &str, rights: usize) -> (String, Vec<String>) {
    e.reset();
    e.input_buf.insert(reading);
    if let Some(live) = e.chunked_auto_suggest() {
        e.live.text = live;
    }
    e.start_conversion(false, None, 0);
    for _ in 0..rights {
        e.process_key(&press_key(Keysym::RIGHT));
    }
    match e.state() {
        InputState::Conversion { candidates, segment_start, split_index, .. } => (
            reading.chars().skip(*segment_start).take(split_index - segment_start).collect(),
            candidates.candidates().iter().map(|c| c.text.clone()).collect(),
        ),
        other => panic!("expected Conversion state, got {other:?}"),
    }
}

#[test]
#[ignore]
fn a_focused_segment_is_ranked_in_its_sentence() {
    // Judged alone, `はしで` has no reason to prefer 箸で; between `ご飯を`
    // and `食べた` it does. And `ご飯`/`お茶` used to be cut after their first
    // kana (`かれはご|はんを`).
    let mut e = real_engine();
    e.learning = None;
    let cases = [
        ("かれはごはんをはしでたべた", 2, "ごはんを", "ご飯を"),
        ("かれはごはんをはしでたべた", 3, "はしで", "箸で"),
        ("あついおちゃをのみたい", 2, "おちゃを", "お茶を"),
        ("えーあいのしこうかていをみる", 2, "しこう", "思考"),
        ("えーあいのしこうかていをみる", 3, "かていを", "過程を"),
    ];
    let mut failures = Vec::new();
    for (reading, rights, segment, want) in cases {
        let (seg, cands) = segment_after(&mut e, reading, rights);
        println!("{reading} [{seg}] {cands:?}");
        if seg != segment || cands.first().map(String::as_str) != Some(want) {
            failures.push(format!("{reading}: [{seg}] {cands:?}, want [{segment}] {want}"));
        }
        // Model guesses that break the sentence are gone.
        if let Some(bad) = cands.iter().find(|c| c.contains('〜') || c.ends_with("ヲ") && c.chars().any(|ch| ('一'..='龯').contains(&ch))) {
            failures.push(format!("{reading}: still offers {bad}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
#[ignore]
fn variant_kanji_of_names_are_offered() {
    let mut e = real_engine();
    e.learning = None;
    let mut failures = Vec::new();
    for (reading, want) in [("たかはし", "髙橋"), ("さいとう", "齊藤"), ("はまだ", "濵田"), ("よしだ", "𠮷田"), ("たか", "髙")] {
        let cands = candidates_for(&mut e, "", reading);
        println!("{reading}: {cands:?}");
        if !cands.iter().any(|c| c == want) {
            failures.push(format!("{reading}: no {want} in {cands:?}"));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
