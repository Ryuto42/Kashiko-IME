use super::*;

// --- Candidate preservation tests ---

#[test]
fn test_live_text_preserved_in_conversion_via_down() {
    // When DOWN is pressed during live conversion, the AI inference result
    // (live_conversion_text) should appear in the candidate list.
    let mut engine = make_live_conversion_engine();

    // Simulate typing "あい" with live conversion active
    engine.process_key(&press('a'));
    engine.process_key(&press('i'));
    engine.live.text = "愛".to_string();

    // Press DOWN → start_conversion()
    let result = engine.process_key(&press_key(Keysym::DOWN));
    assert!(result.consumed);
    assert!(matches!(engine.state(), InputState::Conversion { .. }));

    // The candidate list should contain "愛"
    let candidates = engine.state().candidates().unwrap();
    assert!(
        candidates.candidates().iter().any(|c| c.text == "愛"),
        "AI inference result '愛' should be in the candidate list"
    );
}

#[test]
fn test_live_text_not_duplicated_in_conversion() {
    // If the live_text matches the reading, it should not be duplicated
    let mut engine = make_live_conversion_engine();

    engine.process_key(&press('a'));
    engine.process_key(&press('i'));
    // live_conversion_text same as hiragana reading → should not be added
    engine.live.text = "あい".to_string();

    let result = engine.process_key(&press_key(Keysym::DOWN));
    assert!(result.consumed);
    assert!(matches!(engine.state(), InputState::Conversion { .. }));

    // "あい" should not appear twice (it's same as reading, so live_text is skipped)
    let candidates = engine.state().candidates().unwrap();
    let count = candidates
        .candidates()
        .iter()
        .filter(|c| c.text == "あい")
        .count();
    assert_eq!(count, 1, "Reading should appear exactly once");
}

#[test]
fn test_suggest_result_preserved_in_start_conversion() {
    // When Space is pressed, the previous auto-suggest/live conversion result
    // should be preserved in the candidate list even if re-inference doesn't produce it.
    // (Without a kanji converter, build_conversion_candidates returns fallback only,
    // so the live_conversion_text would be lost without the preservation logic.)
    let mut engine = InputMethodEngine::new();

    engine.process_key(&press('a'));
    engine.process_key(&press('i'));
    engine.live.text = "愛".to_string();

    // Press Space → start_conversion()
    let result = engine.process_key(&press_key(Keysym::SPACE));
    assert!(result.consumed);
    assert!(matches!(engine.state(), InputState::Conversion { .. }));

    // "愛" should be preserved in the candidate list
    let candidates = engine.state().candidates().unwrap();
    assert!(
        candidates.candidates().iter().any(|c| c.text == "愛"),
        "Previous suggest result '愛' should be preserved in candidates"
    );
}

#[test]
fn test_empty_live_text_not_added_to_candidates() {
    // When live_conversion_text is empty, no extra candidate should be added
    let mut engine = make_live_conversion_engine();

    engine.process_key(&press('a'));
    engine.process_key(&press('i'));
    // Force empty to test the "no live text" scenario
    engine.live.text.clear();

    // DOWN → start_conversion()
    let result = engine.process_key(&press_key(Keysym::DOWN));
    assert!(result.consumed);

    // Should have candidates but no empty-string candidate
    if let Some(candidates) = engine.state().candidates() {
        assert!(
            !candidates.candidates().iter().any(|c| c.text.is_empty()),
            "Empty candidate should not be in the list"
        );
    }
}

// --- likelihood reranking + quality gate ---

/// Build a candidate list from `(text, source)` pairs.
fn annotated(items: &[(&str, CandidateSource)]) -> Vec<AnnotatedCandidate> {
    items
        .iter()
        .map(|(t, s)| AnnotatedCandidate::new(*t, s.clone()))
        .collect()
}

/// Build gate input from `(index, score)` pairs; nothing is protected and no
/// worst-character measurement is supplied, so only the average-based rules
/// are exercised.
fn scored(pairs: &[(usize, Option<f32>)]) -> Vec<super::super::conversion::Scored> {
    pairs
        .iter()
        .map(|(index, score)| super::super::conversion::Scored {
            index: *index,
            score: *score,
            worst: None,
            never_drop: false,
        })
        .collect()
}

#[test]
fn a_single_disbelieved_character_drops_the_candidate() {
    // The average cannot see local damage in a long sentence: `…形で対欧できます`
    // averages 0.62 against 0.03 for `…形で対応できます`, inside any sensible
    // margin, while its worst character scores 13.8.
    use super::super::conversion::{gate_and_reorder, Scored};
    let cands = vec![
        AnnotatedCandidate::new("対応".to_string(), CandidateSource::Model),
        AnnotatedCandidate::new("対欧".to_string(), CandidateSource::Model),
        AnnotatedCandidate::new("退陰".to_string(), CandidateSource::Model),
    ];
    let s = |index, mean, worst| Scored {
        index,
        score: Some(mean),
        worst: Some(worst),
        never_drop: false,
    };
    // The third candidate keeps the list at `MIN_KEPT_SCORED`, so the second
    // being dropped is the worst-character rule and not the floor.
    let out = gate_and_reorder(&cands, &[s(0, 0.03, 0.4), s(2, 0.10, 1.2), s(1, 0.10, 13.8)]);
    assert_eq!(texts(&out), vec!["対応", "退陰"]);
}

fn texts(cands: &[AnnotatedCandidate]) -> Vec<&str> {
    cands.iter().map(|c| c.text.as_str()).collect()
}

#[test]
fn junk_candidates_are_dropped_not_just_ranked_lower() {
    // The reported problem: near homophones (`約割`, `役割り`, `パット`) padded
    // the list below the correct reading. They must be removed, so the number
    // of candidates shown follows quality instead of a fixed count.
    let mut cands = annotated(&[
        ("役割と構造要素", CandidateSource::Model),
        ("約割と構造要素", CandidateSource::Model),
        ("役割りと構造要素", CandidateSource::Model),
        ("やくわり", CandidateSource::Fallback),
    ]);
    let scored = scored(&[(0, Some(0.4)), (1, Some(1.9)), (2, Some(2.4))]);
    cands = super::super::conversion::gate_and_reorder(&cands, &scored);
    // MIN_KEPT_SCORED keeps the runner-up, but the worst one is gone.
    assert_eq!(texts(&cands), vec!["役割と構造要素", "約割と構造要素", "やくわり"]);
}

#[test]
fn a_uniformly_good_list_is_kept_in_full() {
    // The other half of the rule: good candidates are not truncated to a cap.
    let mut cands = annotated(&[
        ("天気", CandidateSource::Model),
        ("転機", CandidateSource::Model),
        ("典器", CandidateSource::Model),
        ("てんき", CandidateSource::Fallback),
    ]);
    let scored = scored(&[(0, Some(0.30)), (1, Some(0.40)), (2, Some(0.55))]);
    cands = super::super::conversion::gate_and_reorder(&cands, &scored);
    assert_eq!(texts(&cands), vec!["天気", "転機", "典器", "てんき"]);
}

#[test]
fn gate_keeps_surrounding_groups_in_place() {
    // Learning stays on top and the fallbacks stay at the bottom; only the
    // model block is reordered and filtered.
    let mut cands = annotated(&[
        ("学習", CandidateSource::Learning),
        ("転機", CandidateSource::Model),
        ("天気", CandidateSource::Model),
        ("てんき", CandidateSource::Fallback),
    ]);
    let scored = scored(&[(1, Some(2.0)), (2, Some(0.5))]);
    cands = super::super::conversion::gate_and_reorder(&cands, &scored);
    assert_eq!(texts(&cands), vec!["学習", "天気", "転機", "てんき"]);
}

#[test]
fn unscoreable_candidates_are_kept_after_scored_ones() {
    let mut cands = annotated(&[
        ("あ", CandidateSource::Model),
        ("い", CandidateSource::Model),
        ("う", CandidateSource::Model),
    ]);
    // `あ` can't be scored: keep it (no evidence against it), but after the
    // scored ones.
    let scored = scored(&[(0, None), (1, Some(0.5)), (2, Some(0.4))]);
    cands = super::super::conversion::gate_and_reorder(&cands, &scored);
    assert_eq!(texts(&cands), vec!["う", "い", "あ"]);
}

#[test]
fn all_bad_scores_still_leave_something_to_pick() {
    // Even when the model is lost, the list must not be emptied.
    let mut cands = annotated(&[
        ("甲", CandidateSource::Model),
        ("乙", CandidateSource::Model),
        ("丙", CandidateSource::Model),
    ]);
    let scored = scored(&[(0, Some(3.0)), (1, Some(3.5)), (2, Some(4.0))]);
    cands = super::super::conversion::gate_and_reorder(&cands, &scored);
    assert_eq!(cands.len(), super::super::conversion::MIN_KEPT_SCORED);
    assert_eq!(texts(&cands), vec!["甲", "乙"]);
}

#[test]
fn reranking_without_a_model_is_a_noop() {
    // Tests run with no kanji converter loaded; reranking must leave the list
    // untouched rather than dropping candidates.
    let mut engine = InputMethodEngine::new();
    engine.converters.kanji = None;
    let mut cands = annotated(&[
        ("転機", CandidateSource::Model),
        ("天気", CandidateSource::Model),
    ]);
    let before: Vec<String> = cands.iter().map(|c| c.text.clone()).collect();
    engine.rerank_by_likelihood("きょうのてんき", &mut cands);
    let after: Vec<String> = cands.iter().map(|c| c.text.clone()).collect();
    assert_eq!(before, after);
}

// --- deleting a candidate with the row's ✕ ---

#[test]
fn deletion_learns_the_phrase_that_was_rejected() {
    // Suppressing the whole surface only helps if the identical sentence comes
    // back, and a long one is never learned at all — so the ✕ has to isolate
    // the phrase being rejected and learn *that*.
    let spans = crate::core::engine::conversion::differing_spans(
        "ごきぼうならしいんほかんをふくすうもじにひろげる",
        "ご希望なら試飲保管を複数文字に広げる",
        "ご希望なら子音補完を複数文字に広げる",
    );
    assert_eq!(
        spans,
        vec![(
            "しいんほかんを".to_string(),
            "試飲保管を".to_string(),
            "子音補完を".to_string()
        )],
        "only the disagreeing phrase should be reported (the particle rides along)"
    );
}

#[test]
fn identical_conversions_teach_nothing() {
    let spans = crate::core::engine::conversion::differing_spans("あいうえ", "藍上", "藍上");
    assert!(spans.is_empty(), "got {spans:?}");
}

#[test]
fn unalignable_conversions_teach_nothing() {
    // A conversion that can't be aligned to the reading must not produce a
    // made-up pairing; nothing is learned rather than something wrong.
    let spans = crate::core::engine::conversion::differing_spans("あいうえ", "全然違う文", "藍上");
    assert!(spans.is_empty(), "got {spans:?}");
}

#[test]
fn compatibility_ideograph_spellings_are_one_candidate() {
    let mut b = super::super::conversion::CandidateBuilder::new();
    b.push(AnnotatedCandidate::new("有難う", CandidateSource::Dictionary));
    b.push(AnnotatedCandidate::new("有\u{FA68}う", CandidateSource::Dictionary));
    assert_eq!(b.into_candidates().len(), 1);
}
