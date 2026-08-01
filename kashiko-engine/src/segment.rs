//! Recovering phrase (文節) boundaries from a conversion result.
//!
//! Segmenting the *reading* alone is guesswork: greedy longest-match over a
//! dictionary reads `きょうのてんきははれです` as `…てんき|はは|れです`, because
//! `はは` (母) is a real word — even though the two `は` belong to different
//! words (the particle, and the first kana of `晴れ`). The conversion the model
//! already produced resolves that ambiguity, since it shows which stretches of
//! reading became kanji:
//!
//! ```text
//! reading    きょう の てんき は は  れです
//! converted  今日   の 天気   は 晴  れです
//! ```
//!
//! Kana in the converted text appear verbatim in the reading, so they act as
//! anchors; whatever sits between two anchors is the reading of the kanji run
//! between them. Aligning on those anchors yields the true boundaries, which is
//! what this module does.

/// Characters that survive conversion unchanged, so they can be matched
/// against the reading: kana and the prolonged sound mark.
fn is_anchor(c: char) -> bool {
    matches!(c, '\u{3041}'..='\u{3096}' | '\u{30A1}'..='\u{30FA}' | '\u{30FC}')
}

/// Phrase boundaries (char offsets into `reading`, ascending, last == length)
/// derived by aligning `converted` against `reading`.
///
/// Trailing kana are merged into the preceding kanji run, so a boundary falls
/// after each "content word + its okurigana/particles" group — the unit a user
/// expects Left/Right to step over (`今日の|天気は|晴れです`).
///
/// Returns `None` when the two can't be aligned (the conversion dropped or
/// reordered kana), leaving the caller to fall back to dictionary segmentation.
pub fn boundaries_from_conversion(reading: &str, converted: &str) -> Option<Vec<usize>> {
    let r: Vec<char> = reading.chars().collect();
    let c: Vec<char> = converted.chars().collect();
    if r.is_empty() {
        return None;
    }

    let mut bounds: Vec<usize> = Vec::new();
    let mut ri = 0usize; // consumed reading chars
    let mut ci = 0usize;

    while ci < c.len() {
        // Opaque run: kanji (or anything else conversion introduced), whose
        // reading length is unknown but is at least one kana per character.
        let opaque_start = ci;
        while ci < c.len() && !is_anchor(c[ci]) {
            ci += 1;
        }
        let opaque_len = ci - opaque_start;

        if ci >= c.len() {
            // Trailing opaque run: it owns the rest of the reading.
            if ri < r.len() {
                bounds.push(r.len());
                ri = r.len();
            }
            break;
        }

        // Anchor run: matches the reading verbatim.
        let anchor_start = ci;
        while ci < c.len() && is_anchor(c[ci]) {
            ci += 1;
        }
        let anchor: String = c[anchor_start..ci].iter().collect();

        // Start searching past the minimum the opaque run must consume,
        // otherwise an anchor that also appears inside the kanji's reading
        // matches too early (`花は` / `はなは` would give `花` zero kana).
        let search_from = (ri + opaque_len).min(r.len());
        let rest: String = r[search_from..].iter().collect();
        let byte_pos = rest.find(&anchor)?;
        let gap = rest[..byte_pos].chars().count();
        let anchor_end = search_from + gap + anchor.chars().count();
        if anchor_end > r.len() {
            return None;
        }
        bounds.push(anchor_end);
        ri = anchor_end;
    }

    if ri < r.len() {
        bounds.push(r.len());
    }
    bounds.dedup();
    // Must be strictly increasing and cover the whole reading to be usable.
    if bounds.is_empty() || bounds.last() != Some(&r.len()) {
        return None;
    }
    if bounds.windows(2).any(|w| w[0] >= w[1]) {
        return None;
    }
    Some(bounds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_the_reported_sentence_correctly() {
        // The bug: dictionary segmentation read `はは` (母) across the particle
        // `は` and the `は` of `晴れ`, so the focus landed on a nonsense span.
        let bounds =
            boundaries_from_conversion("きょうのてんきははれです", "今日の天気は晴れです")
                .expect("should align");
        // 今日の | 天気は | 晴れです
        assert_eq!(bounds, vec![4, 8, 12]);
    }

    #[test]
    fn kanji_reading_containing_the_anchor_kana() {
        // `花は` / `はなは`: the anchor `は` also starts the kanji's reading, so
        // a naive search would give `花` zero kana.
        let bounds = boundaries_from_conversion("はなは", "花は").expect("should align");
        assert_eq!(bounds, vec![3]);
    }

    #[test]
    fn all_kana_conversion_is_one_segment() {
        let bounds = boundaries_from_conversion("あいうえ", "あいうえ").expect("should align");
        assert_eq!(bounds, vec![4]);
    }

    #[test]
    fn all_kanji_conversion_is_one_segment() {
        let bounds = boundaries_from_conversion("わたし", "私").expect("should align");
        assert_eq!(bounds, vec![3]);
    }

    #[test]
    fn multiple_kanji_groups() {
        // 私は | 学校に | 行きます
        let bounds =
            boundaries_from_conversion("わたしはがっこうにいきます", "私は学校に行きます")
                .expect("should align");
        assert_eq!(bounds, vec![4, 9, 13]);
    }

    #[test]
    fn unalignable_conversion_returns_none() {
        // Conversion dropped kana that the reading has: can't be trusted.
        assert_eq!(boundaries_from_conversion("あいうえお", "漢ん字"), None);
    }

    #[test]
    fn empty_reading_returns_none() {
        assert_eq!(boundaries_from_conversion("", "何か"), None);
    }
}
