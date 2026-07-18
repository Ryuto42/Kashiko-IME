//! Keyboard-aware typo variant generation.
//!
//! Typing mistakes on a QWERTY keyboard happen in *romaji* space, not kana
//! space: a finger slip swaps or substitutes a Latin letter *before* it ever
//! becomes kana. Editing the kana directly therefore misses most real typos —
//! e.g. `t前` came from the keystrokes `t m a e`, and the intended `ため` is
//! one adjacent *romaji* transposition away (`tmae` → `tame`), which no kana
//! edit can reach. This module converts a suspicious kana region back to
//! romaji, applies keyboard-plausible edits (transpositions, adjacent-key
//! substitutions, deletions, vowel insertions after stranded consonants), and
//! converts the results back to kana for the conversion model to try.

use std::collections::HashSet;

use crate::romaji::RomajiConverter;

/// Physical QWERTY neighbors for each letter key, used to model
/// "pressed the key next to the one I meant" substitutions.
fn qwerty_neighbors(c: char) -> &'static str {
    match c {
        'q' => "wa",
        'w' => "qesa",
        'e' => "wrds",
        'r' => "etfd",
        't' => "rygf",
        'y' => "tuhg",
        'u' => "yijh",
        'i' => "uokj",
        'o' => "iplk",
        'p' => "ol",
        'a' => "qwsz",
        's' => "awedxz",
        'd' => "serfcx",
        'f' => "drtgvc",
        'g' => "ftyhbv",
        'h' => "gyujnb",
        'j' => "huikmn",
        'k' => "jiolm",
        'l' => "kop",
        'z' => "asx",
        'x' => "zsdc",
        'c' => "xdfv",
        'v' => "cfgb",
        'b' => "vghn",
        'n' => "bhjm",
        'm' => "njk",
        _ => "",
    }
}

/// Romaji spelling for a single base hiragana, chosen so the round trip
/// through the romaji rules reproduces the original kana.
fn kana_to_romaji_char(c: char) -> Option<&'static str> {
    Some(match c {
        'あ' => "a", 'い' => "i", 'う' => "u", 'え' => "e", 'お' => "o",
        'か' => "ka", 'き' => "ki", 'く' => "ku", 'け' => "ke", 'こ' => "ko",
        'が' => "ga", 'ぎ' => "gi", 'ぐ' => "gu", 'げ' => "ge", 'ご' => "go",
        'さ' => "sa", 'し' => "si", 'す' => "su", 'せ' => "se", 'そ' => "so",
        'ざ' => "za", 'じ' => "zi", 'ず' => "zu", 'ぜ' => "ze", 'ぞ' => "zo",
        'た' => "ta", 'ち' => "ti", 'つ' => "tu", 'て' => "te", 'と' => "to",
        'だ' => "da", 'ぢ' => "di", 'づ' => "du", 'で' => "de", 'ど' => "do",
        'な' => "na", 'に' => "ni", 'ぬ' => "nu", 'ね' => "ne", 'の' => "no",
        'は' => "ha", 'ひ' => "hi", 'ふ' => "fu", 'へ' => "he", 'ほ' => "ho",
        'ば' => "ba", 'び' => "bi", 'ぶ' => "bu", 'べ' => "be", 'ぼ' => "bo",
        'ぱ' => "pa", 'ぴ' => "pi", 'ぷ' => "pu", 'ぺ' => "pe", 'ぽ' => "po",
        'ま' => "ma", 'み' => "mi", 'む' => "mu", 'め' => "me", 'も' => "mo",
        'や' => "ya", 'ゆ' => "yu", 'よ' => "yo",
        'ら' => "ra", 'り' => "ri", 'る' => "ru", 'れ' => "re", 'ろ' => "ro",
        'わ' => "wa", 'を' => "wo",
        'ぁ' => "xa", 'ぃ' => "xi", 'ぅ' => "xu", 'ぇ' => "xe", 'ぉ' => "xo",
        'ゃ' => "xya", 'ゅ' => "xyu", 'ょ' => "xyo",
        'ー' => "-",
        _ => return None,
    })
}

/// Consonant prefix used for ゃゅょ digraphs (きゃ → k + ya → "kya").
fn digraph_consonant(c: char) -> Option<&'static str> {
    Some(match c {
        'き' => "ky", 'ぎ' => "gy", 'し' => "sy", 'じ' => "zy",
        'ち' => "ty", 'ぢ' => "dy", 'に' => "ny", 'ひ' => "hy",
        'び' => "by", 'ぴ' => "py", 'み' => "my", 'り' => "ry",
        _ => return None,
    })
}

/// Convert a kana region (possibly containing stray ASCII letters — the
/// leftovers of the typo itself) to the romaji keystrokes that produced it.
/// Returns `None` when a character can't be mapped (symbols, katakana, ...).
fn region_to_romaji(region: &str) -> Option<String> {
    let chars: Vec<char> = region.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_alphabetic() {
            out.push(c.to_ascii_lowercase());
            i += 1;
            continue;
        }
        if c == 'ん' {
            out.push_str("nn");
            i += 1;
            continue;
        }
        if c == 'っ' {
            // Gemination doubles the next kana's initial consonant.
            let doubled = chars
                .get(i + 1)
                .and_then(|&n| kana_to_romaji_char(n))
                .and_then(|r| r.chars().next())
                .filter(|k| !"aiueon-".contains(*k));
            match doubled {
                Some(k) => out.push(k),
                None => out.push_str("xtu"),
            }
            i += 1;
            continue;
        }
        // ゃゅょ digraphs: consonant prefix + ya/yu/yo.
        if let Some(&next) = chars.get(i + 1)
            && matches!(next, 'ゃ' | 'ゅ' | 'ょ')
            && let Some(prefix) = digraph_consonant(c)
        {
            out.push_str(prefix);
            out.push(match next {
                'ゃ' => 'a',
                'ゅ' => 'u',
                _ => 'o',
            });
            i += 2;
            continue;
        }
        out.push_str(kana_to_romaji_char(c)?);
        i += 1;
    }
    Some(out)
}

/// Round-trip a romaji string back to kana. `None` when the conversion is
/// incomplete (stranded consonants left over) — such an edit can't be what
/// the user meant to type.
fn romaji_to_kana(romaji: &str) -> Option<String> {
    let mut conv = RomajiConverter::new();
    for ch in romaji.chars() {
        conv.push(ch);
    }
    conv.flush();
    let out = conv.output().to_string();
    out.chars()
        .all(|c| !c.is_ascii_alphabetic())
        .then_some(out)
}

/// "Complete the keystroke" variants for a reading with stray ASCII in it:
/// instead of *removing* the leftover romaji (the strip pattern), treat it as
/// an interrupted kana whose vowel was never pressed and finish it with each
/// vowel — `wたし` → わたし/うぃたし/うたし/うぇたし/をたし, `けっk` →
/// けっか/けっき/…. Only the first ASCII run is completed (typos are almost
/// always a single slip); variants that don't round-trip to pure kana are
/// dropped. Returns full readings, never equal to the input.
pub fn romaji_completion_variants(reading: &str) -> Vec<String> {
    let chars: Vec<char> = reading.chars().collect();
    let Some(run_start) = chars.iter().position(|c| c.is_ascii_alphabetic()) else {
        return vec![];
    };
    let run_end = chars[run_start..]
        .iter()
        .position(|c| !c.is_ascii_alphabetic())
        .map_or(chars.len(), |n| run_start + n);
    let prefix: String = chars[..run_start].iter().collect();
    let run: String = chars[run_start..run_end]
        .iter()
        .collect::<String>()
        .to_ascii_lowercase();
    let suffix: String = chars[run_end..].iter().collect();

    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for v in ['a', 'i', 'u', 'e', 'o'] {
        let Some(kana) = romaji_to_kana(&format!("{run}{v}")) else {
            continue;
        };
        let variant = format!("{prefix}{kana}{suffix}");
        if variant != reading && seen.insert(variant.clone()) {
            out.push(variant);
        }
    }
    out
}

/// Keyboard-plausible corrections of a suspicious kana region.
///
/// Edits are applied in romaji space, ordered by how common each mistake is
/// on a physical keyboard — adjacent transpositions first, then neighbor-key
/// substitutions, deletions, and vowel insertions after stranded consonants —
/// then converted back to kana. Variants that don't round-trip to pure kana
/// are dropped. Returns at most `max` distinct kana strings, none equal to
/// the original region.
pub fn keyboard_typo_variants(region: &str, max: usize) -> Vec<String> {
    let Some(romaji) = region_to_romaji(region) else {
        return vec![];
    };
    let chars: Vec<char> = romaji.chars().collect();
    let mut edits: Vec<String> = Vec::new();

    // 1. Adjacent transpositions (`tmae` → `tame`).
    for i in 0..chars.len().saturating_sub(1) {
        if chars[i] != chars[i + 1] {
            let mut v = chars.clone();
            v.swap(i, i + 1);
            edits.push(v.into_iter().collect());
        }
    }
    // 2. Neighbor-key substitutions (`r` typed for `t`, ...).
    for i in 0..chars.len() {
        for n in qwerty_neighbors(chars[i]).chars() {
            let mut v = chars.clone();
            v[i] = n;
            edits.push(v.into_iter().collect());
        }
    }
    // 3. Single deletions (an extra key slipped in).
    for i in 0..chars.len() {
        edits.push(chars[..i].iter().chain(&chars[i + 1..]).collect());
    }
    // 4. Vowel insertion after a stranded consonant (a vowel key missed):
    //    only where a consonant is followed by another consonant or ends the
    //    region, i.e. where the romaji is already broken.
    for i in 0..chars.len() {
        let c = chars[i];
        if !"aiueon-".contains(c) {
            let next_breaks = chars
                .get(i + 1)
                .is_none_or(|&n| !"aiueoyn".contains(n));
            if next_breaks {
                for v in ['a', 'i', 'u', 'e', 'o'] {
                    let mut e = chars.clone();
                    e.insert(i + 1, v);
                    edits.push(e.into_iter().collect());
                }
            }
        }
    }

    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for edit in edits {
        if out.len() >= max {
            break;
        }
        let Some(kana) = romaji_to_kana(&edit) else {
            continue;
        };
        if kana != region && seen.insert(kana.clone()) {
            out.push(kana);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transposition_recovers_intended_kana() {
        // `t前` case: keystrokes `tmae` — one transposition from `tame`.
        let variants = keyboard_typo_variants("tまえ", 24);
        assert!(
            variants.iter().any(|v| v == "ため"),
            "expected ため among variants, got {variants:?}"
        );
    }

    #[test]
    fn completion_variants_finish_the_keystroke() {
        let variants = romaji_completion_variants("wたし");
        assert!(
            variants.iter().any(|v| v == "わたし"),
            "expected わたし among completions, got {variants:?}"
        );
        // Trailing interrupted kana completes too.
        let variants = romaji_completion_variants("けっk");
        assert!(
            variants.iter().any(|v| v == "けっか"),
            "expected けっか among completions, got {variants:?}"
        );
        // No ASCII → nothing to complete.
        assert!(romaji_completion_variants("わたし").is_empty());
    }

    #[test]
    fn missed_vowel_recovers_watashi() {
        // `wたし` case: the vowel after `w` was never pressed — keystrokes
        // `wtasi`, one vowel insertion from `watasi` (わたし).
        let variants = keyboard_typo_variants("wたし", 14);
        assert!(
            variants.iter().any(|v| v == "わたし"),
            "expected わたし among variants, got {variants:?}"
        );
    }

    #[test]
    fn neighbor_substitution_generates_variants() {
        // さ (sa) with `s`→`d` neighbor slip would have produced だ (da).
        let variants = keyboard_typo_variants("さ", 48);
        assert!(
            variants.iter().any(|v| v == "だ"),
            "expected だ among variants, got {variants:?}"
        );
    }

    #[test]
    fn round_trip_rejects_broken_edits() {
        // All returned variants must be pure kana (no stranded consonants).
        for v in keyboard_typo_variants("だったtまえ", 48) {
            assert!(
                v.chars().all(|c| !c.is_ascii_alphabetic()),
                "variant {v} contains stray ascii"
            );
        }
    }

    #[test]
    fn unmappable_region_returns_empty() {
        assert!(keyboard_typo_variants("漢字", 8).is_empty());
    }
}
