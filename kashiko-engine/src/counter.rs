//! Number + counter readings (`にげん` → `2限`).
//!
//! Japanese writes counted quantities with a digit far more often than with the
//! kanji numeral, but the conversion chain has no way to produce one: the
//! dictionary is keyed on whole readings and has no `にげん` entry, and the
//! model's beam only sometimes stumbles on the digit form (it finds `2階` for
//! `にかい` but not `2限` for `にげん`). Measured against the real model, `2限`
//! scores 1.03 where the best candidate offered scored 0.42 and the rest sat
//! above 2.0 — the right answer was simply never on the list.
//!
//! Two tables do the work: number readings (including the geminated and voiced
//! forms a counter triggers, `いっ`/`ろっ`/`じゅっ`) and the counters themselves
//! (`data/counters.tsv`). A reading is a match only when it splits cleanly into
//! one of each, which is what keeps `しごと` from becoming `4毎`.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::kana::katakana_to_hiragana;

const COUNTERS_TSV: &str = include_str!("../data/counters.tsv");

/// Readings of the digits 0–9, plus 10, as they appear before a counter.
///
/// Alternates are listed because Japanese has more than one reading for
/// several digits (`よん`/`し`, `なな`/`しち`, `きゅう`/`く`) and because a
/// following counter geminates some of them (`いっ`, `ろっ`, `はっ`, `じゅっ`).
const UNITS: [(&str, u32); 20] = [
    ("ぜろ", 0),
    ("れい", 0),
    ("いち", 1),
    ("いっ", 1),
    ("に", 2),
    ("さん", 3),
    ("よん", 4),
    ("し", 4),
    ("よ", 4),
    ("ご", 5),
    ("ろく", 6),
    ("ろっ", 6),
    ("なな", 7),
    ("しち", 7),
    ("はち", 8),
    ("はっ", 8),
    ("きゅう", 9),
    ("く", 9),
    ("じゅう", 10),
    ("じゅっ", 10),
];

/// Every number reading this module recognises, mapped to its value.
///
/// Built from [`UNITS`] rather than listed: 1–9, 10–19 (`じゅう` + unit),
/// the round tens (unit + `じゅう`) and 21–99 (unit + `じゅう` + unit), each
/// with every alternate reading of every part.
fn numbers() -> &'static HashMap<String, u32> {
    static NUMBERS: OnceLock<HashMap<String, u32>> = OnceLock::new();
    NUMBERS.get_or_init(|| {
        let mut m: HashMap<String, u32> = HashMap::new();
        let digits = || UNITS.iter().filter(|(_, v)| (1..=9).contains(v));
        let tens = || UNITS.iter().filter(|(_, v)| *v == 10);

        // Shorter readings win a collision: `し` is 4, never 4 spelled oddly.
        let mut put = |reading: String, value: u32| {
            m.entry(reading).or_insert(value);
        };
        for (r, v) in UNITS {
            put(r.to_string(), v);
        }
        for (tr, _) in tens() {
            for (ur, uv) in digits() {
                put(format!("{tr}{ur}"), 10 + uv); // 11–19
            }
        }
        for (dr, dv) in digits() {
            if *dv == 1 {
                continue; // 「いちじゅう」は言わない
            }
            for (tr, _) in tens() {
                put(format!("{dr}{tr}"), dv * 10); // 20, 30, …
                for (ur, uv) in digits() {
                    put(format!("{dr}{tr}{ur}"), dv * 10 + uv); // 21–99
                }
            }
        }
        m
    })
}

fn counters() -> &'static HashMap<&'static str, Vec<&'static str>> {
    static COUNTERS: OnceLock<HashMap<&'static str, Vec<&'static str>>> = OnceLock::new();
    COUNTERS.get_or_init(|| {
        COUNTERS_TSV
            .lines()
            .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
            .filter_map(|l| {
                let mut parts = l.split('\t').map(str::trim).filter(|p| !p.is_empty());
                let reading = parts.next()?;
                let surfaces: Vec<&str> = parts.collect();
                (!surfaces.is_empty()).then_some((reading, surfaces))
            })
            .collect()
    })
}

/// Whether `c` can start a counter surface (`件`, `階`, `時`, `ページ`, …).
///
/// Used to tell a digit that belongs in the text from one the model invented:
/// Japanese writes a digit before a counter, so `候補が4件…` is a real
/// conversion of `…よんけん…` while a digit anywhere else came from nowhere.
pub fn starts_counter(c: char) -> bool {
    static FIRSTS: OnceLock<Vec<char>> = OnceLock::new();
    FIRSTS
        .get_or_init(|| {
            let mut v: Vec<char> = counters()
                .values()
                .flatten()
                .filter_map(|s| s.chars().next())
                .collect();
            v.sort_unstable();
            v.dedup();
            v
        })
        .binary_search(&c)
        .is_ok()
}

/// `reading` as `<digits><counter>`, best first, or empty when it does not
/// split into a number followed by a counter.
///
/// A reading can split more than one way (`さんかい` is `3階` and `3回`); every
/// surface the counter lists is returned, in the table's order.
pub fn candidates(reading: &str) -> Vec<String> {
    let reading = katakana_to_hiragana(reading);
    let chars: Vec<char> = reading.chars().collect();
    let mut out: Vec<String> = Vec::new();
    // Longest number prefix first: `じゅうにかい` is 12階, not 10 followed by a
    // counter reading `にかい`.
    for split in (1..chars.len()).rev() {
        let head: String = chars[..split].iter().collect();
        let tail: String = chars[split..].iter().collect();
        let (Some(value), Some(surfaces)) = (numbers().get(&head), counters().get(tail.as_str()))
        else {
            continue;
        };
        for surface in surfaces {
            let text = format!("{value}{surface}");
            if !out.contains(&text) {
                out.push(text);
            }
        }
        // One number reading is enough; a shorter split of the same reading
        // would be a different, less likely number.
        if !out.is_empty() {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_reported_reading_converts() {
        assert_eq!(candidates("にげん"), vec!["2限", "2軒"]);
    }

    #[test]
    fn a_counter_with_several_surfaces_offers_each() {
        assert_eq!(candidates("さんかい"), vec!["3階", "3回"]);
    }

    #[test]
    fn geminated_and_voiced_forms_are_recognised() {
        assert_eq!(candidates("いっかい"), vec!["1階", "1回"]);
        assert_eq!(candidates("さんがい"), vec!["3階"]);
        assert_eq!(candidates("ろっぽん"), vec!["6本"]);
    }

    #[test]
    fn two_digit_numbers_work() {
        assert_eq!(candidates("じゅうにじ"), vec!["12時"]);
        assert_eq!(candidates("にじゅうごさい"), vec!["25歳", "25才"]);
    }

    #[test]
    fn the_longest_number_prefix_wins() {
        // `じゅうにかい` is the 12th floor, not 10 followed by some counter.
        assert_eq!(candidates("じゅうにかい"), vec!["12階", "12回"]);
    }

    #[test]
    fn a_reading_that_is_not_number_plus_counter_yields_nothing() {
        // `ごと` is not a counter, so `しごと` must not become `4毎`.
        for reading in ["しごと", "かんじ", "たなか", "ゆうと", "にんい", ""] {
            assert!(
                candidates(reading).is_empty(),
                "{reading} produced {:?}",
                candidates(reading)
            );
        }
    }

    #[test]
    fn counter_initials_are_recognised() {
        for c in ['件', '階', '限', '時', '年', '人', '本', 'ペ'] {
            assert!(starts_counter(c), "{c} should start a counter");
        }
        for c in ['慰', '文', '私', 'あ', 'A'] {
            assert!(!starts_counter(c), "{c} should not start a counter");
        }
    }

    #[test]
    fn a_bare_number_is_not_a_counted_quantity() {
        // Nothing follows the number, so there is no counter and no candidate;
        // the number rewriter owns plain digits.
        assert!(candidates("じゅう").is_empty());
    }
}
