//! Katakana loanwords that a writer usually wants in Latin script.
//!
//! `テンプレート` and `ワークフロー` are borrowed words, so no dictionary or
//! model will ever offer `template` or `workflow` for them — the conversion
//! chain only ever produces Japanese script. Engineering and business prose is
//! full of terms written in Latin script even by Japanese writers, and this is
//! the only source that can supply them.
//!
//! The table is deliberately not "every loanword": everyday borrowings that
//! read naturally in katakana (パン, コーヒー, テレビ) are absent, because
//! offering `bread` for パン would be noise. Data lives in
//! `data/loanwords.tsv`.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::kana::hiragana_to_katakana;

const LOANWORDS_TSV: &str = include_str!("../data/loanwords.tsv");

fn table() -> &'static HashMap<&'static str, &'static str> {
    static TABLE: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    TABLE.get_or_init(|| {
        LOANWORDS_TSV
            .lines()
            .filter(|l| !l.trim_start().starts_with('#') && !l.trim().is_empty())
            .filter_map(|l| l.split_once('\t'))
            .map(|(k, v)| (k.trim(), v.trim()))
            .filter(|(k, v)| !k.is_empty() && !v.is_empty())
            .collect()
    })
}

/// The English spelling for `reading`, if it is one of the listed loanwords.
///
/// Accepts the reading in either kana, since the buffer holds hiragana while the
/// table is keyed by katakana.
pub fn english_for(reading: &str) -> Option<&'static str> {
    table().get(hiragana_to_katakana(reading).as_str()).copied()
}

/// How many loanwords are listed. Exposed so a test can catch a data file that
/// silently failed to parse.
pub fn len() -> usize {
    table().len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_data_file_parses() {
        assert!(len() > 300, "only {} entries parsed", len());
    }

    #[test]
    fn the_reported_words_are_covered() {
        assert_eq!(english_for("てんぷれーと"), Some("template"));
        assert_eq!(english_for("わーくふろー"), Some("workflow"));
    }

    #[test]
    fn katakana_and_hiragana_readings_agree() {
        assert_eq!(english_for("リポジトリ"), english_for("りぽじとり"));
    }

    #[test]
    fn multi_word_english_is_allowed() {
        assert_eq!(english_for("ぷるりくえすと"), Some("pull request"));
    }

    #[test]
    fn everyday_loanwords_are_left_alone() {
        // Offering `bread` for パン would be noise, so the table stops short of
        // borrowings that read naturally in katakana.
        for reading in ["ぱん", "こーひー", "てれび", "ぼーる"] {
            assert_eq!(english_for(reading), None, "{reading} should not be listed");
        }
    }

    #[test]
    fn unlisted_readings_return_nothing() {
        assert_eq!(english_for("かんじ"), None);
        assert_eq!(english_for(""), None);
    }
}
