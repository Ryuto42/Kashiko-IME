//! Recognizing a latin word the romaji converter mangled into kana.
//!
//! Typing `WordPress` produces the buffer `WおrdPレッ`: the romaji rules eat
//! whichever letters happen to form a kana, leaving the intended spelling
//! recoverable only from the raw keystrokes. This module decides whether a
//! composition is such a word — as opposed to ordinary Japanese input (whose
//! keystrokes are *also* pure ASCII romaji), a single mistyped key, or a mixed
//! sentence whose latin part is already intact — and scores how confident that
//! reading is, so the caller can rank the candidate against real conversions
//! instead of pinning it to the top or the bottom of the list.

/// Hiragana or katakana (including the prolonged sound mark).
fn is_kana(c: char) -> bool {
    matches!(c, '\u{3041}'..='\u{3096}' | '\u{30A1}'..='\u{30FA}' | '\u{30FC}')
}

/// Letters that Japanese romaji tables almost never produce on their own, so
/// their presence points at an English word rather than a reading.
fn is_rare_in_romaji(c: char) -> bool {
    matches!(c.to_ascii_lowercase(), 'l' | 'q' | 'v' | 'x')
}

fn is_vowel(c: char) -> bool {
    matches!(c.to_ascii_lowercase(), 'a' | 'i' | 'u' | 'e' | 'o')
}

/// Confidence that `raw` (the keystrokes) is a latin word the converter
/// mangled into `buffer`, or `None` when the composition isn't one at all.
///
/// The hard gates matter as much as the score:
/// - the keystrokes must be ≥3 ASCII alphanumerics containing a letter, and
///   must differ from the buffer (otherwise nothing was mangled);
/// - latin letters surviving in the buffer must strictly outnumber its kana.
///   This is what separates `WおrdPレッ` (4 letters vs 3 kana → a latin word)
///   from ordinary Japanese like `わたし` (0 letters, keystrokes `watasi`),
///   a single slip like `wたし` (1 vs 2 → a typo, not a word), and a mixed
///   sentence like `わたしはLinuxが` (5 vs 5) whose latin run is already
///   correct and needs no recovery.
///
/// Higher scores mean "more English-looking": surviving letters, interior
/// capitals (`WordPress`, `iPhone`), letters rare in romaji (`l`/`q`/`v`/`x`),
/// consonant clusters romaji can't form, digits, and length.
pub fn latin_word_confidence(raw: &str, buffer: &str) -> Option<u32> {
    let raw_chars: Vec<char> = raw.chars().collect();
    if raw_chars.len() < 3
        || raw == buffer
        || !raw_chars.iter().any(|c| c.is_ascii_alphabetic())
        || !raw_chars.iter().all(|c| c.is_ascii_alphanumeric())
    {
        return None;
    }

    let buf_letters = buffer.chars().filter(|c| c.is_ascii_alphabetic()).count();
    let buf_kana = buffer.chars().filter(|c| is_kana(*c)).count();
    if buf_letters == 0 || buf_letters <= buf_kana {
        return None;
    }

    let mut score = 0u32;
    // Letters the romaji rules couldn't consume: the core evidence.
    score += buf_letters.min(4) as u32 * 2;
    // An interior capital is a spelling no reading has.
    if raw_chars.iter().skip(1).any(|c| c.is_ascii_uppercase()) {
        score += 4;
    }
    score += raw_chars
        .iter()
        .filter(|c| is_rare_in_romaji(**c))
        .count()
        .min(2) as u32
        * 2;
    // Three consecutive consonants (`rdPr` in WordPress) — romaji syllables
    // can't produce that.
    let mut run = 0usize;
    for c in &raw_chars {
        if c.is_ascii_alphabetic() && !is_vowel(*c) {
            run += 1;
            if run >= 3 {
                score += 2;
                break;
            }
        } else {
            run = 0;
        }
    }
    if raw_chars.iter().any(|c| c.is_ascii_digit()) {
        score += 1;
    }
    if raw_chars.len() >= 5 {
        score += 1;
    }
    Some(score)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_mangled_latin_words() {
        // `WordPress` → `WおrdPレッ`: 4 surviving letters vs 3 kana.
        let score = latin_word_confidence("WordPress", "WおrdPレッ").unwrap();
        assert!(score >= 8, "WordPress should score strongly, got {score}");
        // `iPhone` → `いPhone`: the leading vowel became kana.
        assert!(latin_word_confidence("iPhone", "いPhone").unwrap() >= 8);
    }

    #[test]
    fn rejects_ordinary_japanese_input() {
        // The headline false positive: ordinary Japanese keystrokes are pure
        // ASCII romaji too, but nothing latin survives in the buffer.
        assert_eq!(latin_word_confidence("watasi", "わたし"), None);
        assert_eq!(latin_word_confidence("kyounotenki", "きょうのてんき"), None);
    }

    #[test]
    fn rejects_single_key_typos() {
        // `wたし` is a missed vowel, not a word: 1 letter vs 2 kana. The
        // keyboard-typo search owns this case.
        assert_eq!(latin_word_confidence("wtasi", "wたし"), None);
    }

    #[test]
    fn rejects_mixed_sentence_with_intact_latin() {
        // `Linux` is already correct in the buffer, so the keystrokes
        // (`watasihaLinuxga`) must not be offered as a candidate.
        assert_eq!(
            latin_word_confidence("watasihaLinuxga", "わたしはLinuxが"),
            None
        );
    }

    #[test]
    fn rejects_unmangled_and_too_short() {
        // Pure alphabet-mode input needs no recovery.
        assert_eq!(latin_word_confidence("hello", "hello"), None);
        assert_eq!(latin_word_confidence("ab", "aび"), None);
        // Non-alphanumeric keystrokes aren't a word.
        assert_eq!(latin_word_confidence("a-b!", "aーb"), None);
    }

    #[test]
    fn scores_rank_more_english_looking_words_higher() {
        // Rare letters and an interior capital outrank a plain fragment.
        let strong = latin_word_confidence("SQLite", "SQLいて").unwrap();
        let weak = latin_word_confidence("dsk", "dsк").unwrap_or(0);
        assert!(strong > weak, "strong={strong} weak={weak}");
    }
}
