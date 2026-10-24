use unicode_normalization::UnicodeNormalization;

/// Apply NFKC normalization to text.
///
/// This is needed for models whose tokenizer does NOT support full-width ASCII
/// characters in its vocabulary. Without NFKC normalization, characters like
/// `（`, `）`, `！`, `？` are incorrectly tokenized as EOS tokens, causing
/// generation to stop prematurely.
///
/// NFKC normalization converts:
/// - Full-width ASCII → Half-width: `（` → `(`, `！` → `!`, `？` → `?`
/// - Full-width digits → Half-width: `０` → `0`, `１` → `1`
/// - Compatibility characters → Canonical forms
///
/// Note: Hiragana, Katakana, and Kanji are NOT affected by NFKC normalization.
/// The special jinen tokens (U+EE00-U+EE02) in Private Use Area are also preserved.
pub fn normalize_nfkc(text: &str) -> String {
    text.nfkc().collect()
}

/// True if the text contains any hiragana or katakana **letter**.
///
/// Used to distinguish "real" reading inputs (kana the model can convert)
/// from symbol-only or alphabet-only inputs that would only produce
/// hallucinated model output.
///
/// Only actual kana letters count — punctuation that lives in the katakana
/// block (U+30A0 double hyphen, U+30FB middle dot `・`, U+30FC prolonged
/// mark `ー`, iteration marks U+30FD–U+30FE) is intentionally excluded.
/// Otherwise typing just `・` or `ー` would let the model run on a
/// punctuation-only reading and hallucinate.
pub fn contains_kana(text: &str) -> bool {
    text.chars()
        .any(|c| matches!(c, '\u{3041}'..='\u{3096}' | '\u{30A1}'..='\u{30FA}'))
}

/// True if every character is kana — a letter, or the punctuation that belongs
/// to a kana word (`ー`, `・`, iteration marks). Empty text is not kana.
///
/// The complement of "contains a kanji, a latin letter or a digit", which is
/// what a caller checking for an unconverted reading actually wants.
pub fn is_all_kana(text: &str) -> bool {
    !text.is_empty()
        && text.chars().all(|c| {
            matches!(c, '\u{3041}'..='\u{3096}' | '\u{309D}'..='\u{309E}' | '\u{30A0}'..='\u{30FE}')
        })
}

/// Convert hiragana to katakana
pub fn hiragana_to_katakana(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            // Hiragana range (U+3041-U+3096) -> Katakana (U+30A1-U+30F6)
            '\u{3041}'..='\u{3096}' => std::char::from_u32(c as u32 + 0x60).unwrap_or(c),
            _ => c,
        })
        .collect()
}

/// Convert katakana to hiragana
pub fn katakana_to_hiragana(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            // Katakana range (U+30A1-U+30F6) -> Hiragana (U+3041-U+3096)
            '\u{30A1}'..='\u{30F6}' => std::char::from_u32(c as u32 - 0x60).unwrap_or(c),
            _ => c,
        })
        .collect()
}

/// Map a single full-width katakana char to its half-width form.
///
/// Voiced/semi-voiced characters expand to two chars (base + dakuten/handakuten).
/// Returns the original char as a single-char string for non-katakana input.
fn katakana_char_to_half(c: char) -> String {
    match c {
        // Sokuon, small kana
        'ァ' => "ｧ".into(),
        'ィ' => "ｨ".into(),
        'ゥ' => "ｩ".into(),
        'ェ' => "ｪ".into(),
        'ォ' => "ｫ".into(),
        'ッ' => "ｯ".into(),
        'ャ' => "ｬ".into(),
        'ュ' => "ｭ".into(),
        'ョ' => "ｮ".into(),
        // a-row through wo
        'ア' => "ｱ".into(),
        'イ' => "ｲ".into(),
        'ウ' => "ｳ".into(),
        'エ' => "ｴ".into(),
        'オ' => "ｵ".into(),
        'カ' => "ｶ".into(),
        'キ' => "ｷ".into(),
        'ク' => "ｸ".into(),
        'ケ' => "ｹ".into(),
        'コ' => "ｺ".into(),
        'サ' => "ｻ".into(),
        'シ' => "ｼ".into(),
        'ス' => "ｽ".into(),
        'セ' => "ｾ".into(),
        'ソ' => "ｿ".into(),
        'タ' => "ﾀ".into(),
        'チ' => "ﾁ".into(),
        'ツ' => "ﾂ".into(),
        'テ' => "ﾃ".into(),
        'ト' => "ﾄ".into(),
        'ナ' => "ﾅ".into(),
        'ニ' => "ﾆ".into(),
        'ヌ' => "ﾇ".into(),
        'ネ' => "ﾈ".into(),
        'ノ' => "ﾉ".into(),
        'ハ' => "ﾊ".into(),
        'ヒ' => "ﾋ".into(),
        'フ' => "ﾌ".into(),
        'ヘ' => "ﾍ".into(),
        'ホ' => "ﾎ".into(),
        'マ' => "ﾏ".into(),
        'ミ' => "ﾐ".into(),
        'ム' => "ﾑ".into(),
        'メ' => "ﾒ".into(),
        'モ' => "ﾓ".into(),
        'ヤ' => "ﾔ".into(),
        'ユ' => "ﾕ".into(),
        'ヨ' => "ﾖ".into(),
        'ラ' => "ﾗ".into(),
        'リ' => "ﾘ".into(),
        'ル' => "ﾙ".into(),
        'レ' => "ﾚ".into(),
        'ロ' => "ﾛ".into(),
        'ワ' => "ﾜ".into(),
        'ヲ' => "ｦ".into(),
        'ン' => "ﾝ".into(),
        // Voiced (dakuten) — expand to base + ﾞ
        'ガ' => "ｶﾞ".into(),
        'ギ' => "ｷﾞ".into(),
        'グ' => "ｸﾞ".into(),
        'ゲ' => "ｹﾞ".into(),
        'ゴ' => "ｺﾞ".into(),
        'ザ' => "ｻﾞ".into(),
        'ジ' => "ｼﾞ".into(),
        'ズ' => "ｽﾞ".into(),
        'ゼ' => "ｾﾞ".into(),
        'ゾ' => "ｿﾞ".into(),
        'ダ' => "ﾀﾞ".into(),
        'ヂ' => "ﾁﾞ".into(),
        'ヅ' => "ﾂﾞ".into(),
        'デ' => "ﾃﾞ".into(),
        'ド' => "ﾄﾞ".into(),
        'バ' => "ﾊﾞ".into(),
        'ビ' => "ﾋﾞ".into(),
        'ブ' => "ﾌﾞ".into(),
        'ベ' => "ﾍﾞ".into(),
        'ボ' => "ﾎﾞ".into(),
        'ヴ' => "ｳﾞ".into(),
        // Semi-voiced (handakuten) — expand to base + ﾟ
        'パ' => "ﾊﾟ".into(),
        'ピ' => "ﾋﾟ".into(),
        'プ' => "ﾌﾟ".into(),
        'ペ' => "ﾍﾟ".into(),
        'ポ' => "ﾎﾟ".into(),
        // Long sound, punctuation
        'ー' => "ｰ".into(),
        '・' => "･".into(),
        '。' => "｡".into(),
        '、' => "､".into(),
        '「' => "｢".into(),
        '」' => "｣".into(),
        // Standalone dakuten / handakuten
        '゛' => "ﾞ".into(),
        '゜' => "ﾟ".into(),
        _ => c.to_string(),
    }
}

/// True if every character is a hiragana letter (U+3041–U+3096).
///
/// Used to decide whether a candidate deserves the mozc-style `[全]ひらがな`
/// width-form annotation. Empty strings return false.
pub fn is_pure_hiragana(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|c| matches!(c, '\u{3041}'..='\u{3096}'))
}

/// True if every character is a full-width katakana letter (U+30A1–U+30FA,
/// plus the prolonged sound mark U+30FC).
///
/// Used to decide whether a candidate deserves the mozc-style `[全]カタカナ`
/// width-form annotation. Empty strings return false.
pub fn is_pure_full_katakana(text: &str) -> bool {
    !text.is_empty()
        && text
            .chars()
            .all(|c| matches!(c, '\u{30A1}'..='\u{30FA}' | '\u{30FC}'))
}

/// Whether `converted` contains a run of `min_run`+ consecutive hiragana that
/// also appears verbatim in `reading` — a strong signal the model passed that
/// stretch through *unconverted* (it couldn't make sense of it), which is what
/// a typo region looks like. e.g. reading `できないようらなけす` (typo of
/// `…ようなら…`) yields the candidate `できないようらな消す`, whose 7-char
/// run `できないようらな` is a substring of the reading → returns true. The
/// correctly-converted `出来ないようなら消す` does *not* contain such a run
/// (`ないようなら` is not in that typo'd reading) → returns false.
///
/// Kanji output naturally breaks up hiragana runs, so a genuine conversion
/// (with okurigana ≤ a few kana between kanji) stays under the threshold while
/// a passed-through typo stretch trips it.
pub fn has_long_unconverted_run(converted: &str, reading: &str, min_run: usize) -> bool {
    longest_passthrough_run(converted, reading, min_run).is_some()
}

/// The longest hiragana run in `converted` (at least `min_run` chars) that
/// also appears verbatim in `reading` — i.e. the stretch the model passed
/// through unconverted. Returns `None` when no such run exists. Used both to
/// flag low-quality candidates (see [`has_long_unconverted_run`]) and to
/// locate the typo region for kana-level correction search.
pub fn longest_passthrough_run(converted: &str, reading: &str, min_run: usize) -> Option<String> {
    let is_hira = |c: char| matches!(c, '\u{3041}'..='\u{3096}');
    let mut best: Option<String> = None;
    let mut run = String::new();
    let consider = |run: &str, best: &mut Option<String>| {
        if run.chars().count() >= min_run
            && reading.contains(run)
            && best
                .as_ref()
                .is_none_or(|b| run.chars().count() > b.chars().count())
        {
            *best = Some(run.to_string());
        }
    };
    for c in converted.chars() {
        if is_hira(c) {
            run.push(c);
        } else {
            consider(&run, &mut best);
            run.clear();
        }
    }
    consider(&run, &mut best);
    best
}

/// Convert full-width katakana to half-width katakana.
///
/// Voiced characters expand into two half-width characters (base + ﾞ/ﾟ).
/// Non-katakana characters pass through unchanged.
pub fn katakana_to_half_width(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        out.push_str(&katakana_char_to_half(c));
    }
    out
}

/// Convert hiragana to half-width katakana.
///
/// Equivalent to `katakana_to_half_width(hiragana_to_katakana(text))`.
pub fn hiragana_to_half_katakana(text: &str) -> String {
    katakana_to_half_width(&hiragana_to_katakana(text))
}

/// Map a half-width ASCII alphanumeric character (digit / Latin letter) to
/// its full-width form (e.g. `a` → `ａ`, `Z` → `Ｚ`, `5` → `５`). All other
/// characters pass through unchanged.
pub fn ascii_to_fullwidth_char(c: char) -> char {
    match c {
        '0'..='9' => char::from_u32(c as u32 - 0x30 + 0xFF10).unwrap_or(c),
        'A'..='Z' => char::from_u32(c as u32 - 0x41 + 0xFF21).unwrap_or(c),
        'a'..='z' => char::from_u32(c as u32 - 0x61 + 0xFF41).unwrap_or(c),
        _ => c,
    }
}

/// Map a full-width ASCII alphanumeric character to its half-width form
/// (e.g. `ａ` → `a`, `Ｚ` → `Z`, `５` → `5`). All other characters pass
/// through unchanged.
pub fn fullwidth_to_ascii_char(c: char) -> char {
    match c {
        '\u{FF10}'..='\u{FF19}' => char::from_u32(c as u32 - 0xFF10 + 0x30).unwrap_or(c),
        '\u{FF21}'..='\u{FF3A}' => char::from_u32(c as u32 - 0xFF21 + 0x41).unwrap_or(c),
        '\u{FF41}'..='\u{FF5A}' => char::from_u32(c as u32 - 0xFF41 + 0x61).unwrap_or(c),
        _ => c,
    }
}

/// Map printable ASCII to its full-width form (space becomes U+3000).
pub fn ascii_to_fullwidth(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            ' ' => '\u{3000}',
            '!'..='~' => char::from_u32(c as u32 - 0x21 + 0xFF01).unwrap_or(c),
            _ => c,
        })
        .collect()
}

/// Map full-width ASCII forms (and U+3000) back to half-width ASCII.
pub fn fullwidth_to_ascii(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\u{3000}' => ' ',
            '\u{FF01}'..='\u{FF5E}' => char::from_u32(c as u32 - 0xFF01 + 0x21).unwrap_or(c),
            _ => c,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contains_kana() {
        // Real kana letters → true
        assert!(contains_kana("あ"));
        assert!(contains_kana("ア"));
        assert!(contains_kana("ヴ")); // U+30F4 in U+30A1..=U+30FA
        assert!(contains_kana("コーヒー"));
        assert!(contains_kana("カ・ド")); // mixed kana + middle dot still has kana

        // Punctuation in the katakana block — NOT kana.
        // Without this exclusion, typing just `・` or `ー` would let the
        // model run on a punctuation-only reading and hallucinate.
        assert!(!contains_kana("・"));
        assert!(!contains_kana("ー"));
        assert!(!contains_kana("ヽ"));
        assert!(!contains_kana("ヾ"));

        // Other non-kana inputs.
        assert!(!contains_kana(""));
        assert!(!contains_kana("123"));
        assert!(!contains_kana("「」"));
        assert!(!contains_kana("漢字")); // kanji, not kana
        assert!(!contains_kana("abc"));
    }

    #[test]
    fn test_is_pure_hiragana() {
        assert!(is_pure_hiragana("あ"));
        assert!(is_pure_hiragana("あいうえお"));
        assert!(is_pure_hiragana("がっこう"));

        assert!(!is_pure_hiragana(""));
        assert!(!is_pure_hiragana("ア")); // katakana
        assert!(!is_pure_hiragana("あア")); // mixed
        assert!(!is_pure_hiragana("あ漢"));
        assert!(!is_pure_hiragana("ーあ")); // prolonged mark is katakana block
    }

    #[test]
    fn test_has_long_unconverted_run() {
        // Typo passthrough: the model left the mistyped stretch
        // (`できないようらな` = 8 hiragana) as raw kana, and it's present in the
        // reading. Trips at the production threshold of 7.
        let reading = "できないようらなけす";
        assert!(has_long_unconverted_run("できないようらな消す", reading, 7));
        // The correctly-converted candidate breaks the run with kanji and its
        // remaining hiragana ("ないようなら") is NOT in the typo'd reading.
        assert!(!has_long_unconverted_run(
            "出来ないようなら消す",
            reading,
            7
        ));
        // A normal conversion with only short okurigana runs is fine.
        assert!(!has_long_unconverted_run(
            "私は学校に行きます",
            "わたしはがっこうにいきます",
            7
        ));
        // Ordinary adverb+particle hiragana chain (`はとてもいい` = 6) must NOT
        // trip the threshold — this is the false-positive guard.
        assert!(!has_long_unconverted_run(
            "今日はとてもいい日",
            "きょうはとてもいいひ",
            7
        ));
        // Run must actually appear in the reading to count.
        assert!(!has_long_unconverted_run("あいうえお", "かきくけこ", 7));
    }

    #[test]
    fn test_is_pure_full_katakana() {
        assert!(is_pure_full_katakana("ア"));
        assert!(is_pure_full_katakana("アイウエオ"));
        assert!(is_pure_full_katakana("コーヒー")); // includes prolonged mark
        assert!(is_pure_full_katakana("ヴ"));

        assert!(!is_pure_full_katakana(""));
        assert!(!is_pure_full_katakana("あ")); // hiragana
        assert!(!is_pure_full_katakana("ｱ")); // half-width
        assert!(!is_pure_full_katakana("ア漢"));
        assert!(!is_pure_full_katakana("・")); // middle dot not a kana letter
    }

    #[test]
    fn test_hiragana_to_katakana() {
        assert_eq!(hiragana_to_katakana("あいうえお"), "アイウエオ");
        assert_eq!(hiragana_to_katakana("こんにちは"), "コンニチハ");
        assert_eq!(hiragana_to_katakana("きゃきゅきょ"), "キャキュキョ");
        assert_eq!(hiragana_to_katakana("がぎぐげご"), "ガギグゲゴ");
        assert_eq!(hiragana_to_katakana("ぱぴぷぺぽ"), "パピプペポ");

        // Mixed with non-hiragana should pass through
        assert_eq!(hiragana_to_katakana("abc123"), "abc123");
        assert_eq!(hiragana_to_katakana("あいうabc"), "アイウabc");
    }

    #[test]
    fn test_katakana_to_hiragana() {
        assert_eq!(katakana_to_hiragana("アイウエオ"), "あいうえお");
        assert_eq!(katakana_to_hiragana("コンニチハ"), "こんにちは");
        assert_eq!(katakana_to_hiragana("キャキュキョ"), "きゃきゅきょ");
    }

    #[test]
    fn test_round_trip() {
        let original = "こんにちは";
        let katakana = hiragana_to_katakana(original);
        let back = katakana_to_hiragana(&katakana);
        assert_eq!(original, back);
    }

    #[test]
    fn test_katakana_to_half_width() {
        assert_eq!(katakana_to_half_width("アイウエオ"), "ｱｲｳｴｵ");
        assert_eq!(katakana_to_half_width("カキクケコ"), "ｶｷｸｹｺ");
        assert_eq!(katakana_to_half_width("ガッコウ"), "ｶﾞｯｺｳ");
        assert_eq!(katakana_to_half_width("パピプペポ"), "ﾊﾟﾋﾟﾌﾟﾍﾟﾎﾟ");
        assert_eq!(katakana_to_half_width("キャキュキョ"), "ｷｬｷｭｷｮ");
        assert_eq!(katakana_to_half_width("コーヒー"), "ｺｰﾋｰ");
        assert_eq!(katakana_to_half_width("ヴ"), "ｳﾞ");
        // Punctuation
        assert_eq!(katakana_to_half_width("「アイウ」"), "｢ｱｲｳ｣");
        // Pass through non-katakana
        assert_eq!(katakana_to_half_width("abc"), "abc");
        assert_eq!(katakana_to_half_width("漢字"), "漢字");
    }

    #[test]
    fn test_hiragana_to_half_katakana() {
        assert_eq!(hiragana_to_half_katakana("あ"), "ｱ");
        assert_eq!(hiragana_to_half_katakana("がっこう"), "ｶﾞｯｺｳ");
        assert_eq!(hiragana_to_half_katakana("ぱぴぷぺぽ"), "ﾊﾟﾋﾟﾌﾟﾍﾟﾎﾟ");
    }

    #[test]
    fn test_ascii_to_fullwidth_char() {
        // Digits
        assert_eq!(ascii_to_fullwidth_char('0'), '０');
        assert_eq!(ascii_to_fullwidth_char('9'), '９');
        // Uppercase letters
        assert_eq!(ascii_to_fullwidth_char('A'), 'Ａ');
        assert_eq!(ascii_to_fullwidth_char('Z'), 'Ｚ');
        // Lowercase letters
        assert_eq!(ascii_to_fullwidth_char('a'), 'ａ');
        assert_eq!(ascii_to_fullwidth_char('z'), 'ｚ');
        // Pass-through for non-ASCII-alphanumerics
        assert_eq!(ascii_to_fullwidth_char(' '), ' ');
        assert_eq!(ascii_to_fullwidth_char('!'), '!');
        assert_eq!(ascii_to_fullwidth_char('あ'), 'あ');
        assert_eq!(ascii_to_fullwidth_char('Ａ'), 'Ａ');
    }

    #[test]
    fn test_fullwidth_to_ascii_char() {
        // Digits
        assert_eq!(fullwidth_to_ascii_char('０'), '0');
        assert_eq!(fullwidth_to_ascii_char('９'), '9');
        // Uppercase letters
        assert_eq!(fullwidth_to_ascii_char('Ａ'), 'A');
        assert_eq!(fullwidth_to_ascii_char('Ｚ'), 'Z');
        // Lowercase letters
        assert_eq!(fullwidth_to_ascii_char('ａ'), 'a');
        assert_eq!(fullwidth_to_ascii_char('ｚ'), 'z');
        // Pass-through
        assert_eq!(fullwidth_to_ascii_char('a'), 'a');
        assert_eq!(fullwidth_to_ascii_char('あ'), 'あ');
        assert_eq!(fullwidth_to_ascii_char('！'), '！'); // not part of ASCII alphanumerics
    }

    #[test]
    fn test_normalize_nfkc() {
        // Full-width ASCII should be converted to half-width
        assert_eq!(normalize_nfkc("（）"), "()");
        assert_eq!(normalize_nfkc("！？"), "!?");
        assert_eq!(normalize_nfkc("Ａｂｃ"), "Abc");
        assert_eq!(normalize_nfkc("０１２３"), "0123");

        // Full-width punctuation
        assert_eq!(normalize_nfkc("、。"), "、。"); // These are NOT full-width ASCII
        assert_eq!(normalize_nfkc("「」"), "「」"); // Japanese brackets preserved

        // Hiragana, Katakana, Kanji should be preserved
        assert_eq!(normalize_nfkc("あいうえお"), "あいうえお");
        assert_eq!(normalize_nfkc("アイウエオ"), "アイウエオ");
        assert_eq!(normalize_nfkc("漢字"), "漢字");

        // Mixed text
        assert_eq!(normalize_nfkc("（カッコ）テスト！"), "(カッコ)テスト!");

        // Special jinen tokens (Private Use Area U+EE00-U+EE02) should be preserved
        assert_eq!(normalize_nfkc("\u{ee00}"), "\u{ee00}");
        assert_eq!(normalize_nfkc("\u{ee01}"), "\u{ee01}");
        assert_eq!(normalize_nfkc("\u{ee02}"), "\u{ee02}");
        assert_eq!(
            normalize_nfkc("\u{ee02}context\u{ee00}input\u{ee01}"),
            "\u{ee02}context\u{ee00}input\u{ee01}"
        );
    }

    #[test]
    fn test_ascii_fullwidth_round_trip() {
        assert_eq!(ascii_to_fullwidth("Ab 1-~"), "Ａｂ　１－～");
        assert_eq!(fullwidth_to_ascii("Ａｂ　１－～"), "Ab 1-~");
        assert_eq!(ascii_to_fullwidth("あ"), "あ");
    }
}
