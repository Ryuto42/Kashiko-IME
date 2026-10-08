//! Telling English words apart from romaji in one stream of keystrokes.
//!
//! `kyouhagoogledekensaku` is typed without switching modes, so the romaji
//! converter alone turns `google` into `ごおｇぇ`. This module finds the English
//! words in the keystrokes so the caller can keep them in Latin script:
//! `きょうは` + `google` + `でけんさく`.
//!
//! The split is a shortest-path search over the keystrokes. A stretch read as
//! romaji costs nothing when it converts cleanly, and a lot for every letter
//! the romaji rules can't consume; a known English word costs a little, or
//! less than nothing for a product name (`amazon` reads as あまぞん, but nobody
//! means that). So English only wins where romaji fails or where the word is
//! unmistakably English, and Japanese stays Japanese by default — `make`,
//! `name`, `time` read as まけ・なめ・ちめ unless the context is English.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::romaji::RomajiConverter;

const ENGLISH_TSV: &str = include_str!("../data/english.tsv");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WordKind {
    /// Product or proper name: English even where it reads as romaji.
    Proper,
    /// Ordinary word: English only where romaji fails or context says so.
    Common,
}

struct Lexicon {
    words: HashMap<String, WordKind>,
    /// Usual spelling where it isn't the lowercase word (`github` → `GitHub`).
    spellings: HashMap<String, String>,
    /// Same words, sorted, for "is this the start of a word" while typing.
    sorted: Vec<String>,
}

fn lexicon() -> &'static Lexicon {
    static LEXICON: OnceLock<Lexicon> = OnceLock::new();
    LEXICON.get_or_init(|| {
        let mut words = HashMap::new();
        let mut spellings = HashMap::new();
        for line in ENGLISH_TSV.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut cols = line.split('\t');
            let (Some(word), Some(kind)) = (cols.next(), cols.next()) else {
                continue;
            };
            let word = word.to_ascii_lowercase();
            let kind = if kind.trim() == "p" { WordKind::Proper } else { WordKind::Common };
            if let Some(spelling) = cols.next().map(str::trim).filter(|s| !s.is_empty()) {
                spellings.insert(word.clone(), spelling.to_string());
            }
            words.insert(word, kind);
        }
        for english in crate::loanword::all_english() {
            if english.chars().all(|c| c.is_ascii_alphabetic()) {
                words
                    .entry(english.to_ascii_lowercase())
                    .or_insert(WordKind::Common);
            }
        }
        let mut sorted: Vec<String> = words.keys().cloned().collect();
        sorted.sort();
        Lexicon { words, spellings, sorted }
    })
}

impl Lexicon {
    fn starts_word(&self, prefix: &str) -> bool {
        let i = self.sorted.partition_point(|w| w.as_str() < prefix);
        self.sorted.get(i).is_some_and(|w| w.starts_with(prefix))
    }
}

/// One stretch of the keystrokes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment {
    /// Romaji, to be converted to kana.
    Japanese(String),
    /// An English word, kept as typed.
    English(String),
}

/// What the user chose for a word before, overriding the lexicon.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Learned {
    English,
    Japanese,
}

/// Inputs to [`split`] besides the keystrokes.
pub struct Options<'a> {
    /// The text before the cursor is English (`I want `), so ordinary English
    /// words win over their romaji reading.
    pub english_context: bool,
    /// The user's earlier choice for a word (lowercase), if any.
    pub learned: &'a dyn Fn(&str) -> Option<Learned>,
    /// Whether a kana reading is a Japanese word (a dictionary entry). An
    /// ordinary English word typed on its own whose romaji reading means
    /// nothing (`issue` → いっすえ) is English.
    pub is_japanese_word: &'a dyn Fn(&str) -> bool,
}

/// Cost of one letter the romaji rules could not consume.
const STRANDED: i32 = 10;
/// Cost of a small-kana spelling (`l`/`x` prefixes): Japanese text rarely
/// needs one, English words (`hello`, `apple`) produce them all the time.
const SMALL_KANA_SPELLING: i32 = 2;
const PROPER_WORD: i32 = -1;
const COMMON_WORD: i32 = 1;
const COMMON_WORD_IN_ENGLISH: i32 = -1;
const LEARNED_WORD: i32 = -3;
/// Shorter ordinary words (`to`, `no`, `is`) collide with particles; they are
/// only English inside English text.
const MIN_COMMON_LEN: usize = 3;
const MAX_WORD_LEN: usize = 24;
/// Inside a sentence, an ordinary word whose romaji reading is no Japanese
/// word (`issue` → いっすえ) is English from this length on; shorter ones
/// (`time` → ちめ) turn up inside real romaji too often.
const MEANINGLESS_READING_MIN_LEN: usize = 5;

/// Split `raw` into romaji and English words. The last segment may be a word
/// still being typed. Returns a single `Japanese` segment when there is no
/// English in it.
pub fn split(raw: &str, opts: &Options) -> Vec<Segment> {
    let chars: Vec<char> = raw.chars().collect();
    let n = chars.len();
    if n == 0 {
        return Vec::new();
    }
    let mut conv = RomajiConverter::new();
    let lower: Vec<char> = chars.iter().map(|c| c.to_ascii_lowercase()).collect();

    // Positions where the plain romaji reading can be cut without changing it:
    // an English word has to start and end on one, or `tsugitsugi` would yield
    // `つgitすぎ`.
    let whole = kana_of(&mut conv, &chars, true).0;
    let boundaries: Vec<bool> = (0..=n)
        .map(|p| {
            p == 0
                || p == n
                || format!(
                    "{}{}",
                    kana_of(&mut conv, &chars[..p], true).0,
                    kana_of(&mut conv, &chars[p..], true).0
                ) == whole
        })
        .collect();

    let lex = lexicon();
    let mut english: Vec<(usize, usize, i32)> = Vec::new();
    for i in (0..n).filter(|&i| boundaries[i]) {
        for j in i + 2..=n.min(i + MAX_WORD_LEN) {
            if !lower[j - 1].is_ascii_alphabetic() {
                break;
            }
            // `pythonnobug`: romaji reads the `nn` as ん, but a word ending
            // in n followed by n + vowel is the word and the next syllable.
            let n_then_n = j < n && lower[j - 1] == 'n' && lower[j] == 'n';
            if !boundaries[j] && !n_then_n {
                continue;
            }
            let word: String = lower[i..j].iter().collect();
            let cost = match (opts.learned)(&word) {
                Some(Learned::Japanese) => continue,
                // Short words collide with particles (`in` inside `kaiinkai`).
                Some(Learned::English) if word.len() >= MIN_COMMON_LEN || opts.english_context => {
                    LEARNED_WORD
                }
                Some(Learned::English) => continue,
                None => match lex.words.get(&word) {
                    Some(WordKind::Proper) => PROPER_WORD,
                    Some(WordKind::Common) if opts.english_context => COMMON_WORD_IN_ENGLISH,
                    // Its reading must end on a whole syllable: `tag` (たg) is
                    // more likely the start of `taga` than a word.
                    Some(WordKind::Common)
                        if ((i == 0 && j == n) || word.len() >= MEANINGLESS_READING_MIN_LEN)
                            && kana_of(&mut conv, &chars[i..j], false).1.is_empty()
                            && !(opts.is_japanese_word)(&kana_of(&mut conv, &chars[i..j], true).0) =>
                    {
                        COMMON_WORD_IN_ENGLISH
                    }
                    Some(WordKind::Common) if word.len() >= MIN_COMMON_LEN => COMMON_WORD,
                    // A word still being typed at the end (`googl`).
                    None if j == n
                        && word.len() >= MIN_COMMON_LEN
                        && lex.starts_word(&word) =>
                    {
                        COMMON_WORD
                    }
                    _ => continue,
                },
            };
            english.push((i, j, cost));
        }
    }
    if english.is_empty() {
        return vec![Segment::Japanese(raw.to_string())];
    }

    let mut points: Vec<usize> = vec![0, n];
    for &(i, j, _) in &english {
        points.push(i);
        points.push(j);
    }
    points.sort_unstable();
    points.dedup();

    // best[p] = (cost, English words used, previous point, was English)
    let mut best: HashMap<usize, (i32, usize, usize, bool)> = HashMap::new();
    best.insert(0, (0, 0, 0, false));
    for &from in &points {
        let Some(&(cost, words, _, _)) = best.get(&from) else {
            continue;
        };
        let mut relax = |to: usize, step: i32, is_english: bool| {
            let cand = (cost + step, words + is_english as usize, from, is_english);
            let better = best
                .get(&to)
                .is_none_or(|cur| (cand.0, cand.1) < (cur.0, cur.1));
            if better {
                best.insert(to, cand);
            }
        };
        for &to in points.iter().filter(|&&p| p > from) {
            relax(to, romaji_cost(&mut conv, &chars[from..to], to == n), false);
        }
        for &(_, to, step) in english.iter().filter(|e| e.0 == from) {
            relax(to, step, true);
        }
    }

    let mut out: Vec<Segment> = Vec::new();
    let mut at = n;
    while at > 0 {
        let (_, _, from, is_english) = best[&at];
        let text: String = chars[from..at].iter().collect();
        let seg = if is_english { Segment::English(text) } else { Segment::Japanese(text) };
        match (out.last_mut(), seg) {
            (Some(Segment::Japanese(next)), Segment::Japanese(text)) => next.insert_str(0, &text),
            (_, seg) => out.push(seg),
        }
        at = from;
    }
    out.reverse();
    out
}

/// Romaji conversion of `chars`: `(kana, still pending)`. With `flush`, the
/// pending tail is converted too (a trailing `n` becomes ん).
fn kana_of(conv: &mut RomajiConverter, chars: &[char], flush: bool) -> (String, String) {
    conv.reset();
    for &c in chars {
        conv.push(c);
    }
    if flush {
        conv.flush();
    }
    (conv.output().to_string(), conv.buffer().to_string())
}

fn romaji_cost(conv: &mut RomajiConverter, chars: &[char], at_end: bool) -> i32 {
    // At the very end, letters still pending are an unfinished syllable, not a
    // failure.
    let (kana, _) = kana_of(conv, chars, !at_end);
    let stranded = kana.chars().filter(|c| c.is_ascii_alphabetic()).count() as i32;
    let small = chars
        .iter()
        .filter(|c| matches!(c.to_ascii_lowercase(), 'l' | 'x' | 'q'))
        .count() as i32;
    stranded * STRANDED + small * SMALL_KANA_SPELLING
}

/// The display text for `segments`, and the romaji still pending at its end.
///
/// An English word typed in lowercase takes its usual spelling: the one the
/// user committed before (`learned_spelling`, by lowercase word), else the
/// word table's (`github` → `GitHub`). One typed with capitals is kept as is.
pub fn render(
    segments: &[Segment],
    learned_spelling: &dyn Fn(&str) -> Option<String>,
) -> (String, String) {
    let mut conv = RomajiConverter::new();
    let mut text = String::new();
    let mut pending = String::new();
    for (i, seg) in segments.iter().enumerate() {
        match seg {
            Segment::English(word) if word.chars().any(|c| c.is_ascii_uppercase()) => {
                text.push_str(word)
            }
            Segment::English(word) => {
                let spelling = learned_spelling(word)
                    .or_else(|| lexicon().spellings.get(word.as_str()).cloned());
                text.push_str(spelling.as_deref().unwrap_or(word));
            }
            Segment::Japanese(romaji) => {
                let chars: Vec<char> = romaji.chars().collect();
                let last = i + 1 == segments.len();
                let (kana, rest) = kana_of(&mut conv, &chars, !last);
                text.push_str(&kana);
                if last {
                    pending = rest;
                }
            }
        }
    }
    (text, pending)
}

/// The whole of `raw` read as romaji (flushed), as if no word were English.
pub fn all_kana(raw: &str) -> String {
    thread_local! {
        static CONV: std::cell::RefCell<RomajiConverter> = std::cell::RefCell::new(RomajiConverter::new());
    }
    let chars: Vec<char> = raw.chars().collect();
    CONV.with(|conv| kana_of(&mut conv.borrow_mut(), &chars, true).0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn none(_: &str) -> Option<Learned> {
        None
    }

    fn words(kana: &str) -> bool {
        ["まけ", "なめ", "さけ"].contains(&kana)
    }

    fn opts(english_context: bool) -> Options<'static> {
        Options { english_context, learned: &none, is_japanese_word: &words }
    }

    fn shown(raw: &str) -> String {
        let segs = split(raw, &opts(false));
        let (text, pending) = render(&segs, &|_| None);
        format!("{text}{pending}")
    }

    #[test]
    fn english_inside_japanese() {
        assert_eq!(shown("kyouhagoogledekensaku"), "きょうはGoogleでけんさく");
        assert_eq!(shown("githubnoerrorwomiru"), "GitHubのerrorをみる");
        assert_eq!(shown("pythonnobu"), "Pythonのぶ");
    }

    #[test]
    fn plain_japanese_is_untouched() {
        for raw in ["watashihanihonjindesu", "tsugitsugi", "kaigiwoshimasu", "makenai", "namae", "kitte"] {
            assert_eq!(
                split(raw, &opts(false)),
                vec![Segment::Japanese(raw.to_string())],
                "{raw}"
            );
        }
    }

    #[test]
    fn product_names_win_over_their_romaji_reading() {
        assert_eq!(shown("amazondekaimono"), "Amazonでかいもの");
    }

    #[test]
    fn small_kana_spellings_point_at_english() {
        assert_eq!(shown("hello"), "hello");
    }

    #[test]
    fn a_word_being_typed_shows_in_latin() {
        assert_eq!(shown("kyouhagoogl"), "きょうはgoogl");
        // Still a valid romaji prefix: nothing to decide yet.
        assert_eq!(shown("kyouhagoog"), "きょうはごおg");
    }

    #[test]
    fn a_trailing_consonant_is_not_evidence() {
        // It may still become a syllable (`bugu…`, `nihasu`).
        assert_eq!(shown("pythonnobug"), "Pythonのぶg");
        assert_eq!(shown("nihas"), "にはs");
    }

    #[test]
    fn product_names_take_their_usual_spelling() {
        assert_eq!(shown("githubnoiphone"), "GitHubのiPhone");
        // A spelling the user committed wins, and capitals they typed stay.
        let segs = vec![Segment::English("zoom".into()), Segment::English("GITHUB".into())];
        let learned = |w: &str| (w == "zoom").then(|| "ZOOM".to_string());
        assert_eq!(render(&segs, &learned).0, "ZOOMGITHUB");
    }

    #[test]
    fn a_lone_word_whose_reading_means_nothing_is_english() {
        assert_eq!(shown("issue"), "issue");
        assert_eq!(shown("sake"), "さけ");
    }

    #[test]
    fn english_context_takes_ordinary_words() {
        let segs = split("make", &opts(true));
        assert_eq!(segs, vec![Segment::English("make".into())]);
        assert_eq!(shown("make"), "まけ");
    }

    #[test]
    fn learned_choices_override() {
        let japanese = |w: &str| (w == "google").then_some(Learned::Japanese);
        let segs = split(
            "google",
            &Options {
                english_context: false,
                learned: &japanese,
                is_japanese_word: &words,
            },
        );
        assert_eq!(segs, vec![Segment::Japanese("google".into())]);
        let english = |w: &str| (w == "sushi").then_some(Learned::English);
        let segs = split(
            "sushi",
            &Options {
                english_context: false,
                learned: &english,
                is_japanese_word: &words,
            },
        );
        assert_eq!(segs, vec![Segment::English("sushi".into())]);
    }

    /// Japanese sentences typed as romaji must never sprout English. Point
    /// `ROMAJI_CORPUS` at a file of romaji lines (one sentence each).
    #[test]
    #[ignore]
    fn japanese_sentences_stay_japanese() {
        let Ok(path) = std::env::var("ROMAJI_CORPUS") else {
            return;
        };
        let text = std::fs::read_to_string(path).unwrap();
        let all = |_: &str| true;
        let mut bad = Vec::new();
        for line in text.lines() {
            // Every prefix, as the split runs on each keystroke.
            for end in 1..=line.len() {
                let raw = &line[..end];
                let segs = split(
                    raw,
                    &Options { english_context: false, learned: &none, is_japanese_word: &all },
                );
                if segs.iter().any(|s| matches!(s, Segment::English(_))) {
                    bad.push(format!("{raw} -> {segs:?}"));
                    break;
                }
            }
        }
        println!("{} lines, {} with English", text.lines().count(), bad.len());
        assert!(bad.is_empty(), "{bad:#?}");
    }
}
