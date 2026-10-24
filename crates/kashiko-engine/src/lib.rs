pub mod counter;
pub mod dict;
pub mod kana;
pub mod kanji;
pub mod latin;
pub mod learning;
pub mod learning_file;
pub mod loanword;
pub mod mixed;
pub mod rewriter;
pub mod romaji;
pub mod segment;
pub mod typo;
pub mod variant;

pub use dict::{Candidate as DictCandidate, DictEntry, Dictionary, LookupResult};
pub use kana::{
    ascii_to_fullwidth, contains_kana, fullwidth_to_ascii, has_long_unconverted_run,
    hiragana_to_half_katakana, hiragana_to_katakana, is_all_kana, is_pure_full_katakana,
    is_pure_hiragana, katakana_to_hiragana, longest_passthrough_run, normalize_nfkc,
};
pub use kanji::{Backend, KanaKanjiConverter};
pub use learning::{LearningCache, has_unrenderable_char};
pub use rewriter::{
    AlphabetRewriter, EmojiRewriter, HalfWidthKatakanaRewriter, RewriteOutput, Rewriter,
    RewriterChain, SymbolRewriter, description as symbol_description,
};
pub use romaji::{BackspaceResult, ConversionEvent, RomajiConverter};
