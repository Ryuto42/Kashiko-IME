//! Learning cache for remembering user-selected conversion results.
//!
//! Records which surface forms the user chose for each reading, and
//! boosts those candidates on subsequent conversions. Persisted as a
//! simple TSV file (`reading\tsurface\tfrequency\tlast_access`).

use std::collections::HashMap;

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// A single learned conversion entry.
#[derive(Debug, Clone)]
pub struct LearningEntry {
    /// Surface form (e.g. "今日")
    pub surface: String,
    /// Number of times this surface was selected
    pub frequency: u32,
    /// Last selection time as Unix timestamp (seconds)
    pub last_access: u64,
}

/// In-memory cache of user learning data.
///
/// Keyed by reading (hiragana). Each reading maps to a list of surface
/// entries with frequency and recency metadata.
#[derive(Debug)]
pub struct LearningCache {
    entries: HashMap<String, Vec<LearningEntry>>,
    max_entries: usize,
    dirty: bool,
}

impl LearningCache {
    /// Default maximum number of total entries across all readings.
    pub const DEFAULT_MAX_ENTRIES: usize = 10_000;

    /// Surface forms with this many characters or more are never learned or
    /// shown. Long sentences committed once should not be memorized wholesale.
    pub const MAX_LEARN_SURFACE_CHARS: usize = 12;

    /// Frequency marking a *suppressed* entry — a candidate the user deleted
    /// with the row's ✕. Stored in the same map so it persists with the rest,
    /// but treated as a blocklist rather than a suggestion. 0 can never occur
    /// naturally: `record` starts at 1.
    pub const SUPPRESSED_FREQUENCY: u32 = 0;

    /// Length cap for all-latin surfaces (`InputMethodKit`). Higher than the
    /// Japanese cap: the limit exists to reject memorized sentences, and a
    /// long English identifier is still a single word.
    pub const MAX_LEARN_LATIN_CHARS: usize = 24;

    /// Create an empty cache with the given entry limit.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
            dirty: false,
        }
    }

    /// Record a user selection. Increments frequency and updates last_access.
    ///
    /// Long surfaces (>= `MAX_LEARN_SURFACE_CHARS`) and low-quality surfaces
    /// (unconverted / leftover romaji / repeated-char garbage) are ignored so
    /// that mistyped or unconverted sentences never pollute the cache.
    pub fn record(&mut self, reading: &str, surface: &str) {
        if !Self::is_learnable(reading, surface) {
            return;
        }
        let now = now_unix();
        let entries = self.entries.entry(reading.to_string()).or_default();

        if let Some(entry) = entries.iter_mut().find(|e| e.surface == surface) {
            entry.frequency += 1;
            entry.last_access = now;
        } else {
            entries.push(LearningEntry {
                surface: surface.to_string(),
                frequency: 1,
                last_access: now,
            });
        }
        self.dirty = true;
    }

    /// Whether a `(reading, surface)` pair is worth learning.
    ///
    /// Rejects: overly long surfaces, unconverted surfaces (equal to the
    /// reading or all-kana), surfaces with leftover ASCII romaji mixed into
    /// Japanese, and surfaces containing a run of 3+ identical characters
    /// (mistyped repeats). A *pure* latin word (`WordPress`) is accepted —
    /// see below.
    pub fn is_learnable(reading: &str, surface: &str) -> bool {
        if surface.is_empty() {
            return false;
        }
        // A surface that is entirely latin letters/digits is a real word the
        // user typed (`WordPress`, `iPhone`), not a conversion failure: the
        // romaji converter mangles those keystrokes into kana, so learning
        // them is the only way to offer the spelling back later. Recognized
        // here because the leftover-romaji and no-kanji rules below exist for
        // *Japanese* surfaces and would both reject it.
        // Spaces, hyphens and dots are part of the word here: the loanword table
        // offers `pull request`, `in-memory`, `one-time password`. Without them
        // a multi-word entry counted as Japanese with no kanji and was rejected
        // by the rule below — offered every time, learned never.
        let pure_latin = surface
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, ' ' | '-' | '.' | '_'))
            && surface.chars().any(|c| c.is_ascii_alphabetic());
        // Feature 1: length cap. Latin words get a longer allowance — the cap
        // targets runaway *sentences*, and `InputMethodKit` is one word.
        let max_chars = if pure_latin {
            Self::MAX_LEARN_LATIN_CHARS
        } else {
            Self::MAX_LEARN_SURFACE_CHARS
        };
        if surface.chars().count() >= max_chars {
            return false;
        }
        // Feature 3-2: ASCII mixed into Japanese is usually leftover romaji the
        // model never converted (`できないt消す`). But mixed script is also
        // legitimate — `私はLinuxが` is a documented conversion result — so
        // reject only when *every* latin run looks like a stray fragment: one
        // or two lowercase letters. A run with a capital or ≥3 letters is a
        // word, not a slip.
        if !pure_latin {
            let mut runs = surface.split(|c: char| !c.is_ascii_alphanumeric());
            let mut saw_run = false;
            let all_fragments = runs.all(|run| {
                if run.is_empty() {
                    return true;
                }
                saw_run = true;
                run.chars().count() <= 2 && run.chars().all(|c| c.is_ascii_lowercase())
            });
            if saw_run && all_fragments {
                return false;
            }
        }
        // Unrenderable characters (rare historical kana, stray model control
        // tokens) — see `has_unrenderable_char`.
        if has_unrenderable_char(surface) {
            return false;
        }
        // Feature 3-1: unconverted — identical to reading, or contains no kanji
        // despite being non-trivially long.
        if surface == reading {
            return false;
        }
        let has_kanji = surface.chars().any(is_kanji);
        if !pure_latin && !has_kanji && surface.chars().count() >= 6 {
            return false;
        }
        // Feature 3-3: 3+ consecutive identical characters (garbage repeats).
        let mut prev: Option<char> = None;
        let mut run = 1usize;
        for c in surface.chars() {
            if Some(c) == prev {
                run += 1;
                if run >= 3 {
                    return false;
                }
            } else {
                run = 1;
                prev = Some(c);
            }
        }
        true
    }

    /// Exact-match lookup: returns `(surface, score)` pairs sorted by score descending.
    pub fn lookup(&self, reading: &str) -> Vec<(String, f64)> {
        let now = now_unix();
        let Some(entries) = self.entries.get(reading) else {
            return Vec::new();
        };
        let mut scored: Vec<(String, f64)> = entries
            .iter()
            .filter(|e| e.frequency != Self::SUPPRESSED_FREQUENCY)
            .map(|e| (e.surface.clone(), score(e, now)))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored
    }

    /// Suppress `surface` for `reading` so it is never offered again, even when
    /// it comes from the model or a dictionary rather than learning.
    ///
    /// Recorded as a learning entry with a sentinel frequency, so it rides
    /// along with the existing persistence and needs no second file. Entries
    /// like this are filtered out of lookups and used as a blocklist instead.
    pub fn suppress(&mut self, reading: &str, surface: &str) {
        let entries = self.entries.entry(reading.to_string()).or_default();
        entries.retain(|e| e.surface != surface);
        entries.push(LearningEntry {
            surface: surface.to_string(),
            frequency: Self::SUPPRESSED_FREQUENCY,
            last_access: now_unix(),
        });
        self.dirty = true;
    }

    /// Take back one `record` of `(reading, surface)`: lower its frequency by
    /// one and drop the entry when that reaches zero. Suppressed entries are
    /// left alone.
    pub fn unrecord(&mut self, reading: &str, surface: &str) {
        let Some(entries) = self.entries.get_mut(reading) else {
            return;
        };
        let Some(pos) = entries
            .iter()
            .position(|e| e.surface == surface && e.frequency != Self::SUPPRESSED_FREQUENCY)
        else {
            return;
        };
        if entries[pos].frequency > 1 {
            entries[pos].frequency -= 1;
        } else {
            entries.remove(pos);
        }
        if entries.is_empty() {
            self.entries.remove(reading);
        }
        self.dirty = true;
    }

    /// Drop every entry, suppressions included.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.dirty = true;
    }

    /// Whether `surface` has been suppressed for `reading` (or for a reading
    /// that starts with it, matching how predictive lookup would surface it).
    pub fn is_suppressed(&self, reading: &str, surface: &str) -> bool {
        self.entries.iter().any(|(key, entries)| {
            (key == reading || key.starts_with(reading))
                && entries
                    .iter()
                    .any(|e| e.surface == surface && e.frequency == Self::SUPPRESSED_FREQUENCY)
        })
    }

    /// Forget a learned entry, so the candidate stops being suggested.
    ///
    /// Removes `surface` from `reading`'s entries and, since a surface picked
    /// up by predictive (prefix) lookup can be keyed under a longer reading
    /// than what was typed, from every reading that starts with `reading` too —
    /// otherwise deleting a suggestion would leave it reappearing from the
    /// prefix match. Returns whether anything was removed.
    pub fn forget(&mut self, reading: &str, surface: &str) -> bool {
        let mut removed = false;
        for (key, entries) in self.entries.iter_mut() {
            if key == reading || key.starts_with(reading) {
                let before = entries.len();
                entries.retain(|e| e.surface != surface);
                removed |= entries.len() != before;
            }
        }
        self.entries.retain(|_, entries| !entries.is_empty());
        self.dirty |= removed;
        removed
    }

    /// Whether any learned entry has exactly this surface form, regardless of
    /// reading. Used to recognize a latin word the user has committed before
    /// (e.g. `WordPress`) so it can be offered again even though the romaji
    /// converter turned the keystrokes into kana.
    pub fn has_surface(&self, surface: &str) -> bool {
        self.entries
            .values()
            .any(|entries| entries.iter().any(|e| e.surface == surface))
    }

    /// Latin words the user committed for their own keystrokes (lowercased,
    /// mapped to the spelling they committed — reading `あぴ`, surface `api`),
    /// and those they rejected there (✕ or chose the kana reading instead).
    pub fn latin_word_choices(
        &self,
    ) -> (
        std::collections::HashMap<String, String>,
        std::collections::HashSet<String>,
    ) {
        let mut chosen = std::collections::HashMap::new();
        let mut rejected = std::collections::HashSet::new();
        for (reading, entries) in &self.entries {
            for e in entries {
                if e.surface.is_empty() || !e.surface.chars().all(|c| c.is_ascii_alphabetic()) {
                    continue;
                }
                let word = e.surface.to_ascii_lowercase();
                // Only a choice made for the word's own keystrokes counts.
                // `You` picked as the conversion of ゆー says nothing about
                // `you` typed as romaji, which is よう.
                if *reading != crate::mixed::all_kana(&word) {
                    continue;
                }
                if e.frequency == Self::SUPPRESSED_FREQUENCY {
                    rejected.insert(word);
                } else {
                    chosen.insert(word, e.surface.clone());
                }
            }
        }
        (chosen, rejected)
    }

    /// Highest-scoring `(surface, frequency)` for an exact reading match, if any.
    ///
    /// Used by live conversion to decide whether a learned entry is confident
    /// enough (see `frequency`) to override the model's own guess.
    pub fn top_entry(&self, reading: &str) -> Option<(String, u32)> {
        let now = now_unix();
        self.entries
            .get(reading)?
            .iter()
            .max_by(|a, b| score(a, now).total_cmp(&score(b, now)))
            .map(|e| (e.surface.clone(), e.frequency))
    }

    /// Prefix-match lookup: returns `(reading, surface, score)` triples
    /// for all readings that start with `prefix`, sorted by score descending.
    pub fn prefix_lookup(&self, prefix: &str) -> Vec<(String, String, f64)> {
        let now = now_unix();
        let mut results: Vec<(String, String, f64)> = Vec::new();
        for (reading, entries) in &self.entries {
            if reading.starts_with(prefix) {
                for entry in entries
                    .iter()
                    .filter(|e| e.frequency != Self::SUPPRESSED_FREQUENCY)
                {
                    results.push((reading.clone(), entry.surface.clone(), score(entry, now)));
                }
            }
        }
        results.sort_by(|a, b| b.2.total_cmp(&a.2));
        results
    }

    /// Load a learning cache from disk.
    ///
    /// Format: `reading\tsurface\tfrequency\tlast_access`, lines starting with
    /// `#` are comments. The file is encrypted (see [`crate::learning_file`]);
    /// a plaintext one written before encryption existed still loads, and is
    /// rewritten encrypted on the next save.
    pub fn load(path: &Path, max_entries: usize) -> anyhow::Result<Self> {
        let text = crate::learning_file::read_maybe_encrypted(path)?;
        let mut cache = Self::new(max_entries);
        let encrypted_on_disk = text.starts_with(Self::ENCRYPTED_HEADER);

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() < 4 {
                continue;
            }
            let reading = parts[0];
            let surface = parts[1];
            let frequency: u32 = match parts[2].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let last_access: u64 = match parts[3].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };

            // Drop long / low-quality entries left over from before this
            // filter existed. Skipped lines disappear on the next save().
            if !Self::is_learnable(reading, surface) {
                continue;
            }

            cache
                .entries
                .entry(reading.to_string())
                .or_default()
                .push(LearningEntry {
                    surface: surface.to_string(),
                    frequency,
                    last_access,
                });
        }

        // Not dirty — just loaded from disk. A plaintext file is the exception:
        // it has to be rewritten encrypted, and only a dirty cache gets saved.
        cache.dirty = !encrypted_on_disk;
        Ok(cache)
    }

    /// Header the encrypted form carries once decrypted, used to tell a cache
    /// that was already encrypted from a plaintext one being migrated.
    const ENCRYPTED_HEADER: &'static str = "# kashiko learning cache v2";

    /// Save the cache, evicting low-score entries if over capacity.
    ///
    /// Written encrypted: this is a record of what the user has typed and does
    /// not belong on disk as readable text. See [`crate::learning_file`] for
    /// what that protects against and what it does not.
    pub fn save(&mut self, path: &Path) -> anyhow::Result<()> {
        self.evict();

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let mut out = String::new();
        out.push_str(Self::ENCRYPTED_HEADER);
        out.push('\n');

        // Sort readings for deterministic output
        let mut readings: Vec<&String> = self.entries.keys().collect();
        readings.sort();

        for reading in readings {
            if let Some(entries) = self.entries.get(reading) {
                for entry in entries {
                    use std::fmt::Write as _;
                    writeln!(
                        out,
                        "{}\t{}\t{}\t{}",
                        reading, entry.surface, entry.frequency, entry.last_access
                    )?;
                }
            }
        }

        crate::learning_file::write_encrypted(path, &out)?;
        self.dirty = false;
        Ok(())
    }

    /// Whether there are unsaved changes.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Total number of (reading, surface) pairs across all readings.
    pub fn entry_count(&self) -> usize {
        self.entries.values().map(|v| v.len()).sum()
    }

    /// Evict lowest-score entries until total count is within `max_entries`.
    fn evict(&mut self) {
        let total = self.entry_count();
        if total <= self.max_entries {
            return;
        }

        let now = now_unix();
        // Collect all entries with their (reading, index, score)
        let mut all: Vec<(String, usize, f64)> = Vec::with_capacity(total);
        for (reading, entries) in &self.entries {
            for (i, entry) in entries.iter().enumerate() {
                all.push((reading.clone(), i, score(entry, now)));
            }
        }
        // Sort by score ascending (lowest first = eviction candidates)
        all.sort_by(|a, b| a.2.total_cmp(&b.2));

        let to_remove = total - self.max_entries;
        // Collect indices to remove, grouped by reading
        let mut remove_set: HashMap<String, Vec<usize>> = HashMap::new();
        for &(ref reading, idx, _) in all.iter().take(to_remove) {
            remove_set.entry(reading.clone()).or_default().push(idx);
        }

        // Remove entries in reverse index order to preserve indices
        for (reading, indices) in &mut remove_set {
            indices.sort_unstable();
            indices.reverse();
            if let Some(entries) = self.entries.get_mut(reading) {
                for &idx in indices.iter() {
                    if idx < entries.len() {
                        entries.remove(idx);
                    }
                }
                if entries.is_empty() {
                    self.entries.remove(reading);
                }
            }
        }
    }
}

/// Compute a candidate score: recency-weighted with frequency bonus.
///
/// Inspired by mozc's UserHistoryPredictor: recent selections rank higher,
/// with a logarithmic frequency term to reward repeated use.
fn score(entry: &LearningEntry, now: u64) -> f64 {
    let age_days = if now > entry.last_access {
        (now - entry.last_access) / 86400
    } else {
        0
    };
    let recency = 1.0 / (1.0 + age_days as f64);
    let freq = (entry.frequency as f64).ln_1p();
    recency * 10.0 + freq
}

/// Whether a character is a CJK ideograph (kanji).
fn is_kanji(c: char) -> bool {
    matches!(c,
        '\u{3400}'..='\u{4DBF}'   // CJK Ext A
        | '\u{4E00}'..='\u{9FFF}' // CJK Unified
        | '\u{F900}'..='\u{FAFF}' // CJK Compatibility Ideographs
        | '\u{20000}'..='\u{2A6DF}' // CJK Ext B
    )
}

/// Whether `text` contains a character virtually no system font can render,
/// so learning it would only ever surface as a "missing glyph" box.
///
/// Covers two sources of garbage surfaces:
/// - Kana Supplement / Kana Extended-A / Small Kana Extension
///   (U+1B000–U+1B16F): historical hentaigana variants that appear in the
///   Mozc-derived system dictionary but have no glyph in any common font.
///   These can end up learned if a user accidentally selects one.
/// - Private Use Area (U+E000–U+F8FF, and the supplementary PUA-A/B planes):
///   covers the jinen model's internal control tokens
///   (`CONTEXT_TOKEN`/`INPUT_START_TOKEN`/`OUTPUT_START_TOKEN`, U+EE00–EE02;
///   see `kashiko-engine/README.md`) in case one ever leaks into decoded
///   model output instead of being stripped.
pub fn has_unrenderable_char(text: &str) -> bool {
    text.chars().any(|c| {
        matches!(c,
            '\u{1B000}'..='\u{1B16F}' // Kana Supplement / Kana Extended-A
            | '\u{E000}'..='\u{F8FF}' // Private Use Area
            | '\u{F0000}'..='\u{FFFFD}' // Supplementary PUA-A
            | '\u{100000}'..='\u{10FFFD}' // Supplementary PUA-B
        )
    })
}

/// Current time as Unix timestamp in seconds.
fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_record_and_lookup() {
        let mut cache = LearningCache::new(100);

        cache.record("きょう", "今日");
        cache.record("きょう", "京");
        cache.record("きょう", "今日"); // frequency bump

        let results = cache.lookup("きょう");
        assert_eq!(results.len(), 2);
        // "今日" should have higher score (frequency 2 vs 1)
        assert_eq!(results[0].0, "今日");
        assert_eq!(results[1].0, "京");
    }

    #[test]
    fn unrecord_lowers_frequency_then_removes() {
        let mut cache = LearningCache::new(100);
        cache.record("きょう", "今日");
        cache.record("きょう", "今日");
        cache.unrecord("きょう", "今日");
        assert_eq!(cache.top_entry("きょう"), Some(("今日".to_string(), 1)));
        cache.unrecord("きょう", "今日");
        assert_eq!(cache.entry_count(), 0);
        assert!(cache.lookup("きょう").is_empty());
    }

    #[test]
    fn unrecord_leaves_suppressed_and_unknown_entries() {
        let mut cache = LearningCache::new(100);
        cache.suppress("きょう", "凶");
        cache.unrecord("きょう", "凶");
        cache.unrecord("きょう", "今日");
        cache.unrecord("ほか", "他");
        assert!(cache.is_suppressed("きょう", "凶"));
        assert_eq!(cache.entry_count(), 1);
    }

    #[test]
    fn clear_empties_the_cache() {
        let mut cache = LearningCache::new(100);
        cache.record("きょう", "今日");
        cache.suppress("きょう", "凶");
        cache.clear();
        assert_eq!(cache.entry_count(), 0);
        assert!(cache.is_dirty());
    }

    #[test]
    fn pure_latin_surface_is_learnable() {
        // A latin word the romaji converter mangled into kana must be
        // learnable — it's the only way to offer the spelling back later.
        assert!(LearningCache::is_learnable("wおrdpレッ", "WordPress"));
        assert!(LearningCache::is_learnable("いphおね", "iPhone"));
        // A long latin identifier is one word, so it clears the Japanese cap.
        assert!(LearningCache::is_learnable("いんぷっと", "InputMethodKit"));
        // Stray lowercase fragments in Japanese are leftover romaji → rejected.
        assert!(!LearningCache::is_learnable("できないt", "できないt消す"));
        // Legitimate mixed script (a documented conversion result) is kept.
        assert!(LearningCache::is_learnable(
            "わたしはlinuxが",
            "私はLinuxが"
        ));
        // And a pure-latin surface still obeys its own (longer) cap.
        assert!(!LearningCache::is_learnable(
            "x",
            "Supercalifragilisticexpialidocious"
        ));
    }

    #[test]
    fn latin_surface_round_trips_through_has_surface() {
        let mut cache = LearningCache::new(100);
        cache.record("wおrdpレッ", "WordPress");
        assert!(cache.has_surface("WordPress"));
        assert!(!cache.has_surface("Wordpress"));
    }

    #[test]
    fn test_has_unrenderable_char() {
        // Kana Supplement hentaigana variant (the actual corrupted entry found
        // in a real learning.tsv for reading "を").
        assert!(has_unrenderable_char("\u{1B118}"));
        // jinen model's internal control tokens (Private Use Area).
        assert!(has_unrenderable_char("\u{EE00}"));
        assert!(has_unrenderable_char("\u{EE01}"));
        assert!(has_unrenderable_char("\u{EE02}"));
        // Normal kanji/kana/symbols are unaffected.
        assert!(!has_unrenderable_char("今日"));
        assert!(!has_unrenderable_char("あいう"));
        assert!(!has_unrenderable_char("…"));
    }

    #[test]
    fn test_record_rejects_unrenderable_char() {
        let mut cache = LearningCache::new(100);
        cache.record("を", "\u{1B118}");
        assert!(cache.lookup("を").is_empty());
    }

    #[test]
    fn test_record_rejects_long_and_low_quality() {
        let mut cache = LearningCache::new(100);

        // 12+ char surface: not learned.
        let long = "きょうのてんきははれです"; // 12 chars
        cache.record("きょうのてんきははれです", long);
        assert!(cache.lookup("きょうのてんきははれです").is_empty());

        // Leftover romaji: not learned.
        cache.record("きょうk", "今日k");
        assert!(cache.lookup("きょうk").is_empty());

        // Repeated-char garbage: not learned.
        cache.record("ははは", "ははは");
        assert!(cache.lookup("ははは").is_empty());

        // Unconverted (surface == reading): not learned.
        cache.record("あした", "あした");
        assert!(cache.lookup("あした").is_empty());

        // Normal conversion: learned.
        cache.record("きょう", "今日");
        assert_eq!(cache.lookup("きょう").len(), 1);
    }

    #[test]
    fn test_load_drops_low_quality_entries() {
        use std::io::Write as _;
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "# kashiko learning cache v1").unwrap();
        writeln!(file, "きょう\t今日\t3\t1700000000").unwrap();
        writeln!(file, "ながいよみ\tきょうのてんきははれです\t1\t1700000000").unwrap();
        writeln!(file, "ろーまじ\t今日k\t1\t1700000000").unwrap();
        writeln!(file, "を\t\u{1B118}\t2\t1783581429").unwrap();
        file.flush().unwrap();

        let cache = LearningCache::load(file.path(), 100).unwrap();
        assert_eq!(cache.entry_count(), 1);
        assert_eq!(cache.lookup("きょう").len(), 1);
        assert!(cache.lookup("ながいよみ").is_empty());
        assert!(cache.lookup("ろーまじ").is_empty());
        assert!(cache.lookup("を").is_empty());
    }

    #[test]
    fn test_lookup_empty() {
        let cache = LearningCache::new(100);
        let results = cache.lookup("きょう");
        assert!(results.is_empty());
    }

    #[test]
    fn test_prefix_lookup() {
        let mut cache = LearningCache::new(100);
        cache.record("きょう", "今日");
        cache.record("きょうと", "京都");
        cache.record("あした", "明日");

        let results = cache.prefix_lookup("きょう");
        assert_eq!(results.len(), 2);
        // Both "きょう" and "きょうと" should match
        let readings: Vec<&str> = results.iter().map(|(r, _, _)| r.as_str()).collect();
        assert!(readings.contains(&"きょう"));
        assert!(readings.contains(&"きょうと"));
    }

    #[test]
    fn test_prefix_lookup_no_match() {
        let mut cache = LearningCache::new(100);
        cache.record("きょう", "今日");
        let results = cache.prefix_lookup("あ");
        assert!(results.is_empty());
    }

    #[test]
    fn test_save_and_load() {
        let mut cache = LearningCache::new(100);
        cache.record("きょう", "今日");
        cache.record("きょう", "今日");
        cache.record("きょう", "京");
        cache.record("あした", "明日");

        let file = NamedTempFile::new().unwrap();
        let path = file.path().to_path_buf();

        cache.save(&path).unwrap();
        assert!(!cache.is_dirty());

        let loaded = LearningCache::load(&path, 100).unwrap();
        assert!(!loaded.is_dirty());
        assert_eq!(loaded.entry_count(), 3);

        let results = loaded.lookup("きょう");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "今日"); // frequency 2
    }

    #[test]
    fn test_dirty_flag() {
        let mut cache = LearningCache::new(100);
        assert!(!cache.is_dirty());

        cache.record("きょう", "今日");
        assert!(cache.is_dirty());

        let file = NamedTempFile::new().unwrap();
        cache.save(file.path()).unwrap();
        assert!(!cache.is_dirty());
    }

    #[test]
    fn test_eviction() {
        let mut cache = LearningCache::new(3);

        // Add 5 entries
        cache.record("a", "A");
        cache.record("b", "B");
        cache.record("c", "C");
        cache.record("d", "D");
        cache.record("e", "E");

        // Boost some to give them higher scores
        cache.record("a", "A");
        cache.record("a", "A");
        cache.record("c", "C");

        let file = NamedTempFile::new().unwrap();
        cache.save(file.path()).unwrap();

        // After eviction, should be at most 3 entries
        assert!(cache.entry_count() <= 3);
    }

    #[test]
    fn test_score_recency() {
        let now = now_unix();
        let recent = LearningEntry {
            surface: "A".to_string(),
            frequency: 1,
            last_access: now,
        };
        let old = LearningEntry {
            surface: "B".to_string(),
            frequency: 1,
            last_access: now.saturating_sub(30 * 86400), // 30 days ago
        };
        assert!(score(&recent, now) > score(&old, now));
    }

    #[test]
    fn test_score_frequency() {
        let now = now_unix();
        let high_freq = LearningEntry {
            surface: "A".to_string(),
            frequency: 100,
            last_access: now,
        };
        let low_freq = LearningEntry {
            surface: "B".to_string(),
            frequency: 1,
            last_access: now,
        };
        assert!(score(&high_freq, now) > score(&low_freq, now));
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = LearningCache::load(Path::new("/nonexistent/path"), 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_tsv_format() {
        let mut cache = LearningCache::new(100);
        cache.record("きょう", "今日");

        let file = NamedTempFile::new().unwrap();
        cache.save(file.path()).unwrap();

        // The TSV is the *decrypted* form now — on disk the file is
        // ChaCha20-Poly1305, so what the user typed is not readable there.
        let content = crate::learning_file::read_maybe_encrypted(file.path()).unwrap();
        assert!(content.starts_with(LearningCache::ENCRYPTED_HEADER));
        assert!(content.contains("きょう\t今日\t1\t"));
        let raw = std::fs::read(file.path()).unwrap();
        assert!(!String::from_utf8_lossy(&raw).contains("今日"));
    }

    #[test]
    fn a_plaintext_cache_is_rewritten_encrypted() {
        // Upgrading must keep what the user already taught the IME, and must
        // not leave the plaintext behind.
        let file = NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            "# kashiko learning cache v1\nきょう\t今日\t5\t1700000000\n",
        )
        .unwrap();

        let mut cache = LearningCache::load(file.path(), 100).unwrap();
        assert_eq!(
            cache.lookup("きょう").first().map(|(s, _)| s.as_str()),
            Some("今日")
        );
        // Loading a plaintext cache marks it dirty precisely so this happens.
        assert!(cache.is_dirty());
        cache.save(file.path()).unwrap();

        let raw = std::fs::read(file.path()).unwrap();
        assert!(!String::from_utf8_lossy(&raw).contains("今日"));
        let reloaded = LearningCache::load(file.path(), 100).unwrap();
        assert_eq!(
            reloaded.lookup("きょう").first().map(|(s, _)| s.as_str()),
            Some("今日")
        );
        assert!(!reloaded.is_dirty());
    }

    #[test]
    fn test_tsv_comments_and_blanks_ignored() {
        let file = NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            "# comment\n\nきょう\t今日\t5\t1700000000\n# another comment\n",
        )
        .unwrap();

        let cache = LearningCache::load(file.path(), 100).unwrap();
        assert_eq!(cache.entry_count(), 1);
        let results = cache.lookup("きょう");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "今日");
    }

    #[test]
    fn test_tsv_malformed_lines_skipped() {
        let file = NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            "きょう\t今日\t5\t1700000000\nmalformed_line\nきょう\t京\tbad\t1700000000\n",
        )
        .unwrap();

        let cache = LearningCache::load(file.path(), 100).unwrap();
        // Only the first valid line should be loaded
        assert_eq!(cache.entry_count(), 1);
    }
}

#[cfg(test)]
mod learnability_tests {
    use super::*;

    #[test]
    fn every_added_candidate_source_is_learnable() {
        // The tables that supply candidates no other source can produce
        // (loanwords, counted quantities) are only the *seed*: once the user
        // picks one it has to be learned, so ranking is driven by their own
        // choices from then on. A surface these filters reject would be offered
        // forever without ever being promoted.
        let mut cache = LearningCache::new(100);
        for (reading, surface) in [
            ("てんぷれーと", "template"),
            ("ぷるりくえすと", "pull request"),
            ("にげん", "2限"),
            ("じゅうにじ", "12時"),
            ("ごねん", "5年"),
            ("ゆうと", "優斗"),
            ("たなか", "田仲"),
        ] {
            cache.record(reading, surface);
            assert_eq!(
                cache.lookup(reading).first().map(|(s, _)| s.as_str()),
                Some(surface),
                "{reading} → {surface} was not learned"
            );
        }
    }
}
