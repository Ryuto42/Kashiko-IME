//! Learning cache for remembering user-selected conversion results.
//!
//! Records which surface forms the user chose for each reading, and
//! boosts those candidates on subsequent conversions. Persisted as a
//! simple TSV file (`reading\tsurface\tfrequency\tlast_access`).

use std::collections::HashMap;
use std::io::{BufRead, Write};
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
    /// reading or all-kana), surfaces with leftover ASCII romaji, and surfaces
    /// containing a run of 3+ identical characters (mistyped repeats).
    pub fn is_learnable(reading: &str, surface: &str) -> bool {
        if surface.is_empty() {
            return false;
        }
        // Feature 1: length cap.
        if surface.chars().count() >= Self::MAX_LEARN_SURFACE_CHARS {
            return false;
        }
        // Feature 3-2: leftover romaji (ASCII letters) means it was never converted.
        if surface.chars().any(|c| c.is_ascii_alphabetic()) {
            return false;
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
        if !has_kanji && surface.chars().count() >= 6 {
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
            .map(|e| (e.surface.clone(), score(e, now)))
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        scored
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
                for entry in entries {
                    results.push((reading.clone(), entry.surface.clone(), score(entry, now)));
                }
            }
        }
        results.sort_by(|a, b| b.2.total_cmp(&a.2));
        results
    }

    /// Load a learning cache from a TSV file.
    ///
    /// Format: `reading\tsurface\tfrequency\tlast_access`
    /// Lines starting with `#` are comments.
    pub fn load(path: &Path, max_entries: usize) -> anyhow::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        let mut cache = Self::new(max_entries);

        for line in reader.lines() {
            let line = line?;
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

        // Not dirty — just loaded from disk
        cache.dirty = false;
        Ok(cache)
    }

    /// Save the cache to a TSV file, evicting low-score entries if over capacity.
    pub fn save(&mut self, path: &Path) -> anyhow::Result<()> {
        self.evict();

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        writeln!(writer, "# kashiko learning cache v1")?;

        // Sort readings for deterministic output
        let mut readings: Vec<&String> = self.entries.keys().collect();
        readings.sort();

        for reading in readings {
            if let Some(entries) = self.entries.get(reading) {
                for entry in entries {
                    writeln!(
                        writer,
                        "{}\t{}\t{}\t{}",
                        reading, entry.surface, entry.frequency, entry.last_access
                    )?;
                }
            }
        }

        writer.flush()?;
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

        let content = std::fs::read_to_string(file.path()).unwrap();
        assert!(content.starts_with("# kashiko learning cache v1"));
        assert!(content.contains("きょう\t今日\t1\t"));
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
