use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::Path;

use unicode_normalization::UnicodeNormalization;

/// Errors that can occur during dictionary operations.
#[derive(Debug, thiserror::Error)]
pub enum DictError {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("JSON parse error")]
    Json(#[from] serde_json::Error),

    #[error("invalid dictionary format: {0}")]
    Format(String),
}

type Result<T> = std::result::Result<T, DictError>;
use serde::Deserialize;
use yada::DoubleArray;
use yada::builder::DoubleArrayBuilder;

use crate::kana::katakana_to_hiragana;

/// Decode a text file's bytes, honoring a UTF-8 or UTF-16 (LE/BE) BOM.
fn decode_text_file(bytes: &[u8]) -> Result<String> {
    let utf16 = |rest: &[u8], from: fn([u8; 2]) -> u16| {
        let units: Vec<u16> = rest
            .chunks_exact(2)
            .map(|pair| from([pair[0], pair[1]]))
            .collect();
        String::from_utf16(&units).map_err(|e| DictError::Format(e.to_string()))
    };
    match bytes {
        [0xFF, 0xFE, rest @ ..] => utf16(rest, u16::from_le_bytes),
        [0xFE, 0xFF, rest @ ..] => utf16(rest, u16::from_be_bytes),
        [0xEF, 0xBB, 0xBF, rest @ ..] => {
            String::from_utf8(rest.to_vec()).map_err(|e| DictError::Format(e.to_string()))
        }
        _ => String::from_utf8(bytes.to_vec()).map_err(|e| DictError::Format(e.to_string())),
    }
}

const MAGIC: &[u8; 4] = b"KRKN";
const VERSION: u32 = 1;

/// A candidate surface form with its score.
#[derive(Debug, Clone)]
pub struct Candidate {
    pub surface: String,
    pub score: f32,
}

/// A dictionary entry mapping a reading to its candidate surfaces.
#[derive(Debug, Clone)]
pub struct DictEntry {
    pub reading: String,
    pub candidates: Vec<Candidate>,
}

/// A candidate as returned by a lookup: borrows from the dictionary's arena.
///
/// Distinct from the owned [`Candidate`], which is what builders work with.
/// Lookups hand out borrowed text because the dictionary stores its strings in
/// one flat buffer rather than as individual `String`s — see [`Dictionary`].
#[derive(Debug, Clone, Copy)]
pub struct CandidateRef<'a> {
    pub surface: &'a str,
    pub score: f32,
}

/// Result of a dictionary lookup, borrowing from the Dictionary.
#[derive(Debug)]
pub struct LookupResult<'a> {
    pub reading: &'a str,
    pub candidates: Vec<CandidateRef<'a>>,
}

/// Where one reading's text and candidates live in the arena.
///
/// `u16` lengths because a reading or surface longer than 64KB cannot occur —
/// the on-disk format stores them as `u16` too.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct EntrySpan {
    reading_at: u32,
    cands_at: u32,
    reading_len: u16,
    cands_len: u16,
}

/// Where one candidate's surface lives in the arena, and its score.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct CandSpan {
    surface_at: u32,
    score: f32,
    surface_len: u16,
    _pad: u16,
}

/// A double-array trie dictionary for kana-kanji conversion.
///
/// Text is held in one flat `blob` with `u32` offsets into it, not as owned
/// `String`s. The system dictionary has ~2M readings and ~3M surfaces, and the
/// owned layout needed a separate heap allocation for every one of them plus a
/// `Vec` header per entry: measured on the real dictionary, a 196 MB file
/// occupied 382 MB resident. The arena is three allocations instead of seven
/// million, and holds the same bytes the file does.
pub struct Dictionary {
    trie: DoubleArray<Store<u8>>,
    blob: Store<u8>,
    entries: Store<EntrySpan>,
    cands: Store<CandSpan>,
}

/// A section of the dictionary: built in memory, or a slice of a mapped cache
/// file (see [`Dictionary::load_mapped`]). Mapped pages are file-backed — they
/// are read in only when a lookup touches them and the system can drop them at
/// any time, so they don't count against the process's memory.
enum Store<T: 'static> {
    Owned(Box<[T]>),
    Mapped {
        map: std::sync::Arc<memmap2::Mmap>,
        offset: usize,
        len: usize,
    },
}

impl<T: Copy> std::ops::Deref for Store<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match self {
            Store::Owned(items) => items,
            // SAFETY: `open_mapped` checked that `offset..offset + len * size`
            // lies inside the map and is aligned for `T`; `T` is a `repr(C)`
            // plain-data type for which any bit pattern is valid.
            Store::Mapped { map, offset, len } => unsafe {
                std::slice::from_raw_parts(map.as_ptr().add(*offset).cast::<T>(), *len)
            },
        }
    }
}

const MAPPED_MAGIC: &[u8; 8] = b"KSHKDMAP";
const MAPPED_VERSION: u64 = 1;
/// Magic, version, source fingerprint (2), and offset/count for 4 sections.
const MAPPED_HEADER_LEN: usize = 8 + 3 * 8 + 4 * 2 * 8;
const MAPPED_ALIGN: usize = 16;

fn align_up(n: usize) -> usize {
    n.div_ceil(MAPPED_ALIGN) * MAPPED_ALIGN
}

/// Size and modification time of the source dictionary: a replaced
/// `dict.bin` (a new download) invalidates the cache built from the old one.
fn source_fingerprint(path: &Path) -> Result<(u64, u64)> {
    let meta = std::fs::metadata(path)?;
    let modified = meta
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    Ok((meta.len(), modified))
}

fn as_bytes<T: Copy>(items: &[T]) -> &[u8] {
    // SAFETY: `T` is a `repr(C)` plain-data span type without padding bytes.
    unsafe { std::slice::from_raw_parts(items.as_ptr().cast::<u8>(), std::mem::size_of_val(items)) }
}

/// yada indexes the trie with `get_unchecked`, so a malformed trie would read
/// out of bounds. A builder-produced trie is whole 256-unit blocks and every
/// transition from a non-leaf unit stays inside the block that `offset ^ id`
/// selects, so checking that block for each unit bounds every lookup. The leaf
/// checks keep yada's internal asserts from panicking on a corrupt file.
fn validate_trie(bytes: &[u8], num_entries: usize) -> Result<()> {
    const UNIT_SIZE: usize = 4;
    const BLOCK_UNITS: usize = 256;
    if bytes.is_empty() || !bytes.len().is_multiple_of(UNIT_SIZE * BLOCK_UNITS) {
        return Err(DictError::Format(format!(
            "trie length {} is not a whole number of blocks",
            bytes.len()
        )));
    }
    let num_units = bytes.len() / UNIT_SIZE;
    let unit = |i: usize| {
        u32::from_le_bytes([
            bytes[i * UNIT_SIZE],
            bytes[i * UNIT_SIZE + 1],
            bytes[i * UNIT_SIZE + 2],
            bytes[i * UNIT_SIZE + 3],
        ])
    };
    let is_leaf = |u: u32| u >> 31 == 1;
    if is_leaf(unit(0)) {
        return Err(DictError::Format("trie root is a leaf".to_string()));
    }
    for i in 0..num_units {
        let u = unit(i);
        if is_leaf(u) {
            if (u & 0x7FFF_FFFF) as usize >= num_entries {
                return Err(DictError::Format(format!(
                    "trie value out of range at unit {i}"
                )));
            }
            continue;
        }
        let offset = (u >> 10) << ((u & (1 << 9)) >> 6);
        let base = (offset ^ i as u32) as usize;
        if base / BLOCK_UNITS >= num_units / BLOCK_UNITS {
            return Err(DictError::Format(format!(
                "trie transition out of bounds at unit {i}"
            )));
        }
        if (u >> 8) & 1 == 1 && !is_leaf(unit(base)) {
            return Err(DictError::Format(format!("trie leaf missing at unit {i}")));
        }
    }
    Ok(())
}

impl Dictionary {
    fn text(&self, at: u32, len: u16) -> &str {
        let (at, len) = (at as usize, len as usize);
        // Validated once when the arena was built.
        std::str::from_utf8(&self.blob[at..at + len]).unwrap_or_default()
    }

    fn candidates_of(&self, span: &EntrySpan) -> Vec<CandidateRef<'_>> {
        let at = span.cands_at as usize;
        self.cands[at..at + span.cands_len as usize]
            .iter()
            .map(|c| CandidateRef {
                surface: self.text(c.surface_at, c.surface_len),
                score: c.score,
            })
            .collect()
    }

    fn lookup_at(&self, index: usize) -> Option<LookupResult<'_>> {
        let span = self.entries.get(index)?;
        Some(LookupResult {
            reading: self.text(span.reading_at, span.reading_len),
            candidates: self.candidates_of(span),
        })
    }

    /// Every reading with its candidates, for the builders that need owned data
    /// back (merge, dump, surface search).
    fn iter_entries(&self) -> impl Iterator<Item = (&str, Vec<CandidateRef<'_>>)> {
        self.entries.iter().map(move |s| {
            (
                self.text(s.reading_at, s.reading_len),
                self.candidates_of(s),
            )
        })
    }

    /// Pack owned entries into the flat arena.
    fn pack(entries: &[DictEntry]) -> (Box<[u8]>, Box<[EntrySpan]>, Box<[CandSpan]>) {
        let text_len: usize = entries
            .iter()
            .map(|e| e.reading.len() + e.candidates.iter().map(|c| c.surface.len()).sum::<usize>())
            .sum();
        let cand_total: usize = entries.iter().map(|e| e.candidates.len()).sum();

        let mut blob = Vec::with_capacity(text_len);
        let mut spans = Vec::with_capacity(entries.len());
        let mut cands = Vec::with_capacity(cand_total);

        for e in entries {
            let reading_at = blob.len() as u32;
            blob.extend_from_slice(e.reading.as_bytes());
            let cands_at = cands.len() as u32;
            for c in &e.candidates {
                let surface_at = blob.len() as u32;
                blob.extend_from_slice(c.surface.as_bytes());
                cands.push(CandSpan {
                    surface_at,
                    score: c.score,
                    surface_len: c.surface.len() as u16,
                    _pad: 0,
                });
            }
            spans.push(EntrySpan {
                reading_at,
                cands_at,
                reading_len: e.reading.len() as u16,
                cands_len: e.candidates.len() as u16,
            });
        }
        (
            blob.into_boxed_slice(),
            spans.into_boxed_slice(),
            cands.into_boxed_slice(),
        )
    }
}

// JSON deserialization types
#[derive(Deserialize)]
struct JsonCandidate {
    surface: String,
    score: f32,
}

#[derive(Deserialize)]
struct JsonEntry {
    reading: String,
    candidates: Vec<JsonCandidate>,
}

impl Dictionary {
    /// Build a Dictionary from pre-sorted entries.
    ///
    /// Entries must already be sorted by `reading` bytes and deduplicated.
    /// This is the shared final step for all dictionary builders.
    fn build_from_entries(entries: Vec<DictEntry>) -> Result<Self> {
        let keyset: Vec<(&[u8], u32)> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.reading.as_bytes(), i as u32))
            .collect();

        let trie_bytes = DoubleArrayBuilder::build(&keyset)
            .ok_or_else(|| DictError::Format("failed to build double-array trie".to_string()))?;

        let (blob, spans, cands) = Self::pack(&entries);
        Ok(Dictionary {
            trie: DoubleArray::new(Store::Owned(trie_bytes.into_boxed_slice())),
            blob: Store::Owned(blob),
            entries: Store::Owned(spans),
            cands: Store::Owned(cands),
        })
    }

    /// Build a Dictionary from a JSON file.
    ///
    /// The JSON format is an array of `{reading, candidates: [{surface, score}]}`.
    /// Readings are converted from katakana to hiragana.
    pub fn build_from_json(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let reader = BufReader::new(file);
        let json_entries: Vec<JsonEntry> = serde_json::from_reader(reader)?;

        // Convert to DictEntry with hiragana readings
        let mut entries: Vec<DictEntry> = json_entries
            .into_iter()
            .map(|je| DictEntry {
                reading: katakana_to_hiragana(&je.reading),
                candidates: {
                    let mut cands: Vec<Candidate> = je
                        .candidates
                        .into_iter()
                        .map(|jc| Candidate {
                            surface: jc.surface,
                            score: jc.score,
                        })
                        .collect();
                    cands.sort_by(|a, b| a.score.total_cmp(&b.score));
                    cands
                },
            })
            .collect();

        // Sort by reading bytes for the trie builder
        entries.sort_by(|a, b| a.reading.as_bytes().cmp(b.reading.as_bytes()));

        // Deduplicate entries with the same reading (keep the first occurrence)
        entries.dedup_by(|b, a| a.reading == b.reading);

        Self::build_from_entries(entries)
    }

    /// Save the dictionary to a binary file.
    ///
    /// Format:
    /// ```text
    /// [4B] magic "KRKN"
    /// [4B] version (1u32 LE)
    /// [4B] trie_len (u32 LE)
    /// [trie_len B] trie bytes
    /// [4B] num_entries (u32 LE)
    /// For each entry:
    ///   [2B] reading_len (u16 LE)
    ///   [reading_len B] reading (UTF-8)
    ///   [2B] num_candidates (u16 LE)
    ///   For each candidate:
    ///     [2B] surface_len (u16 LE)
    ///     [surface_len B] surface (UTF-8)
    ///     [4B] score (f32 LE)
    /// ```
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path.as_ref())?;
        let mut w = BufWriter::new(file);

        // Magic + version
        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;

        // Trie data
        let trie_bytes: &[u8] = &self.trie.0;
        w.write_all(&(trie_bytes.len() as u32).to_le_bytes())?;
        w.write_all(trie_bytes)?;

        // Entries
        w.write_all(&(self.entries.len() as u32).to_le_bytes())?;
        for (reading, candidates) in self.iter_entries() {
            w.write_all(&(reading.len() as u16).to_le_bytes())?;
            w.write_all(reading.as_bytes())?;

            w.write_all(&(candidates.len() as u16).to_le_bytes())?;
            for cand in candidates {
                w.write_all(&(cand.surface.len() as u16).to_le_bytes())?;
                w.write_all(cand.surface.as_bytes())?;
                w.write_all(&cand.score.to_le_bytes())?;
            }
        }

        w.flush()?;
        Ok(())
    }

    /// Load a dictionary from a binary file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let mut r = BufReader::new(file);

        // Magic
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(DictError::Format(
                "invalid magic: expected KRKN".to_string(),
            ));
        }

        // Version
        let mut buf4 = [0u8; 4];
        r.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != VERSION {
            return Err(DictError::Format(format!("unsupported version: {version}")));
        }

        // Trie
        r.read_exact(&mut buf4)?;
        let trie_len = u32::from_le_bytes(buf4) as usize;
        const MAX_TRIE_LEN: usize = 100 * 1024 * 1024; // 100 MB
        if trie_len > MAX_TRIE_LEN {
            return Err(DictError::Format(format!(
                "trie_len too large: {} (max {})",
                trie_len, MAX_TRIE_LEN
            )));
        }
        let mut trie_bytes = vec![0u8; trie_len];
        r.read_exact(&mut trie_bytes)?;

        // Entries
        r.read_exact(&mut buf4)?;
        let num_entries = u32::from_le_bytes(buf4) as usize;
        const MAX_ENTRIES: usize = 10_000_000;
        if num_entries > MAX_ENTRIES {
            return Err(DictError::Format(format!(
                "num_entries too large: {} (max {})",
                num_entries, MAX_ENTRIES
            )));
        }
        validate_trie(&trie_bytes, num_entries)?;
        // Stream the entry section into a text-only arena: the file interleaves
        // lengths and scores with the text, and those are already in the index
        // arrays, so copying them into the blob would be dead weight. Streaming
        // rather than reading the section whole also avoids a second copy of it
        // being live at once during startup.
        let section_len = std::fs::metadata(path.as_ref())
            .map(|m| m.len() as usize)
            .unwrap_or(0)
            .saturating_sub(16 + trie_len);
        let mut blob: Vec<u8> = Vec::with_capacity(section_len);
        let mut spans: Vec<EntrySpan> = Vec::with_capacity(num_entries);
        let mut cands: Vec<CandSpan> = Vec::new();

        let mut buf2 = [0u8; 2];
        let mut scratch: Vec<u8> = Vec::with_capacity(256);
        for _ in 0..num_entries {
            r.read_exact(&mut buf2)?;
            let reading_len = u16::from_le_bytes(buf2) as usize;
            let reading_at = blob.len();
            scratch.resize(reading_len, 0);
            r.read_exact(&mut scratch)?;
            std::str::from_utf8(&scratch)
                .map_err(|e| DictError::Format(format!("invalid UTF-8 in reading: {e}")))?;
            blob.extend_from_slice(&scratch);

            r.read_exact(&mut buf2)?;
            let num_candidates = u16::from_le_bytes(buf2) as usize;
            let cands_at = cands.len();
            for _ in 0..num_candidates {
                r.read_exact(&mut buf2)?;
                let surface_len = u16::from_le_bytes(buf2) as usize;
                let surface_at = blob.len();
                scratch.resize(surface_len, 0);
                r.read_exact(&mut scratch)?;
                std::str::from_utf8(&scratch)
                    .map_err(|e| DictError::Format(format!("invalid UTF-8 in surface: {e}")))?;
                blob.extend_from_slice(&scratch);

                r.read_exact(&mut buf4)?;
                cands.push(CandSpan {
                    surface_at: surface_at as u32,
                    score: f32::from_le_bytes(buf4),
                    surface_len: surface_len as u16,
                    _pad: 0,
                });
            }

            // Same ordering the owned layout produced: the index is sorted, the
            // text stays where it was written.
            cands[cands_at..].sort_by(|a, b| a.score.total_cmp(&b.score));
            spans.push(EntrySpan {
                reading_at: reading_at as u32,
                cands_at: cands_at as u32,
                reading_len: reading_len as u16,
                cands_len: num_candidates as u16,
            });
        }
        // Offsets are stored as u32; a larger arena would have wrapped them.
        if blob.len() > u32::MAX as usize || cands.len() > u32::MAX as usize {
            return Err(DictError::Format("entry section too large".to_string()));
        }
        blob.shrink_to_fit();
        tracing::debug!(
            "dict arena: trie {:.1} MB, text {:.1} MB, entry index {:.1} MB, candidate index {:.1} MB ({} candidates)",
            trie_len as f64 / 1048576.0,
            blob.len() as f64 / 1048576.0,
            (spans.len() * std::mem::size_of::<EntrySpan>()) as f64 / 1048576.0,
            (cands.len() * std::mem::size_of::<CandSpan>()) as f64 / 1048576.0,
            cands.len(),
        );

        Ok(Dictionary {
            trie: DoubleArray::new(Store::Owned(trie_bytes.into_boxed_slice())),
            blob: Store::Owned(blob.into_boxed_slice()),
            entries: Store::Owned(spans.into_boxed_slice()),
            cands: Store::Owned(cands.into_boxed_slice()),
        })
    }

    /// Load `path` through a memory-mapped cache next to it (`*.mmap`),
    /// writing that cache first when it is missing or was made from a
    /// different `path`. Falls back to [`Self::load`] if the cache can't be
    /// written or mapped.
    pub fn load_mapped(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let cache = path.with_extension("mmap");
        let source = source_fingerprint(path)?;
        match Self::open_mapped(&cache, source) {
            Ok(dict) => return Ok(dict),
            Err(e) => tracing::info!("building dictionary cache {}: {e}", cache.display()),
        }
        let dict = Self::load(path)?;
        if let Err(e) = dict.write_mapped(&cache, source) {
            tracing::warn!("could not write dictionary cache {}: {e}", cache.display());
            return Ok(dict);
        }
        // Free the in-memory copy before mapping; reload it only if mapping fails.
        drop(dict);
        Self::open_mapped(&cache, source).or_else(|e| {
            tracing::warn!("could not map dictionary cache {}: {e}", cache.display());
            Self::load(path)
        })
    }

    fn write_mapped(&self, cache: &Path, source: (u64, u64)) -> Result<()> {
        let trie: &[u8] = &self.trie.0;
        let sections: [(&[u8], usize); 4] = [
            (trie, trie.len()),
            (&self.blob, self.blob.len()),
            (as_bytes(&self.entries), self.entries.len()),
            (as_bytes(&self.cands), self.cands.len()),
        ];
        let mut header = Vec::with_capacity(MAPPED_HEADER_LEN);
        header.extend_from_slice(MAPPED_MAGIC);
        for v in [MAPPED_VERSION, source.0, source.1] {
            header.extend_from_slice(&v.to_le_bytes());
        }
        let mut offset = MAPPED_HEADER_LEN;
        for (bytes, count) in &sections {
            header.extend_from_slice(&(offset as u64).to_le_bytes());
            header.extend_from_slice(&(*count as u64).to_le_bytes());
            offset = align_up(offset + bytes.len());
        }
        let tmp = cache.with_extension("mmap.tmp");
        {
            let mut w = BufWriter::new(File::create(&tmp)?);
            w.write_all(&header)?;
            let mut written = MAPPED_HEADER_LEN;
            for (bytes, _) in &sections {
                w.write_all(bytes)?;
                written += bytes.len();
                let padded = align_up(written);
                w.write_all(&vec![0u8; padded - written])?;
                written = padded;
            }
            w.flush()?;
            w.get_ref().sync_all()?;
        }
        std::fs::rename(&tmp, cache)?;
        Ok(())
    }

    fn open_mapped(cache: &Path, source: (u64, u64)) -> Result<Self> {
        let file = File::open(cache)?;
        // SAFETY: the cache is only ever replaced by rename, never rewritten in
        // place, so the mapped file's contents don't change underneath us.
        let map = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&file)? });
        let bad = |why: &str| DictError::Format(format!("dictionary cache: {why}"));
        if map.len() < MAPPED_HEADER_LEN || &map[..8] != MAPPED_MAGIC {
            return Err(bad("not a dictionary cache"));
        }
        let word = |i: usize| u64::from_le_bytes(map[8 + i * 8..16 + i * 8].try_into().unwrap());
        if word(0) != MAPPED_VERSION {
            return Err(bad("unsupported version"));
        }
        if (word(1), word(2)) != source {
            return Err(bad("made from a different dictionary"));
        }
        let section = |i: usize, size: usize| -> Result<(usize, usize)> {
            let (offset, count) = (word(3 + i * 2) as usize, word(4 + i * 2) as usize);
            let end = count
                .checked_mul(size)
                .and_then(|n| n.checked_add(offset))
                .ok_or_else(|| bad("section overflows"))?;
            if offset % MAPPED_ALIGN != 0 || end > map.len() {
                return Err(bad("section out of range"));
            }
            Ok((offset, count))
        };
        let (trie_at, trie_len) = section(0, 1)?;
        let (blob_at, blob_len) = section(1, 1)?;
        let (entries_at, entries_len) = section(2, std::mem::size_of::<EntrySpan>())?;
        let (cands_at, cands_len) = section(3, std::mem::size_of::<CandSpan>())?;
        // yada reads the trie unchecked; it must be validated like any file.
        validate_trie(&map[trie_at..trie_at + trie_len], entries_len)?;
        let store = |offset, len| Store::Mapped {
            map: map.clone(),
            offset,
            len,
        };
        Ok(Dictionary {
            trie: DoubleArray::new(store(trie_at, trie_len)),
            blob: store(blob_at, blob_len),
            entries: Store::Mapped {
                map: map.clone(),
                offset: entries_at,
                len: entries_len,
            },
            cands: Store::Mapped {
                map: map.clone(),
                offset: cands_at,
                len: cands_len,
            },
        })
    }

    /// Common prefix search: returns all entries whose reading is a prefix of `input`.
    pub fn common_prefix_search(&self, input: &str) -> Vec<LookupResult<'_>> {
        self.trie
            .common_prefix_search(input.as_bytes())
            .filter_map(|(value, _len)| self.lookup_at(value as usize))
            .collect()
    }

    /// Exact match search: returns the entry whose reading exactly matches `input`.
    pub fn exact_match_search(&self, input: &str) -> Option<LookupResult<'_>> {
        let value = self.trie.exact_match_search(input.as_bytes())?;
        self.lookup_at(value as usize)
    }

    /// Write all entries in the dictionary to `writer` (for inspection/debugging).
    ///
    /// Each line is tab-separated: `reading\tsurface\tscore`.
    /// Returns the total number of entries written.
    pub fn dump_all(&self, writer: &mut dyn std::io::Write) -> std::io::Result<usize> {
        for (reading, candidates) in self.iter_entries() {
            for cand in candidates {
                writeln!(writer, "{}\t{}\t{}", reading, cand.surface, cand.score)?;
            }
        }
        Ok(self.entries.len())
    }

    /// Number of readings in the dictionary.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Search entries by surface form (substring match).
    ///
    /// Returns a list of (reading, surface, score) tuples where surface contains `query`.
    pub fn search_by_surface(&self, query: &str) -> Vec<(String, String, f32)> {
        let mut results = Vec::new();
        for (reading, candidates) in self.iter_entries() {
            for cand in candidates {
                if cand.surface.contains(query) {
                    results.push((reading.to_string(), cand.surface.to_string(), cand.score));
                }
            }
        }
        results
    }

    /// Build a Dictionary from a Mozc/Google IME TSV file.
    ///
    /// The TSV format is `reading\tword\tPOS\tcomment` (tab-separated, 4 columns).
    /// Lines starting with `#` or `!` are comments, empty lines are skipped.
    /// Readings are grouped and converted to `DictEntry` with score 0.0.
    ///
    /// UTF-16 (LE/BE) files with a BOM are decoded as well, so the text export
    /// of the Microsoft IME user dictionary (`!` header line, then
    /// `reading\tword\tPOS`) loads as is. A UTF-8 BOM is dropped.
    pub fn build_from_mozc_tsv(path: impl AsRef<Path>) -> Result<Self> {
        let text = decode_text_file(&std::fs::read(path.as_ref())?)?;

        // reading -> Vec<surface> (preserving insertion order)
        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        let mut order: Vec<String> = Vec::new();

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with('!') {
                continue;
            }

            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() < 2 {
                continue;
            }

            let reading = cols[0].to_string();
            let surface = cols[1].to_string();

            if reading.is_empty() || surface.is_empty() {
                continue;
            }

            let surfaces = groups.entry(reading.clone()).or_insert_with(|| {
                order.push(reading);
                Vec::new()
            });
            // Deduplicate surfaces within the same reading
            if !surfaces.contains(&surface) {
                surfaces.push(surface);
            }
        }

        // Convert to DictEntry
        let mut entries: Vec<DictEntry> = order
            .into_iter()
            .filter_map(|reading| {
                groups.remove(&reading).map(|surfaces| DictEntry {
                    reading,
                    candidates: surfaces
                        .into_iter()
                        .map(|surface| Candidate {
                            surface,
                            score: 0.0,
                        })
                        .collect(),
                })
            })
            .collect();

        // Sort by reading bytes for the trie builder
        entries.sort_by(|a, b| a.reading.as_bytes().cmp(b.reading.as_bytes()));

        // Deduplicate entries with the same reading (keep the first occurrence)
        entries.dedup_by(|b, a| {
            if a.reading == b.reading {
                // Merge candidates from b into a
                for cand in std::mem::take(&mut b.candidates) {
                    if !a.candidates.iter().any(|c| c.surface == cand.surface) {
                        a.candidates.push(cand);
                    }
                }
                true
            } else {
                false
            }
        });

        Self::build_from_entries(entries)
    }

    /// Load a dictionary with auto-detection of format.
    ///
    /// If the file starts with the `KRKN` magic bytes, it is loaded as binary.
    /// Otherwise, it is parsed as Mozc/Google IME TSV format.
    pub fn load_auto(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let mut magic = [0u8; 4];
        let bytes_read = file.read(&mut magic)?;

        if bytes_read >= 4 && &magic == MAGIC {
            // Binary KRKN format
            Dictionary::load(path)
        } else {
            // Mozc TSV format
            Dictionary::build_from_mozc_tsv(path)
        }
    }

    /// Merge multiple dictionaries into one.
    ///
    /// Dictionaries earlier in the list have higher priority: their candidates
    /// appear first for the same reading. Returns `None` if the input is empty.
    pub fn merge(dicts: Vec<Dictionary>) -> Result<Option<Self>> {
        if dicts.is_empty() {
            return Ok(None);
        }

        // Collect all entries, grouped by reading
        let mut merged: HashMap<String, Vec<Candidate>> = HashMap::new();
        let mut reading_order: Vec<String> = Vec::new();

        for dict in dicts {
            for (reading, cands) in dict.iter_entries() {
                if !merged.contains_key(reading) {
                    reading_order.push(reading.to_string());
                }
                let candidates = merged.entry(reading.to_string()).or_default();
                for cand in cands {
                    if !candidates.iter().any(|c| c.surface == cand.surface) {
                        candidates.push(Candidate {
                            surface: cand.surface.to_string(),
                            score: cand.score,
                        });
                    }
                }
            }
        }

        let mut entries: Vec<DictEntry> = reading_order
            .into_iter()
            .filter_map(|reading| {
                merged.remove(&reading).map(|candidates| DictEntry {
                    reading,
                    candidates,
                })
            })
            .collect();

        // Sort by reading bytes for the trie builder
        entries.sort_by(|a, b| a.reading.as_bytes().cmp(b.reading.as_bytes()));

        Self::build_from_entries(entries).map(Some)
    }
}

/// Unescape `\uXXXX` Unicode escape sequences in a string.
///
/// Sudachi CSV files contain literal `\uXXXX` sequences (e.g. `\u0028` for `(`)
/// in surface forms, especially for emoji/kaomoji entries. This function converts
/// them back to actual Unicode characters.
fn unescape_unicode(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            // Peek at next char
            let mut clone = chars.clone();
            if clone.next() == Some('u') {
                // Try to read 4 hex digits
                let hex: String = clone.by_ref().take(4).collect();
                if hex.len() == 4
                    && let Ok(code) = u32::from_str_radix(&hex, 16)
                    && let Some(ch) = char::from_u32(code)
                {
                    result.push(ch);
                    // Advance the actual iterator past 'u' + 4 hex digits
                    chars.next(); // 'u'
                    for _ in 0..4 {
                        chars.next();
                    }
                    continue;
                }
            }
            result.push(c);
        } else {
            result.push(c);
        }
    }
    result
}

/// Parse a single Sudachi CSV file into a map of reading → {surface → min_cost}.
///
/// Sudachi CSV columns:
/// - 4: surface form (見出し 解析結果表示用)
/// - 3: cost (integer)
/// - 11: reading (katakana)
///
/// Readings are NFKC-normalized. Surface forms have `\uXXXX` Unicode escapes decoded.
/// For duplicate (reading, surface) pairs, the minimum cost is kept.
pub fn parse_sudachi_csv(path: &Path) -> Result<HashMap<String, HashMap<String, i32>>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut map: HashMap<String, HashMap<String, i32>> = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let cols: Vec<&str> = line.split(',').collect();
        if cols.len() < 12 {
            continue;
        }

        // Skip AA (kaomoji/emoticon) entries — not useful for IME conversion
        if cols.len() > 6 && cols[5] == "補助記号" && cols[6] == "ＡＡ" {
            continue;
        }

        let surface = unescape_unicode(cols[4]);
        // Entries with left/right context IDs of -1 have no reliable cost from Sudachi;
        // assign a large fallback cost so they rank low.
        let cost: i32 = if cols[1] == "-1" && cols[2] == "-1" {
            99999
        } else {
            match cols[3].parse() {
                Ok(v) => v,
                Err(_) => continue,
            }
        };
        let reading: String = cols[11].nfkc().collect();

        if reading.is_empty() || surface.is_empty() {
            continue;
        }

        let surfaces = map.entry(reading).or_default();
        let entry = surfaces.entry(surface).or_insert(cost);
        if cost < *entry {
            *entry = cost;
        }
    }

    Ok(map)
}

/// Parse multiple Sudachi CSV files and merge into a single map.
///
/// For duplicate (reading, surface) pairs across files, the minimum cost is kept.
pub fn parse_sudachi_csvs(
    paths: &[impl AsRef<Path>],
) -> Result<HashMap<String, HashMap<String, i32>>> {
    let mut merged: HashMap<String, HashMap<String, i32>> = HashMap::new();

    for path in paths {
        let map = parse_sudachi_csv(path.as_ref())?;
        merge_reading_maps(&mut merged, map);
    }

    Ok(merged)
}

/// Merge `source` reading map into `target`, keeping minimum costs.
pub fn merge_reading_maps(
    target: &mut HashMap<String, HashMap<String, i32>>,
    source: HashMap<String, HashMap<String, i32>>,
) {
    for (reading, surfaces) in source {
        let target_surfaces = target.entry(reading).or_default();
        for (surface, cost) in surfaces {
            let entry = target_surfaces.entry(surface).or_insert(cost);
            if cost < *entry {
                *entry = cost;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_json() -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        // Note: readings are in katakana (as they come from the JSON)
        let json = r#"[
            {
                "reading": "キョウ",
                "candidates": [
                    {"surface": "今日", "score": 1.5},
                    {"surface": "京", "score": 0.8}
                ]
            },
            {
                "reading": "キョウト",
                "candidates": [
                    {"surface": "京都", "score": 2.0}
                ]
            },
            {
                "reading": "トウキョウ",
                "candidates": [
                    {"surface": "東京", "score": 2.5}
                ]
            }
        ]"#;
        f.write_all(json.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    fn lookups(dict: &Dictionary) -> Vec<(String, Vec<String>)> {
        ["きょう", "きょうと", "とうきょう", "なし"]
            .iter()
            .map(|r| {
                let cands = dict
                    .exact_match_search(r)
                    .map(|l| l.candidates.iter().map(|c| c.surface.to_string()).collect())
                    .unwrap_or_default();
                (r.to_string(), cands)
            })
            .collect()
    }

    #[test]
    fn mapped_cache_answers_like_the_loaded_dictionary() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("dict.bin");
        Dictionary::build_from_json(create_test_json().path())
            .unwrap()
            .save(&bin)
            .unwrap();
        let loaded = Dictionary::load(&bin).unwrap();
        let mapped = Dictionary::load_mapped(&bin).unwrap();
        assert!(matches!(mapped.blob, Store::Mapped { .. }));
        assert!(dir.path().join("dict.mmap").exists());
        assert_eq!(lookups(&mapped), lookups(&loaded));
        assert_eq!(
            mapped.common_prefix_search("きょうとし").len(),
            loaded.common_prefix_search("きょうとし").len()
        );
        // Second load maps the existing cache.
        let again = Dictionary::load_mapped(&bin).unwrap();
        assert_eq!(lookups(&again), lookups(&loaded));
    }

    #[test]
    fn a_replaced_dictionary_rebuilds_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("dict.bin");
        Dictionary::build_from_json(create_test_json().path())
            .unwrap()
            .save(&bin)
            .unwrap();
        Dictionary::load_mapped(&bin).unwrap();
        let replacement = Dictionary::build_from_entries(vec![DictEntry {
            reading: "あめ".into(),
            candidates: vec![Candidate {
                surface: "雨".into(),
                score: 1.0,
            }],
        }])
        .unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
        replacement.save(&bin).unwrap();
        let mapped = Dictionary::load_mapped(&bin).unwrap();
        assert!(mapped.exact_match_search("あめ").is_some());
        assert!(mapped.exact_match_search("きょう").is_none());
    }

    #[test]
    fn a_corrupt_cache_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("dict.bin");
        Dictionary::build_from_json(create_test_json().path())
            .unwrap()
            .save(&bin)
            .unwrap();
        Dictionary::load_mapped(&bin).unwrap();
        let cache = dir.path().join("dict.mmap");
        let len = std::fs::metadata(&cache).unwrap().len();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&cache)
            .unwrap()
            .set_len(len / 2)
            .unwrap();
        let source = source_fingerprint(&bin).unwrap();
        assert!(Dictionary::open_mapped(&cache, source).is_err());
        // Loading falls back to rebuilding it.
        assert!(
            Dictionary::load_mapped(&bin)
                .unwrap()
                .exact_match_search("きょう")
                .is_some()
        );
    }

    #[test]
    fn test_build_from_json() {
        let json_file = create_test_json();
        let dict = Dictionary::build_from_json(json_file.path()).unwrap();

        // Readings should be converted to hiragana
        let readings: Vec<&str> = dict.iter_entries().map(|(r, _)| r).collect();
        assert!(readings.contains(&"きょう"));
        assert!(readings.contains(&"きょうと"));
        assert!(readings.contains(&"とうきょう"));
    }

    #[test]
    fn test_exact_match_search() {
        let json_file = create_test_json();
        let dict = Dictionary::build_from_json(json_file.path()).unwrap();

        let result = dict.exact_match_search("きょう").unwrap();
        assert_eq!(result.reading, "きょう");
        assert_eq!(result.candidates.len(), 2);
        // Candidates should be sorted by score ascending
        assert_eq!(result.candidates[0].surface, "京");
        assert!((result.candidates[0].score - 0.8).abs() < f32::EPSILON);
        assert_eq!(result.candidates[1].surface, "今日");
        assert!((result.candidates[1].score - 1.5).abs() < f32::EPSILON);

        assert!(dict.exact_match_search("きょうとふ").is_none());
    }

    #[test]
    fn test_common_prefix_search() {
        let json_file = create_test_json();
        let dict = Dictionary::build_from_json(json_file.path()).unwrap();

        // "きょうと" should match both "きょう" and "きょうと"
        let results = dict.common_prefix_search("きょうと");
        assert_eq!(results.len(), 2);
        let readings: Vec<&str> = results.iter().map(|r| r.reading).collect();
        assert!(readings.contains(&"きょう"));
        assert!(readings.contains(&"きょうと"));
    }

    #[test]
    fn test_save_and_load() {
        let json_file = create_test_json();
        let dict = Dictionary::build_from_json(json_file.path()).unwrap();

        let bin_file = NamedTempFile::new().unwrap();
        dict.save(bin_file.path()).unwrap();

        let loaded = Dictionary::load(bin_file.path()).unwrap();

        // Verify loaded dictionary works the same
        let result = loaded.exact_match_search("きょう").unwrap();
        assert_eq!(result.reading, "きょう");
        assert_eq!(result.candidates.len(), 2);
        assert_eq!(result.candidates[0].surface, "京");
        assert!((result.candidates[0].score - 0.8).abs() < f32::EPSILON);

        let results = loaded.common_prefix_search("きょうと");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_load_rejects_corrupt_trie() {
        let json_file = create_test_json();
        let dict = Dictionary::build_from_json(json_file.path()).unwrap();
        let bin_file = NamedTempFile::new().unwrap();
        dict.save(bin_file.path()).unwrap();

        // Root unit (right after magic, version, trie_len) pointing far past
        // the end of the trie.
        let mut bytes = std::fs::read(bin_file.path()).unwrap();
        bytes[12..16].copy_from_slice(&(0x1F_FFFFu32 << 10).to_le_bytes());
        std::fs::write(bin_file.path(), &bytes).unwrap();

        assert!(matches!(
            Dictionary::load(bin_file.path()),
            Err(DictError::Format(_))
        ));
    }

    #[test]
    fn test_validate_trie_rejects_partial_block() {
        assert!(validate_trie(&[0u8; 1020], 0).is_err());
        assert!(validate_trie(&[], 0).is_err());
        assert!(validate_trie(&[0u8; 1024], 0).is_ok());
    }

    #[test]
    fn test_no_match() {
        let json_file = create_test_json();
        let dict = Dictionary::build_from_json(json_file.path()).unwrap();

        assert!(dict.exact_match_search("おおさか").is_none());
        assert!(dict.common_prefix_search("おおさか").is_empty());
    }

    fn create_test_sudachi_csv() -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        // Sudachi CSV format: col0,col1,col2,cost,surface(解析結果表示用),...,col11=reading(katakana)
        // Columns 0-11 (12 columns minimum), surface is taken from column 4
        let csv = "\
col0,col1,col2,5000,今日,col5,col6,col7,col8,col9,col10,キョウ
col0,col1,col2,6000,京,col5,col6,col7,col8,col9,col10,キョウ
col0,col1,col2,4000,京都,col5,col6,col7,col8,col9,col10,キョウト
col0,col1,col2,3000,東京,col5,col6,col7,col8,col9,col10,トウキョウ
col0,col1,col2,4500,今日,col5,col6,col7,col8,col9,col10,キョウ
";
        f.write_all(csv.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_parse_sudachi_csv() {
        let csv_file = create_test_sudachi_csv();
        let map = parse_sudachi_csv(csv_file.path()).unwrap();

        // Check reading groups
        assert!(map.contains_key("キョウ"));
        assert!(map.contains_key("キョウト"));
        assert!(map.contains_key("トウキョウ"));

        // Check surfaces for キョウ
        let kyou = &map["キョウ"];
        assert_eq!(kyou.len(), 2); // 今日, 京
        assert_eq!(kyou["今日"], 4500); // min(5000, 4500) = 4500
        assert_eq!(kyou["京"], 6000);

        // Check surfaces for キョウト
        let kyouto = &map["キョウト"];
        assert_eq!(kyouto.len(), 1);
        assert_eq!(kyouto["京都"], 4000);
    }

    #[test]
    fn test_parse_sudachi_csvs_merge() {
        let csv1 = create_test_sudachi_csv();

        let mut csv2 = NamedTempFile::new().unwrap();
        csv2.write_all(
            "col0,col1,col2,3500,大阪,col5,col6,col7,col8,col9,col10,オオサカ\n".as_bytes(),
        )
        .unwrap();
        csv2.write_all(
            "col0,col1,col2,4000,今日,col5,col6,col7,col8,col9,col10,キョウ\n".as_bytes(),
        )
        .unwrap();
        csv2.flush().unwrap();

        let paths = vec![csv1.path().to_path_buf(), csv2.path().to_path_buf()];
        let map = parse_sudachi_csvs(&paths).unwrap();

        assert!(map.contains_key("オオサカ"));
        assert_eq!(map["オオサカ"]["大阪"], 3500);

        // 今日 should have min cost across both files: min(4500, 4000) = 4000
        assert_eq!(map["キョウ"]["今日"], 4000);
    }

    #[test]
    fn test_unescape_unicode() {
        // Basic escapes
        assert_eq!(unescape_unicode(r"\u0028"), "(");
        assert_eq!(unescape_unicode(r"\u0029"), ")");

        // Mixed content (kaomoji-like)
        assert_eq!(
            unescape_unicode(r"ムカ!σ\u0028`・ω・ ́;\u0029"),
            "ムカ!σ(`・ω・ ́;)"
        );

        // No escapes
        assert_eq!(unescape_unicode("hello"), "hello");

        // Incomplete escape (should be left as-is)
        assert_eq!(unescape_unicode(r"\u00"), r"\u00");

        // Backslash not followed by 'u'
        assert_eq!(unescape_unicode(r"\n"), r"\n");
    }

    #[test]
    fn test_parse_sudachi_csv_unicode_unescape() {
        let mut f = NamedTempFile::new().unwrap();
        // Kaomoji entry with \u0028 and \u0029 escapes in column 4
        f.write_all(
            b"col0,col1,col2,5000,\\u0028*\\u0029,col5,col6,col7,col8,col9,col10,\xE3\x82\xAD\xE3\x82\xB4\xE3\x82\xA6\n",
        )
        .unwrap();
        f.flush().unwrap();

        let map = parse_sudachi_csv(f.path()).unwrap();
        let surfaces = &map["キゴウ"];
        // \u0028 → (, \u0029 → )
        assert!(
            surfaces.contains_key("(*)"),
            "Expected (*), got: {:?}",
            surfaces.keys().collect::<Vec<_>>()
        );
    }

    fn create_test_mozc_tsv() -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        let tsv = "# Comment line\n\
                    きょう\t今日\t名詞\t\n\
                    きょう\t京\t名詞\t\n\
                    きょうと\t京都\t名詞\tcity\n\
                    とうきょう\t東京\t名詞\tcapital\n";
        f.write_all(tsv.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_build_from_mozc_tsv() {
        let tsv_file = create_test_mozc_tsv();
        let dict = Dictionary::build_from_mozc_tsv(tsv_file.path()).unwrap();

        // Should have 3 readings
        assert_eq!(dict.entries.len(), 3);

        // Check exact match
        let result = dict.exact_match_search("きょう").unwrap();
        assert_eq!(result.candidates.len(), 2);
        assert_eq!(result.candidates[0].surface, "今日");
        assert_eq!(result.candidates[1].surface, "京");
        assert!((result.candidates[0].score - 0.0).abs() < f32::EPSILON);

        let result = dict.exact_match_search("きょうと").unwrap();
        assert_eq!(result.candidates.len(), 1);
        assert_eq!(result.candidates[0].surface, "京都");
    }

    #[test]
    fn test_build_from_mozc_tsv_skips_invalid() {
        let mut f = NamedTempFile::new().unwrap();
        let tsv = "# Comment\n\
                    \n\
                    single_column\n\
                    \t\t名詞\t\n\
                    きょう\t今日\t名詞\t\n";
        f.write_all(tsv.as_bytes()).unwrap();
        f.flush().unwrap();

        let dict = Dictionary::build_from_mozc_tsv(f.path()).unwrap();
        assert_eq!(dict.entries.len(), 1);
        assert_eq!(
            dict.exact_match_search("きょう").unwrap().candidates[0].surface,
            "今日"
        );
    }

    #[test]
    fn test_build_from_mozc_tsv_dedup_surfaces() {
        let mut f = NamedTempFile::new().unwrap();
        let tsv = "きょう\t今日\t名詞\t\n\
                    きょう\t今日\t副詞\t\n\
                    きょう\t京\t名詞\t\n";
        f.write_all(tsv.as_bytes()).unwrap();
        f.flush().unwrap();

        let dict = Dictionary::build_from_mozc_tsv(f.path()).unwrap();
        let result = dict.exact_match_search("きょう").unwrap();
        // "今日" should appear only once (deduplicated)
        assert_eq!(result.candidates.len(), 2);
        assert_eq!(result.candidates[0].surface, "今日");
        assert_eq!(result.candidates[1].surface, "京");
    }

    #[test]
    fn test_load_auto_binary() {
        let json_file = create_test_json();
        let dict = Dictionary::build_from_json(json_file.path()).unwrap();

        let bin_file = NamedTempFile::new().unwrap();
        dict.save(bin_file.path()).unwrap();

        // load_auto should detect KRKN magic and load as binary
        let loaded = Dictionary::load_auto(bin_file.path()).unwrap();
        let result = loaded.exact_match_search("きょう").unwrap();
        assert_eq!(result.candidates.len(), 2);
    }

    #[test]
    fn test_load_auto_ms_ime_utf16le_text() {
        let text = "!Microsoft IME Dictionary Tool\r\n\
                    !Version: 98.00\r\n\
                    きょう\t今日\t名詞\r\n\
                    かしこ\t賢\t名詞\r\n";
        let mut bytes = vec![0xFF, 0xFE];
        for unit in text.encode_utf16() {
            bytes.extend_from_slice(&unit.to_le_bytes());
        }
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&bytes).unwrap();
        f.flush().unwrap();

        let dict = Dictionary::load_auto(f.path()).unwrap();
        assert_eq!(dict.entries.len(), 2);
        assert_eq!(
            dict.exact_match_search("きょう").unwrap().candidates[0].surface,
            "今日"
        );
        assert!(dict.exact_match_search("かしこ").is_some());
    }

    #[test]
    fn test_build_from_mozc_tsv_utf16be_and_utf8_bom() {
        let text = "!header\nきょう\t今日\t名詞\n";
        let mut be = vec![0xFE, 0xFF];
        for unit in text.encode_utf16() {
            be.extend_from_slice(&unit.to_be_bytes());
        }
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&be).unwrap();
        f.flush().unwrap();
        let dict = Dictionary::build_from_mozc_tsv(f.path()).unwrap();
        assert_eq!(dict.entries.len(), 1);

        let mut bom8 = vec![0xEF, 0xBB, 0xBF];
        bom8.extend_from_slice(text.as_bytes());
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&bom8).unwrap();
        f.flush().unwrap();
        let dict = Dictionary::build_from_mozc_tsv(f.path()).unwrap();
        assert_eq!(dict.entries.len(), 1);
        assert!(dict.exact_match_search("きょう").is_some());
    }

    #[test]
    fn test_load_auto_mozc_tsv() {
        let tsv_file = create_test_mozc_tsv();

        // load_auto should detect non-KRKN and parse as Mozc TSV
        let dict = Dictionary::load_auto(tsv_file.path()).unwrap();
        let result = dict.exact_match_search("きょう").unwrap();
        assert_eq!(result.candidates.len(), 2);
        assert_eq!(result.candidates[0].surface, "今日");
    }

    #[test]
    fn test_merge_dictionaries() {
        // Create two TSV dictionaries
        let mut f1 = NamedTempFile::new().unwrap();
        f1.write_all("きょう\t今日\t名詞\t\nきょうと\t京都\t名詞\t\n".as_bytes())
            .unwrap();
        f1.flush().unwrap();

        let mut f2 = NamedTempFile::new().unwrap();
        f2.write_all("きょう\t教\t名詞\t\nおおさか\t大阪\t名詞\t\n".as_bytes())
            .unwrap();
        f2.flush().unwrap();

        let dict1 = Dictionary::build_from_mozc_tsv(f1.path()).unwrap();
        let dict2 = Dictionary::build_from_mozc_tsv(f2.path()).unwrap();

        let merged = Dictionary::merge(vec![dict1, dict2]).unwrap().unwrap();

        // "きょう" should have candidates from both, dict1 first
        let result = merged.exact_match_search("きょう").unwrap();
        assert_eq!(result.candidates.len(), 2);
        assert_eq!(result.candidates[0].surface, "今日");
        assert_eq!(result.candidates[1].surface, "教");

        // "きょうと" from dict1
        assert!(merged.exact_match_search("きょうと").is_some());
        // "おおさか" from dict2
        assert!(merged.exact_match_search("おおさか").is_some());
    }

    #[test]
    fn test_merge_empty() {
        let result = Dictionary::merge(vec![]).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_sudachi_csv_skips_invalid_lines() {
        let mut f = NamedTempFile::new().unwrap();
        // Too few columns
        f.write_all(b"short,line,only\n").unwrap();
        // Empty line
        f.write_all(b"\n").unwrap();
        // Invalid cost
        f.write_all(b"surface,col1,col2,notanumber,col4,col5,col6,col7,col8,col9,col10,reading\n")
            .unwrap();
        // Valid line
        f.write_all("OK,col1,col2,100,col4,col5,col6,col7,col8,col9,col10,オッケー\n".as_bytes())
            .unwrap();
        f.flush().unwrap();

        let map = parse_sudachi_csv(f.path()).unwrap();
        assert_eq!(map.len(), 1);
        assert!(map.contains_key("オッケー"));
    }
}
