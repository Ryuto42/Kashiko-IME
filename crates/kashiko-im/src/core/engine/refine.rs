//! Background refinement of live conversion.
//!
//! Each keystroke converts greedily, which is what keeps typing fast. Once
//! typing pauses, the chunks are converted again on the worker thread with a
//! beam of [`BEAM_WIDTH`], and the preedit is updated where the beam disagrees.
//! On AJIMEE-Bench this takes live conversion from 168 to 173 of 200, for
//! ~200 ms of a background thread per sentence — too much for every key, cheap
//! once per pause.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use kashiko_engine::KanaKanjiConverter;

use super::conversion::invents_ascii;
use super::{ComposingChunk, EngineAction, EngineResult, InputMethodEngine};
use crate::core::state::InputState;

/// Beam width of the refinement pass: 2 gained 3 of 200, 3 gained 4, 5 no
/// more than 3 at half again the cost.
const BEAM_WIDTH: usize = 3;
/// How long typing has to pause before the worker starts.
const PAUSE: Duration = Duration::from_millis(250);

/// Longest stretch of reading converted in one beam pass. Consecutive chunks
/// are refined together so the beam sees the sentence, not the 30-character
/// pieces live conversion cuts it into.
const MAX_UNIT_CHARS: usize = 80;

pub struct RefineJob {
    converter: Arc<KanaKanjiConverter>,
    /// The reading the job was built for; a different buffer makes it stale.
    buffer: String,
    /// Document context left of the composition.
    base_ctx: String,
    max_ctx: usize,
    /// Every chunk's reading and current conversion.
    chunks: Vec<(String, String)>,
    /// Runs of chunks to refine together, as `start..end` chunk indices.
    units: Vec<std::ops::Range<usize>>,
    cancel: Arc<AtomicBool>,
}

pub struct RefineOutcome {
    buffer: String,
    /// `(chunk range, reading, beam conversion)` for each unit refined.
    results: Vec<(std::ops::Range<usize>, String, String)>,
}

/// Wait out the pause, then convert each unit with the beam, the text before
/// it as left context (refined where it was refined).
pub fn run_refine_job(job: RefineJob) -> RefineOutcome {
    let mut outcome = RefineOutcome {
        buffer: job.buffer,
        results: Vec::new(),
    };
    let step = Duration::from_millis(10);
    let mut waited = Duration::ZERO;
    while waited < PAUSE {
        if job.cancel.load(Ordering::Relaxed) {
            return outcome;
        }
        std::thread::sleep(step);
        waited += step;
    }
    let mut texts: Vec<String> = job.chunks.iter().map(|c| c.1.clone()).collect();
    for unit in job.units {
        if job.cancel.load(Ordering::Relaxed) {
            break;
        }
        let preceding: String = texts[..unit.start].concat();
        let all = format!("{}{preceding}", job.base_ctx);
        let n = all.chars().count();
        let lctx: String = all.chars().skip(n.saturating_sub(job.max_ctx)).collect();
        let reading: String = job.chunks[unit.clone()]
            .iter()
            .map(|c| c.0.as_str())
            .collect();
        let katakana = kashiko_engine::hiragana_to_katakana(&reading);
        if let Some(best) = job
            .converter
            .convert(&katakana, &lctx, BEAM_WIDTH)
            .ok()
            .and_then(|beam| beam.into_iter().next())
        {
            // Later units read this one as context.
            for t in &mut texts[unit.clone()] {
                t.clear();
            }
            texts[unit.start] = best.clone();
            outcome.results.push((unit, reading, best));
        }
    }
    outcome
}

impl InputMethodEngine {
    /// Hand the refinement of the current live conversion to the worker, once
    /// per change. `None` when there is nothing to refine or a job is out.
    pub fn take_refine_job(&mut self) -> Option<RefineJob> {
        if !self.refine_wanted
            || self.refine_running
            || !matches!(self.state, InputState::Composing { .. })
            || self.live.text.is_empty()
        {
            return None;
        }
        self.refine_wanted = false;
        let converter = self.converters.kanji.clone()?;
        let units = refine_units(&self.chunks);
        if units.is_empty() {
            return None;
        }
        self.refine_cancel = Arc::new(AtomicBool::new(false));
        self.refine_running = true;
        Some(RefineJob {
            converter,
            buffer: self.input_buf.text.clone(),
            base_ctx: self.truncate_context_for_api(),
            max_ctx: self.config.max_api_context_len,
            chunks: self
                .chunks
                .iter()
                .map(|c| (c.reading.clone(), c.converted.clone()))
                .collect(),
            units,
            cancel: self.refine_cancel.clone(),
        })
    }

    /// Stop the running refinement: a key changed what it would refine.
    pub fn cancel_refine(&mut self) {
        self.refine_cancel.store(true, Ordering::Relaxed);
    }

    /// Put the beam's conversions in place of the greedy ones and redraw the
    /// preedit. `None` when the composition has moved on or nothing changed.
    ///
    /// A refined unit replaces its chunks with new ones cut at the conversion's
    /// own phrase boundaries, so typing on reuses them like any other chunk.
    pub fn apply_refine_outcome(&mut self, outcome: RefineOutcome) -> Option<EngineResult> {
        self.refine_running = false;
        if !matches!(self.state, InputState::Composing { .. })
            || self.live.text.is_empty()
            || self.input_buf.text != outcome.buffer
        {
            return None;
        }
        let chunk_len = self.config.composing_chunk_len.max(1);
        let mut changed = false;
        // Back to front, so earlier ranges stay valid as chunks are replaced.
        for (range, reading, text) in outcome.results.into_iter().rev() {
            let Some(chunks) = self.chunks.get(range.clone()) else {
                continue;
            };
            if chunks
                .iter()
                .map(|c| c.reading.as_str())
                .collect::<String>()
                != reading
            {
                continue;
            }
            let current: String = chunks.iter().map(|c| c.converted.as_str()).collect();
            let acceptable = text != current
                && !kashiko_engine::has_unrenderable_char(&text)
                && !invents_ascii(&reading, &text)
                && self
                    .learning
                    .as_ref()
                    .is_none_or(|cache| !cache.is_suppressed(&reading, &text));
            let pieces = if acceptable {
                split_into_chunks(&reading, &text, chunk_len)
            } else {
                None
            };
            match pieces {
                Some(pieces) => {
                    self.chunks.splice(range, pieces);
                    changed = true;
                }
                None => {
                    for chunk in &mut self.chunks[range] {
                        chunk.refined = true;
                    }
                }
            }
        }
        if !changed {
            return None;
        }
        self.live.text = self.chunks.iter().map(|c| c.converted.as_str()).collect();
        let preedit = self.set_composing_state();
        Some(EngineResult::consumed().with_action(EngineAction::UpdatePreedit(preedit)))
    }
}

/// The runs of chunks to refine together. Punctuation between unsettled
/// chunks joins them: the clause after a `、` is read with the one before it.
/// Anything else settled (a learned form, digits, latin) ends a run.
fn refine_units(chunks: &[ComposingChunk]) -> Vec<std::ops::Range<usize>> {
    let joinable = |c: &ComposingChunk| {
        !c.refined
            || c.reading
                .chars()
                .all(|ch| matches!(ch, '、' | '。' | '，' | '．' | '・' | '！' | '？'))
    };
    let mut units: Vec<std::ops::Range<usize>> = Vec::new();
    let mut unit_chars = 0usize;
    for (i, chunk) in chunks.iter().enumerate() {
        if !joinable(chunk) {
            continue;
        }
        let len = chunk.reading.chars().count();
        match units.last_mut() {
            Some(unit) if unit.end == i && unit_chars + len <= MAX_UNIT_CHARS => {
                unit.end = i + 1;
                unit_chars += len;
            }
            _ => {
                units.push(i..i + 1);
                unit_chars = len;
            }
        }
    }
    units.retain(|unit| chunks[unit.clone()].iter().any(|c| !c.refined));
    units
}

/// `text` (the conversion of `reading`) cut into settled chunks of at most
/// `max` reading chars at its phrase boundaries; a phrase longer than that is a
/// chunk of its own. `None` when the two can't be aligned.
fn split_into_chunks(reading: &str, text: &str, max: usize) -> Option<Vec<ComposingChunk>> {
    let spans = kashiko_engine::segment::align(reading, text)?;
    let r: Vec<char> = reading.chars().collect();
    let c: Vec<char> = text.chars().collect();
    let mut out: Vec<ComposingChunk> = Vec::new();
    let (mut from_r, mut from_c) = (0usize, 0usize);
    for (i, span) in spans.iter().enumerate() {
        let last = i + 1 == spans.len();
        let next_len = spans
            .get(i + 1)
            .map_or(0, |s| s.reading_end - s.reading_start);
        if last || span.reading_end - from_r + next_len > max {
            out.push(ComposingChunk {
                reading: r[from_r..span.reading_end].iter().collect(),
                converted: c[from_c..span.converted_end].iter().collect(),
                refined: true,
            });
            (from_r, from_c) = (span.reading_end, span.converted_end);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pieces(reading: &str, text: &str, max: usize) -> Vec<(String, String)> {
        split_into_chunks(reading, text, max)
            .unwrap()
            .into_iter()
            .map(|c| {
                assert!(c.refined);
                (c.reading, c.converted)
            })
            .collect()
    }

    #[test]
    fn a_refined_unit_is_cut_at_phrase_boundaries() {
        let p = pieces("きょうのてんきははれです", "今日の天気は晴れです", 8);
        assert_eq!(
            p,
            vec![
                ("きょうのてんきは".into(), "今日の天気は".into()),
                ("はれです".into(), "晴れです".into()),
            ]
        );
    }

    #[test]
    fn a_phrase_longer_than_a_chunk_stays_whole() {
        let p = pieces("きょうのてんきははれです", "今日の天気は晴れです", 2);
        let readings: Vec<&str> = p.iter().map(|(r, _)| r.as_str()).collect();
        assert_eq!(readings, vec!["きょうの", "てんきは", "はれです"]);
    }

    #[test]
    fn an_unaligned_conversion_is_not_split() {
        assert!(split_into_chunks("きょうは", "今日が", 30).is_none());
    }

    #[test]
    fn units_join_across_punctuation_but_not_settled_chunks() {
        let chunk = |r: &str, refined| ComposingChunk {
            reading: r.into(),
            converted: r.into(),
            refined,
        };
        let chunks = vec![
            chunk("ことで", false),
            chunk("、", true),
            chunk("おんじょうを", false),
            chunk("123", true),
            chunk("とりもどし", false),
            chunk("はれ", true),
            chunk("。", true),
        ];
        assert_eq!(refine_units(&chunks), vec![0..3, 4..5]);
    }
}
