//! Backend interface for kanji conversion using llama.cpp

use super::error::KanjiError;
use super::hf_download::{get_tokenizer_path, get_variant_path};
use super::llamacpp::LlamaCppModel;
use super::model_config::{ModelFamily, VariantConfig, registry};
use super::{CONTEXT_TOKEN, INPUT_START_TOKEN, OUTPUT_START_TOKEN};
use crate::kana::hiragana_to_katakana;

type Result<T> = super::error::Result<T>;

/// Configuration for kanji conversion
#[derive(Debug, Clone)]
pub struct ConversionConfig {
    /// Maximum number of new tokens to generate
    pub max_new_tokens: usize,
}

impl Default for ConversionConfig {
    fn default() -> Self {
        Self { max_new_tokens: 50 }
    }
}

/// Build a prompt in jinen format
pub fn build_jinen_prompt(katakana: &str, context: &str) -> String {
    format!(
        "{}{}{}{}{}",
        CONTEXT_TOKEN, context, INPUT_START_TOKEN, katakana, OUTPUT_START_TOKEN
    )
}

/// Clean model output by trimming whitespace.
///
/// Special tokens (BOS/EOS) are handled at the decode level via
/// `skip_special_tokens` rather than string replacement.
pub fn clean_model_output(text: &str) -> String {
    text.trim().to_string()
}

/// Inference backend configuration (llama.cpp GGUF format with external tokenizer)
#[derive(Debug, Clone)]
pub struct Backend {
    gguf_path: String,
    tokenizer_json_path: String,
    /// Display name for the model (variant id for registry models, "custom" for GGUF paths)
    display_name: String,
}

impl Backend {
    /// Create a backend from a `(ModelFamily, VariantConfig)` pair.
    ///
    /// Downloads the GGUF and the external tokenizer from HuggingFace.
    pub fn from_variant(family: &ModelFamily, variant: &VariantConfig) -> Result<Self> {
        let path = get_variant_path(family, variant)?;
        let tokenizer_path = get_tokenizer_path(family)?;
        Ok(Backend {
            gguf_path: path.to_string_lossy().to_string(),
            tokenizer_json_path: tokenizer_path.to_string_lossy().to_string(),
            display_name: variant.id.clone(),
        })
    }

    /// Create a backend by looking up a variant id in the global registry.
    ///
    /// E.g. `Backend::from_variant_id("jinen-v1-xsmall-q5")`
    pub fn from_variant_id(variant_id: &str) -> Result<Self> {
        let (family, variant) = registry()
            .find_variant(variant_id)
            .ok_or_else(|| KanjiError::UnknownVariant(variant_id.to_string()))?;
        Self::from_variant(family, variant)
    }
}

/// Kanji converter using llama.cpp backend
pub struct KanaKanjiConverter {
    model: LlamaCppModel,
    config: ConversionConfig,
    display_name: String,
}

impl KanaKanjiConverter {
    /// Create a new converter with the specified backend
    pub fn new(backend: Backend) -> Result<Self> {
        Self::with_config(backend, ConversionConfig::default())
    }

    /// Create a new converter with the specified backend and configuration
    pub fn with_config(backend: Backend, config: ConversionConfig) -> Result<Self> {
        let model = LlamaCppModel::from_file(&backend.gguf_path, &backend.tokenizer_json_path)?;
        Ok(KanaKanjiConverter {
            model,
            config,
            display_name: backend.display_name,
        })
    }

    /// Set the number of threads for inference (0 = default).
    pub fn set_n_threads(&mut self, n: u32) {
        self.model.set_n_threads(n);
    }

    /// Convert hiragana to kanji candidates
    ///
    /// # Arguments
    /// * `reading` - Input reading in hiragana
    /// * `context` - Left context (previously converted text)
    /// * `num_candidates` - Number of candidates to generate
    ///
    /// # Returns
    /// Vector of conversion candidates
    pub fn convert(
        &self,
        reading: &str,
        context: &str,
        num_candidates: usize,
    ) -> Result<Vec<String>> {
        // Convert hiragana to katakana (model expects katakana input)
        let katakana = hiragana_to_katakana(reading);

        // Build prompt in jinen format
        let prompt = build_jinen_prompt(&katakana, context);

        // Tokenize
        let tokens = self.model.tokenize(&prompt)?;
        let eos = Some(self.model.eos_token_id().0);

        let mut candidates = Vec::with_capacity(num_candidates);

        if num_candidates == 1 {
            // Single candidate: use greedy decoding (faster)
            let output_tokens = self
                .model
                .generate(&tokens, self.config.max_new_tokens, eos)?;
            let generated = &output_tokens[tokens.len()..];
            let text = self.model.decode(generated, true)?;
            let clean = clean_model_output(&text);

            if !clean.is_empty() {
                candidates.push(clean);
            }
        } else {
            // Multiple candidates: use beam search
            let results = self.model.generate_beam_search(
                &tokens,
                self.config.max_new_tokens,
                eos,
                num_candidates,
            )?;

            for (output_tokens, _score) in results {
                let text = self.model.decode(&output_tokens, true)?;
                let clean = clean_model_output(&text);

                if !clean.is_empty() && !candidates.contains(&clean) {
                    candidates.push(clean);
                }
            }
        }

        // If no candidates, return the original reading
        if candidates.is_empty() {
            candidates.push(reading.to_string());
        }

        Ok(candidates)
    }

    /// Convert for live display, retrying with a small beam only when greedy
    /// decoding produced something the model itself rates as unlikely.
    ///
    /// Greedy is myopic: it commits to the best token at each step and can walk
    /// into a bad sentence that beam search would avoid. Running a beam on every
    /// keystroke is far too slow, but the greedy pass already reports its own
    /// log-probability for free (see `generate_greedy_scored`), so its quality
    /// can be checked at no cost and the beam spent only on the rare bad case.
    ///
    /// Returns the conversion, or the reading itself when nothing usable came
    /// out (matching `convert`'s fallback).
    pub fn convert_live(&self, reading: &str, context: &str) -> Result<String> {
        let katakana = hiragana_to_katakana(reading);
        let prompt = build_jinen_prompt(&katakana, context);
        let tokens = self.model.tokenize(&prompt)?;
        let eos = Some(self.model.eos_token_id().0);

        let (out_tokens, log_prob) =
            self.model
                .generate_greedy_scored(&tokens, self.config.max_new_tokens, eos)?;
        let generated = &out_tokens[tokens.len()..];
        let greedy = clean_model_output(&self.model.decode(generated, true)?);

        // Normalize per output character so the threshold means the same thing
        // for a two-character word and a whole clause.
        let n_chars = greedy.chars().count();
        let greedy_nll = if n_chars == 0 {
            f32::INFINITY
        } else {
            -log_prob / n_chars as f32
        };
        if greedy_nll <= Self::LIVE_RESCUE_NLL_THRESHOLD {
            return Ok(if greedy.is_empty() {
                reading.to_string()
            } else {
                greedy
            });
        }

        // Greedy looks unnatural — spend a narrow beam and keep its best.
        // Beam scores are cumulative log-probabilities, highest first.
        let beams = self.model.generate_beam_search(
            &tokens,
            self.config.max_new_tokens,
            eos,
            Self::LIVE_RESCUE_BEAM_WIDTH,
        )?;
        // Beam results hold only the generated tokens — unlike `generate`, which
        // returns the prompt followed by them — so they are decoded whole.
        let rescued = beams
            .iter()
            .filter_map(|(toks, score)| {
                let text = clean_model_output(&self.model.decode(toks, true).ok()?);
                (!text.is_empty()).then(|| (text, *score))
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(text, _)| text);

        Ok(match rescued {
            Some(text) => text,
            None if greedy.is_empty() => reading.to_string(),
            None => greedy,
        })
    }

    /// Per-character NLL above which the live greedy conversion is considered
    /// bad enough to be worth a beam-search retry.
    pub const LIVE_RESCUE_NLL_THRESHOLD: f32 = 1.5;

    /// Beam width for that retry — narrow, since it runs on the typing path.
    pub const LIVE_RESCUE_BEAM_WIDTH: usize = 3;

    /// NLL threshold above which a conversion reads as unnatural Japanese.
    /// Per-character, so it is comparable across candidate lengths.
    pub const TYPO_NLL_THRESHOLD: f32 = 1.5;

    /// Check if a (reading, surface) pair is likely a typo based on LLM NLL score
    pub fn is_typo(&self, reading: &str, surface: &str, context: &str) -> Result<bool> {
        let mut scorer = super::llamacpp::NllScorer::new(&self.model, 512)?;
        let katakana = hiragana_to_katakana(reading);
        let nll = scorer.compute_nll(&katakana, surface, context)?;
        Ok(nll > Self::TYPO_NLL_THRESHOLD)
    }

    /// Score each candidate as per-character negative log-likelihood given
    /// `reading` and the surrounding `context` — lower is more natural.
    ///
    /// One scorer for the whole list, so the shared prompt stays in the KV
    /// cache across candidates instead of being re-evaluated per candidate.
    /// Returns one score per input candidate, in the same order;
    /// `None` marks a candidate the model could not score (its position is
    /// preserved so callers can zip the two lists).
    pub fn score_candidates(
        &self,
        reading: &str,
        context: &str,
        candidates: &[String],
    ) -> Result<Vec<Option<f32>>> {
        Ok(self
            .score_candidates_detailed(reading, context, candidates)?
            .into_iter()
            .map(|d| d.map(|d| d.mean))
            .collect())
    }

    /// As [`Self::score_candidates`], but keeping the worst single character of
    /// each candidate alongside its average.
    pub fn score_candidates_detailed(
        &self,
        reading: &str,
        context: &str,
        candidates: &[String],
    ) -> Result<Vec<Option<super::llamacpp::NllDetail>>> {
        let mut scorer = super::llamacpp::NllScorer::new(&self.model, 512)?;
        let katakana = hiragana_to_katakana(reading);
        Ok(candidates
            .iter()
            .map(|cand| scorer.compute_nll_detail(&katakana, cand, context).ok())
            .collect())
    }

    /// Filter out candidates that are likely typos or exceed maximum length.
    pub fn filter_candidates(
        &self,
        reading: &str,
        context: &str,
        candidates: Vec<String>,
    ) -> Result<Vec<String>> {
        let mut scorer = super::llamacpp::NllScorer::new(&self.model, 512)?;
        let katakana = hiragana_to_katakana(reading);

        let mut filtered = Vec::new();
        for cand in candidates {
            if cand.chars().count() >= 12 {
                continue;
            }
            match scorer.compute_nll(&katakana, &cand, context) {
                Ok(nll) if nll > Self::TYPO_NLL_THRESHOLD => {}
                // Unscoreable candidates are kept: a scoring failure is not
                // evidence against the candidate.
                _ => filtered.push(cand),
            }
        }
        Ok(filtered)
    }

    /// Get a human-readable model name for display
    pub fn model_display_name(&self) -> &str {
        &self.display_name
    }

    /// Count only the input (reading) tokens, excluding context and special tokens
    pub fn count_input_tokens(&self, reading: &str) -> Result<usize> {
        let katakana = hiragana_to_katakana(reading);
        let tokens = self.model.tokenize(&katakana)?;
        Ok(tokens.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]

    fn test_default_model_conversion() {
        let backend =
            Backend::from_variant_id("jinen-v1-small-q5").expect("Failed to load default model");
        let converter = KanaKanjiConverter::new(backend).expect("Failed to create converter");

        let result = converter.convert("かんじ", "", 1);
        assert!(result.is_ok(), "Conversion failed: {:?}", result.err());

        let candidates = result.unwrap();
        assert!(!candidates.is_empty(), "No candidates returned");

        let output = &candidates[0];
        assert!(
            !output.contains("ã"),
            "Output contains mojibake: '{}'",
            output
        );
    }

    #[test]

    fn test_xsmall_special_tokens() {
        use super::super::hf_download::{get_path_by_id, get_tokenizer_path_by_id};
        use super::super::{CONTEXT_TOKEN, INPUT_START_TOKEN, OUTPUT_START_TOKEN};
        let path = get_path_by_id("jinen-v1-xsmall-q5").expect("Failed to download GGUF");
        let tok_path =
            get_tokenizer_path_by_id("jinen-v1-xsmall-q5").expect("Failed to download tokenizer");
        let model = LlamaCppModel::from_file(&path, &tok_path).expect("Failed to load model");

        let prompt = build_jinen_prompt("テスト", "");
        let tokens = model.tokenize(&prompt).expect("Failed to tokenize");

        let mut found_context = false;
        let mut found_input_start = false;
        let mut found_output_start = false;

        for token in &tokens {
            let display = model.decode_token_for_display(*token);
            if display.contains(CONTEXT_TOKEN) {
                found_context = true;
            }
            if display.contains(INPUT_START_TOKEN) {
                found_input_start = true;
            }
            if display.contains(OUTPUT_START_TOKEN) {
                found_output_start = true;
            }
        }

        assert!(found_context, "CONTEXT token (U+EE02) not found");
        assert!(found_input_start, "INPUT_START token (U+EE00) not found");
        assert!(found_output_start, "OUTPUT_START token (U+EE01) not found");
    }

    #[test]

    fn test_xsmall_conversion() {
        let backend =
            Backend::from_variant_id("jinen-v1-xsmall-q5").expect("Failed to download GGUF");
        let converter = KanaKanjiConverter::new(backend).expect("Failed to create converter");

        let result = converter.convert("かんじ", "", 1);
        assert!(result.is_ok(), "Conversion failed: {:?}", result.err());

        let candidates = result.unwrap();
        assert!(!candidates.is_empty(), "No candidates returned");

        let output = &candidates[0];
        assert!(
            !output.contains("ã"),
            "Output contains mojibake (GPT-2 byte encoding leak): '{}'",
            output
        );
    }
}

#[cfg(test)]
mod thread_safety {
    /// The deep search runs on a worker thread sharing this converter with the
    /// main loop, so both must be able to hold it at once. Nothing here mutates
    /// the model: every inference call builds its own llama.cpp context.
    #[test]
    fn the_converter_can_be_shared_across_threads() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<super::KanaKanjiConverter>();
    }
}
