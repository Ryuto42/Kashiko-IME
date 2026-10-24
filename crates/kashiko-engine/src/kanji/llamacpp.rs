//! llama.cpp based GGUF inference for kanji conversion
//!
//! This module provides an alternative to Candle's GGUF implementation using
//! llama.cpp's optimized inference engine via the llama-cpp-2 crate.
//!
//! Enable with the `llamacpp` feature flag.

use super::error::KanjiError;
type Result<T> = super::error::Result<T>;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::OnceLock;

/// Global llama.cpp backend (can only be initialized once)
static LLAMA_BACKEND: OnceLock<std::result::Result<LlamaBackend, String>> = OnceLock::new();

/// Get or initialize the global llama.cpp backend
fn get_backend() -> Result<&'static LlamaBackend> {
    let result = LLAMA_BACKEND.get_or_init(|| {
        let mut backend = LlamaBackend::init().map_err(|e| e.to_string())?;
        backend.void_logs();
        Ok(backend)
    });
    match result {
        Ok(backend) => Ok(backend),
        Err(e) => Err(KanjiError::ModelLoad(
            format!("Failed to initialize llama.cpp backend: {}", e).into(),
        )),
    }
}

/// Convert bytes to hex display format for partial UTF-8 sequences
/// Reject token sequences that can't fit in the context before they reach
/// llama.cpp, which may abort the whole process instead of returning an error.
fn ensure_fits(n_tokens: usize, n_ctx: u32) -> Result<()> {
    if n_tokens == 0 || n_tokens > n_ctx as usize {
        return Err(KanjiError::Inference(
            format!("{n_tokens} tokens do not fit in n_ctx {n_ctx}").into(),
        ));
    }
    Ok(())
}

fn bytes_to_hex_display(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("<{:02X}>", b)).collect()
}

/// Load and configure an external HuggingFace tokenizer from a `tokenizer.json` file.
fn load_tokenizer<P: AsRef<Path>>(path: P) -> Result<tokenizers::Tokenizer> {
    let mut tokenizer =
        tokenizers::Tokenizer::from_file(path.as_ref()).map_err(KanjiError::TokenizerLoad)?;
    // Disable padding and truncation — we handle sequence length ourselves
    // and padding tokens would corrupt the model input.
    tokenizer.with_padding(None);
    tokenizer.with_truncation(None).ok();
    Ok(tokenizer)
}

/// A beam candidate with generated tokens and cumulative score
#[derive(Clone)]
struct BeamState {
    tokens: Vec<LlamaToken>,
    score: f32,
}

/// The context live conversion keeps between keystrokes, and the tokens its
/// KV cache holds, so a keystroke only evaluates what changed since the last.
#[derive(Default)]
struct LiveCache {
    /// Borrows `LlamaCppModel::model`; the `'static` only hides that from the
    /// borrow checker. It is declared before `model` so it is dropped first,
    /// and it never leaves the owning `LlamaCppModel`.
    ctx: Option<llama_cpp_2::context::LlamaContext<'static>>,
    kv: Vec<LlamaToken>,
}

// SAFETY: the context is only touched under `LlamaCppModel::live`'s mutex.
unsafe impl Send for LiveCache {}

/// llama.cpp based GPT-2 model for GGUF inference
pub struct LlamaCppModel {
    live: std::sync::Mutex<LiveCache>,
    model: LlamaModel,
    n_ctx: u32,
    /// External HuggingFace tokenizer (always required).
    /// `tokenize()` and `decode()` use this instead of llama.cpp's built-in tokenizer.
    external_tokenizer: tokenizers::Tokenizer,
    /// Number of threads for inference (0 = use llama.cpp default)
    n_threads: u32,
    /// NFKC-normalize text before tokenizing (see `ModelFamily::nfkc_prompt`).
    nfkc: bool,
}

impl LlamaCppModel {
    /// Load a GGUF model using llama.cpp with an external tokenizer.
    ///
    /// GPT-2 models use CPU only (Metal has issues with GPT-2).
    pub fn from_file<P: AsRef<Path>, T: AsRef<Path>>(path: P, tokenizer_json: T) -> Result<Self> {
        let backend = get_backend()?;

        // GPT-2 has Metal issues, use CPU
        let model_params = LlamaModelParams::default().with_n_gpu_layers(0);

        let model = LlamaModel::load_from_file(backend, path.as_ref(), &model_params)
            .map_err(|e| KanjiError::ModelLoad(e.into()))?;
        let external_tokenizer = load_tokenizer(tokenizer_json)?;

        Ok(Self {
            live: Default::default(),
            model,
            n_ctx: 256,
            external_tokenizer,
            n_threads: 0,
            nfkc: false,
        })
    }

    /// Load a GGUF model with a pre-tokenizer type override.
    ///
    /// Some models use custom pre-tokenizer types (e.g., `gpt2-small-japanese-char`)
    /// that llama.cpp doesn't recognize. This method overrides the `tokenizer.ggml.pre`
    /// metadata key to a compatible type before loading.
    pub fn from_file_with_pre_tokenizer_override<P: AsRef<Path>, T: AsRef<Path>>(
        path: P,
        tokenizer_json: T,
        pre_tokenizer: &str,
    ) -> Result<Self> {
        use llama_cpp_2::model::params::kv_overrides::ParamOverrideValue;
        use std::ffi::CString;
        use std::pin::pin;

        let backend = get_backend()?;

        let mut params = pin!(LlamaModelParams::default().with_n_gpu_layers(0));

        let key =
            CString::new("tokenizer.ggml.pre").map_err(|e| KanjiError::ModelLoad(e.into()))?;
        let mut str_value: [std::os::raw::c_char; 128] = [0; 128];
        for (i, &byte) in pre_tokenizer.as_bytes().iter().enumerate() {
            if i >= 127 {
                break;
            }
            str_value[i] = byte as std::os::raw::c_char;
        }
        params
            .as_mut()
            .append_kv_override(&key, ParamOverrideValue::Str(str_value));

        let model = LlamaModel::load_from_file(backend, path.as_ref(), &params)
            .map_err(|e| KanjiError::ModelLoad(e.into()))?;
        let external_tokenizer = load_tokenizer(tokenizer_json)?;

        Ok(Self {
            live: Default::default(),
            model,
            n_ctx: 256,
            external_tokenizer,
            n_threads: 0,
            nfkc: false,
        })
    }

    /// Load a GGUF model with explicit context window size
    pub fn from_file_with_n_ctx<P: AsRef<Path>, T: AsRef<Path>>(
        path: P,
        tokenizer_json: T,
        n_ctx: u32,
    ) -> Result<Self> {
        let backend = get_backend()?;

        let model_params = LlamaModelParams::default().with_n_gpu_layers(0);

        let model = LlamaModel::load_from_file(backend, path.as_ref(), &model_params)
            .map_err(|e| KanjiError::ModelLoad(e.into()))?;
        let external_tokenizer = load_tokenizer(tokenizer_json)?;

        Ok(Self {
            live: Default::default(),
            model,
            n_ctx,
            external_tokenizer,
            n_threads: 0,
            nfkc: false,
        })
    }

    /// Set the number of threads for inference.
    /// 0 means use llama.cpp default (typically all cores).
    /// Free the context live conversion keeps between keystrokes (its KV cache
    /// and compute buffers); the next conversion creates it again.
    pub fn release_live_cache(&self) {
        *self.live.lock().unwrap_or_else(|e| e.into_inner()) = LiveCache::default();
    }

    pub fn set_n_threads(&mut self, n: u32) {
        self.n_threads = n;
        *self.live.get_mut().unwrap_or_else(|e| e.into_inner()) = LiveCache::default();
    }

    pub fn set_nfkc(&mut self, nfkc: bool) {
        self.nfkc = nfkc;
    }

    pub fn nfkc(&self) -> bool {
        self.nfkc
    }

    /// Build LlamaContextParams with configured n_threads
    fn context_params(&self) -> LlamaContextParams {
        let params = LlamaContextParams::default().with_n_ctx(Some(
            NonZeroU32::new(self.n_ctx).expect("n_ctx must be non-zero"),
        ));
        if self.n_threads > 0 {
            params
                .with_n_threads(self.n_threads as i32)
                .with_n_threads_batch(self.n_threads as i32)
        } else {
            params
        }
    }

    /// Tokenize a string using the external tokenizer
    pub fn tokenize(&self, text: &str) -> Result<Vec<LlamaToken>> {
        let normalized;
        let text = if self.nfkc {
            normalized = crate::kana::normalize_nfkc(text);
            normalized.as_str()
        } else {
            text
        };
        let encoding = self
            .external_tokenizer
            .encode(text, false)
            .map_err(KanjiError::Inference)?;
        let tokens: Vec<LlamaToken> = encoding
            .get_ids()
            .iter()
            .map(|&id| LlamaToken(id as i32))
            .collect();
        Ok(tokens)
    }

    /// Decode tokens to string using the external tokenizer
    ///
    /// When `skip_special_tokens` is true, special tokens (BOS, EOS, EOG) are
    /// excluded from the output.
    pub fn decode(&self, tokens: &[LlamaToken], skip_special_tokens: bool) -> Result<String> {
        let ids: Vec<u32> = tokens.iter().map(|t| t.0 as u32).collect();
        let text = self
            .external_tokenizer
            .decode(&ids, skip_special_tokens)
            .map_err(KanjiError::Inference)?;
        Ok(text)
    }

    /// Decode a single token for display purposes.
    ///
    /// For byte-level BPE tokens that represent partial UTF-8 sequences,
    /// this returns a hex representation like `<0xE3>` instead of replacement characters.
    pub fn decode_token_for_display(&self, token: LlamaToken) -> String {
        match self.model.token_to_piece_bytes(token, 32, true, None) {
            Ok(bytes) => {
                if let Ok(s) = std::str::from_utf8(&bytes) {
                    // Valid UTF-8, return as-is (escape control chars)
                    if s.chars().all(|c| !c.is_control() || c == ' ' || c == '\n') {
                        s.to_string()
                    } else {
                        // Has control characters, show hex
                        bytes_to_hex_display(&bytes)
                    }
                } else {
                    // Invalid UTF-8 (partial sequence), show hex
                    bytes_to_hex_display(&bytes)
                }
            }
            Err(_) => format!("<{}>", token.0),
        }
    }

    /// Generate tokens with greedy decoding
    pub fn generate(
        &self,
        input_tokens: &[LlamaToken],
        max_new_tokens: usize,
        eos_token_id: Option<i32>,
    ) -> Result<Vec<LlamaToken>> {
        self.generate_with_sampler(
            input_tokens,
            max_new_tokens,
            eos_token_id,
            LlamaSampler::greedy(),
        )
        .map(|(tokens, _score)| tokens)
    }

    /// Greedy generation that also reports how likely the model thought its own
    /// output was: the summed log-probability of the generated tokens.
    ///
    /// Free to obtain (see `generate_with_sampler`), which is what makes it
    /// usable on the per-keystroke live-conversion path: the caller can judge
    /// the result's quality without spending a second inference pass on it.
    pub fn generate_greedy_scored(
        &self,
        input_tokens: &[LlamaToken],
        max_new_tokens: usize,
        eos_token_id: Option<i32>,
    ) -> Result<(Vec<LlamaToken>, f32)> {
        self.generate_with_sampler(
            input_tokens,
            max_new_tokens,
            eos_token_id,
            LlamaSampler::greedy(),
        )
    }

    /// As [`Self::generate_greedy_scored`], with `draft` as a guess of the
    /// output (the previous keystroke's conversion). The whole draft is checked
    /// in one batch; the longest prefix greedy decoding agrees with is taken
    /// without generating it token by token, and decoding goes on from there.
    /// The result is exactly what plain greedy decoding produces.
    pub fn generate_greedy_scored_draft(
        &self,
        input_tokens: &[LlamaToken],
        max_new_tokens: usize,
        eos_token_id: Option<i32>,
        draft: &[LlamaToken],
    ) -> Result<(Vec<LlamaToken>, f32)> {
        let mut cache = self.live.lock().unwrap_or_else(|e| e.into_inner());
        let result = self.greedy_draft_cached(
            &mut cache,
            input_tokens,
            max_new_tokens,
            eos_token_id,
            draft,
        );
        if result.is_err() {
            *cache = LiveCache::default();
        }
        result
    }

    fn greedy_draft_cached(
        &self,
        cache: &mut LiveCache,
        input_tokens: &[LlamaToken],
        max_new_tokens: usize,
        eos_token_id: Option<i32>,
        draft: &[LlamaToken],
    ) -> Result<(Vec<LlamaToken>, f32)> {
        ensure_fits(input_tokens.len() + 1, self.n_ctx)?;
        let max_new_tokens = max_new_tokens.min(self.n_ctx as usize - input_tokens.len());
        let draft = &draft[..draft.len().min(max_new_tokens.saturating_sub(1))];
        let p = input_tokens.len();
        let model_eos = self.model.token_eos();
        let is_end = |t: LlamaToken| {
            eos_token_id.is_some_and(|eos| t.0 == eos)
                || t == model_eos
                || self.model.is_eog_token(t)
        };
        let argmax = |logits: &[f32]| {
            let best = logits
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| {
                    if x > b.1 { (i, x) } else { b }
                });
            LlamaToken(best.0 as i32)
        };

        if cache.ctx.is_none() {
            let ctx = self
                .model
                .new_context(get_backend()?, self.context_params())
                .map_err(|e| KanjiError::Inference(e.into()))?;
            // SAFETY: see `LiveCache::ctx`.
            cache.ctx = Some(unsafe {
                std::mem::transmute::<
                    llama_cpp_2::context::LlamaContext<'_>,
                    llama_cpp_2::context::LlamaContext<'static>,
                >(ctx)
            });
            cache.kv.clear();
        }
        let ctx = cache.ctx.as_mut().expect("just created");

        // Keep what the cache already holds of prompt + draft. The prompt's
        // last token is always re-evaluated: its logits predict the output.
        let target: Vec<LlamaToken> = input_tokens.iter().chain(draft).copied().collect();
        let common = cache
            .kv
            .iter()
            .zip(&target)
            .take_while(|(a, b)| a == b)
            .count();
        let keep = common.min(p - 1);
        ctx.clear_kv_cache_seq(Some(0), Some(keep as u32), None)
            .map_err(|e| KanjiError::Inference(e.into()))?;
        cache.kv.truncate(keep);
        let mut batch = LlamaBatch::new(target.len() - keep, 1);
        for (offset, token) in target[keep..].iter().enumerate() {
            let pos = keep + offset;
            batch
                .add(*token, pos as i32, &[0], pos + 1 >= p)
                .map_err(|e| KanjiError::Inference(e.into()))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| KanjiError::Inference(e.into()))?;
        cache.kv.extend_from_slice(&target[keep..]);

        let mut generated = input_tokens.to_vec();
        let mut total_log_prob = 0.0f32;
        // Logits predicting output token `i` sit at position p - 1 + i.
        let mut next = None;
        for i in 0..=draft.len() {
            let logits = ctx.get_logits_ith((p - 1 + i - keep) as i32);
            let token = argmax(logits);
            if is_end(token) {
                return Ok((generated, total_log_prob));
            }
            total_log_prob += log_prob_of(logits, token.0 as usize, logits.len());
            generated.push(token);
            if i == draft.len() || token != draft[i] {
                next = Some(token);
                break;
            }
        }
        // Drop the cached draft past the point of disagreement, then continue
        // as plain greedy decoding from the last chosen token.
        let accepted = generated.len() - p - 1;
        ctx.clear_kv_cache_seq(Some(0), Some((p + accepted) as u32), None)
            .map_err(|e| KanjiError::Inference(e.into()))?;
        cache.kv.truncate(p + accepted);
        let mut token = next.expect("loop always picks a token or returns");
        while generated.len() - p < max_new_tokens {
            batch.clear();
            batch
                .add(token, (generated.len() - 1) as i32, &[0], true)
                .map_err(|e| KanjiError::Inference(e.into()))?;
            ctx.decode(&mut batch)
                .map_err(|e| KanjiError::Inference(e.into()))?;
            cache.kv.push(token);
            let logits = ctx.get_logits_ith(0);
            token = argmax(logits);
            if is_end(token) {
                break;
            }
            total_log_prob += log_prob_of(logits, token.0 as usize, logits.len());
            generated.push(token);
        }
        Ok((generated, total_log_prob))
    }

    /// Generate multiple candidates using true beam search algorithm
    ///
    /// This implements proper beam search that tracks cumulative probabilities
    /// at every step and keeps the globally best beam_size candidates.
    ///
    /// # Arguments
    /// * `input_tokens` - Input token sequence
    /// * `max_new_tokens` - Maximum new tokens to generate per candidate
    /// * `eos_token_id` - Optional EOS token ID to stop generation
    /// * `beam_size` - Number of candidates to keep at each step
    ///
    /// Returns candidates sorted by cumulative probability (highest first)
    pub fn generate_beam_search(
        &self,
        input_tokens: &[LlamaToken],
        max_new_tokens: usize,
        eos_token_id: Option<i32>,
        beam_size: usize,
    ) -> Result<Vec<(Vec<LlamaToken>, f32)>> {
        self.generate_beam_search_impl(input_tokens, max_new_tokens, eos_token_id, beam_size, None)
    }

    /// As [`Self::generate_beam_search`], stopping at the next step once
    /// `cancel` is set.
    pub fn generate_beam_search_cancellable(
        &self,
        input_tokens: &[LlamaToken],
        max_new_tokens: usize,
        eos_token_id: Option<i32>,
        beam_size: usize,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Vec<(Vec<LlamaToken>, f32)>> {
        self.generate_beam_search_impl(
            input_tokens,
            max_new_tokens,
            eos_token_id,
            beam_size,
            cancel,
        )
    }

    /// Generate multiple candidates using depth-1 beam selection followed by greedy decoding
    ///
    /// This is a simplified approach: select top-k initial tokens based on probability,
    /// then generate the rest of each sequence using greedy decoding independently.
    /// This is faster than true beam search but may miss globally optimal candidates.
    ///
    /// # Arguments
    /// * `input_tokens` - Input token sequence
    /// * `max_new_tokens` - Maximum new tokens to generate per candidate
    /// * `eos_token_id` - Optional EOS token ID to stop generation
    /// * `beam_size` - Number of candidates to generate
    ///
    /// Returns candidates sorted by initial token probability (highest first)
    pub fn generate_beam_search_d1_greedy(
        &self,
        input_tokens: &[LlamaToken],
        max_new_tokens: usize,
        eos_token_id: Option<i32>,
        beam_size: usize,
    ) -> Result<Vec<(Vec<LlamaToken>, f32)>> {
        self.generate_beam_search_d1_greedy_batch(
            input_tokens,
            max_new_tokens,
            eos_token_id,
            beam_size,
        )
    }

    /// Generate multiple candidates using batch inference (depth-1 beam + greedy)
    ///
    /// Uses shared KV cache for input tokens across all sequences.
    /// Selects top-k initial tokens, then generates greedily for each beam.
    fn generate_beam_search_d1_greedy_batch(
        &self,
        input_tokens: &[LlamaToken],
        max_new_tokens: usize,
        eos_token_id: Option<i32>,
        beam_size: usize,
    ) -> Result<Vec<(Vec<LlamaToken>, f32)>> {
        ensure_fits(input_tokens.len() + 1, self.n_ctx)?;
        let max_new_tokens = max_new_tokens.min(self.n_ctx as usize - input_tokens.len());
        let backend = get_backend()?;

        // Set n_batch and n_ubatch large enough to avoid batch splitting
        // which causes "coupled sequences" error
        let batch_size = input_tokens
            .len()
            .saturating_mul(beam_size)
            .saturating_add(64)
            .min(u32::MAX as usize) as u32;
        let ctx_params = self
            .context_params()
            .with_n_seq_max(beam_size.try_into().unwrap_or(32))
            .with_n_batch(batch_size)
            .with_n_ubatch(batch_size);

        let mut ctx = self
            .model
            .new_context(backend, ctx_params)
            .map_err(|e| KanjiError::Inference(e.into()))?;

        let model_eos = self.model.token_eos();
        let input_len = input_tokens.len();

        // Step 1: Process input tokens for ALL sequences in one batch
        // Add each token separately for each sequence (not coupled)
        let mut batch = LlamaBatch::new(512, 1);

        for (i, token) in input_tokens.iter().enumerate() {
            for seq_id in 0..beam_size as i32 {
                let is_last = i == input_len - 1 && seq_id == 0; // Only first seq needs logits
                batch
                    .add(*token, i as i32, &[seq_id], is_last)
                    .map_err(|e| KanjiError::Inference(e.into()))?;
            }
        }
        ctx.decode(&mut batch)
            .map_err(|e| KanjiError::Inference(e.into()))?;

        // Step 2: Get top-k initial tokens (from any seq, all have same logits at this point)
        let logits = ctx.get_logits();
        let (top_tokens, top_log_probs) = self.get_top_k_tokens(logits, beam_size);

        // Step 3: Initialize beam state
        let mut beam_tokens: Vec<Vec<LlamaToken>> = top_tokens.iter().map(|&t| vec![t]).collect();
        let beam_scores: Vec<f32> = top_log_probs.clone();
        let mut beam_finished: Vec<bool> = vec![false; beam_size];

        // Check if any initial tokens are EOS
        for (i, &token) in top_tokens.iter().enumerate() {
            if self.is_eos_token(token, eos_token_id, model_eos) {
                beam_finished[i] = true;
            }
        }

        // Step 4: Add initial tokens to each beam's sequence
        batch.clear();
        for (beam_idx, &token) in top_tokens.iter().enumerate() {
            if !beam_finished[beam_idx] {
                batch
                    .add(token, input_len as i32, &[beam_idx as i32], true)
                    .map_err(|e| KanjiError::Inference(e.into()))?;
            }
        }

        if batch.n_tokens() > 0 {
            ctx.decode(&mut batch)
                .map_err(|e| KanjiError::Inference(e.into()))?;
        }

        // Step 5: Generate tokens for all beams in parallel
        let mut samplers: Vec<LlamaSampler> =
            (0..beam_size).map(|_| LlamaSampler::greedy()).collect();

        for _step in 0..(max_new_tokens - 1) {
            // Count active beams
            let active_count = beam_finished.iter().filter(|&&f| !f).count();
            if active_count == 0 {
                break;
            }

            // Sample next token for each active beam
            // Track which beams added tokens to know their logit positions
            let mut active_beams: Vec<usize> = Vec::new();
            let mut new_tokens: Vec<LlamaToken> = Vec::new();

            for (beam_idx, finished) in beam_finished.iter().enumerate() {
                if *finished {
                    continue;
                }
                // Logit position corresponds to order in the batch (0, 1, 2, ...)
                let logit_idx = active_beams.len() as i32;
                let new_token = samplers[beam_idx].sample(&ctx, logit_idx);
                active_beams.push(beam_idx);
                new_tokens.push(new_token);
            }

            // Process sampled tokens
            batch.clear();
            for (i, beam_idx) in active_beams.iter().enumerate() {
                let new_token = new_tokens[i];

                // Check for EOS
                if self.is_eos_token(new_token, eos_token_id, model_eos) {
                    beam_finished[*beam_idx] = true;
                } else {
                    beam_tokens[*beam_idx].push(new_token);
                    let pos = (input_len + beam_tokens[*beam_idx].len() - 1) as i32;
                    batch
                        .add(new_token, pos, &[*beam_idx as i32], true)
                        .map_err(|e| KanjiError::Inference(e.into()))?;
                }
            }

            // Decode all active beams at once
            if batch.n_tokens() > 0 {
                ctx.decode(&mut batch)
                    .map_err(|e| KanjiError::Inference(e.into()))?;
            } else {
                break;
            }
        }

        // Collect results
        let results: Vec<(Vec<LlamaToken>, f32)> =
            beam_tokens.into_iter().zip(beam_scores).collect();

        Ok(results)
    }

    /// Internal implementation of true beam search algorithm
    ///
    /// Unlike depth-1 beam methods which select top-k initial tokens
    /// and then generate greedily, this implements proper beam search that
    /// tracks cumulative probabilities at every step and keeps the globally
    /// best beam_size candidates.
    ///
    /// # Algorithm
    ///
    /// 1. Start with top-k initial tokens as beams
    /// 2. At each step:
    ///    - For each active beam, get top-k candidate next tokens
    ///    - Score each candidate: beam_score + log_prob(new_token)
    ///    - Keep only the best beam_size candidates globally
    /// 3. Repeat until all beams reach EOS or max_new_tokens
    ///
    /// True beam search implementation without KV cache sharing.
    /// This implementation processes full sequences at each step to avoid
    /// KV cache copy issues with GPT-2 models. It's slower but more reliable.
    fn generate_beam_search_impl(
        &self,
        input_tokens: &[LlamaToken],
        max_new_tokens: usize,
        eos_token_id: Option<i32>,
        beam_size: usize,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<Vec<(Vec<LlamaToken>, f32)>> {
        ensure_fits(input_tokens.len() + 1, self.n_ctx)?;
        let max_new_tokens = max_new_tokens.min(self.n_ctx as usize - input_tokens.len());
        let model_eos = self.model.token_eos();

        // One context for the whole search. The prompt is evaluated once and
        // stays in the KV cache; each beam then only evaluates the tokens that
        // differ from what the cache already holds after the prompt — usually
        // just its last token, since sibling beams share everything before it.
        // (Evaluating every beam from scratch in a fresh context made one
        // search cost hundreds of full-sentence passes.)
        let backend = get_backend()?;
        let mut ctx = self
            .model
            .new_context(backend, self.context_params())
            .map_err(|e| KanjiError::Inference(e.into()))?;
        let mut cached: Vec<LlamaToken> = Vec::new();

        // Step 1: Get initial logits
        let initial_logits =
            Self::eval_after_prompt(&mut ctx, input_tokens, &mut cached, &[], true)?;
        let (top_tokens, top_log_probs) = self.get_top_k_tokens(&initial_logits, beam_size);

        // Initialize beams, partitioning EOS tokens into finished
        let mut beams: Vec<BeamState> = Vec::with_capacity(beam_size);
        let mut finished_beams: Vec<BeamState> = Vec::new();

        for (&token, &log_prob) in top_tokens.iter().zip(top_log_probs.iter()) {
            let beam = BeamState {
                tokens: vec![token],
                score: log_prob,
            };

            if self.is_eos_token(token, eos_token_id, model_eos) {
                finished_beams.push(beam);
            } else {
                beams.push(beam);
            }
        }

        // Expansion factor
        let expand_k = beam_size.max(4);

        // Step 2: Main beam search loop
        for _step in 0..(max_new_tokens - 1) {
            if beams.is_empty()
                || cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Relaxed))
            {
                break;
            }

            // Early termination check
            if finished_beams.len() >= beam_size {
                let best_finished = finished_beams
                    .iter()
                    .map(|b| b.score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let best_active = beams
                    .iter()
                    .map(|b| b.score)
                    .fold(f32::NEG_INFINITY, f32::max);
                if best_active < best_finished {
                    break;
                }
            }

            // Collect candidates from all beams
            let mut candidates: Vec<BeamState> = Vec::new();

            for beam in &beams {
                let logits = Self::eval_after_prompt(
                    &mut ctx,
                    input_tokens,
                    &mut cached,
                    &beam.tokens,
                    false,
                )?;
                let (top_tokens, top_log_probs) = self.get_top_k_tokens(&logits, expand_k);

                // Create candidates
                for (&token, &log_prob) in top_tokens.iter().zip(top_log_probs.iter()) {
                    let mut new_tokens = beam.tokens.clone();
                    new_tokens.push(token);

                    candidates.push(BeamState {
                        tokens: new_tokens,
                        score: beam.score + log_prob,
                    });
                }
            }

            // Sort and keep top beam_size candidates
            candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
            candidates.truncate(beam_size);

            // Partition into finished and active beams
            beams.clear();
            for candidate in candidates {
                let last_token = match candidate.tokens.last() {
                    Some(&t) => t,
                    None => continue,
                };

                if self.is_eos_token(last_token, eos_token_id, model_eos) {
                    finished_beams.push(candidate);
                } else {
                    beams.push(candidate);
                }
            }
        }

        // Combine all results
        let mut all_results: Vec<(Vec<LlamaToken>, f32)> = finished_beams
            .into_iter()
            .chain(beams)
            .map(|b| (b.tokens, b.score))
            .collect();

        // Sort by score and take top beam_size
        all_results.sort_by(|a, b| b.1.total_cmp(&a.1));
        all_results.truncate(beam_size);

        Ok(all_results)
    }

    /// Logits after `prompt` + `suffix` in `ctx`, evaluating only what the KV
    /// cache doesn't already hold. `cached` tracks the suffix currently in the
    /// cache after the prompt; `first` evaluates the prompt itself.
    fn eval_after_prompt(
        ctx: &mut llama_cpp_2::context::LlamaContext<'_>,
        prompt: &[LlamaToken],
        cached: &mut Vec<LlamaToken>,
        suffix: &[LlamaToken],
        first: bool,
    ) -> Result<Vec<f32>> {
        let (start, tokens): (usize, &[LlamaToken]) = if first {
            (0, prompt)
        } else {
            let common = cached
                .iter()
                .zip(suffix)
                .take_while(|(a, b)| a == b)
                .count();
            // Re-evaluate at least the last token: its logits are the output.
            let keep = common.min(suffix.len().saturating_sub(1));
            ctx.clear_kv_cache_seq(Some(0), Some((prompt.len() + keep) as u32), None)
                .map_err(|e| KanjiError::Inference(e.into()))?;
            cached.truncate(keep);
            (prompt.len() + keep, &suffix[keep..])
        };
        let mut batch = LlamaBatch::new(tokens.len().max(1), 1);
        for (i, token) in tokens.iter().enumerate() {
            batch
                .add(*token, (start + i) as i32, &[0], i + 1 == tokens.len())
                .map_err(|e| KanjiError::Inference(e.into()))?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| KanjiError::Inference(e.into()))?;
        if !first {
            cached.extend_from_slice(tokens);
        }
        Ok(ctx.get_logits_ith(batch.n_tokens() - 1).to_vec())
    }

    /// Get top-k tokens from logits with log probabilities
    fn get_top_k_tokens(&self, logits: &[f32], k: usize) -> (Vec<LlamaToken>, Vec<f32>) {
        // Convert logits to log probabilities using log-softmax
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f32 = logits
            .iter()
            .map(|&x| (x - max_logit).exp())
            .sum::<f32>()
            .ln()
            + max_logit;

        let mut token_scores: Vec<(usize, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &x)| (i, x - log_sum_exp))
            .collect();
        token_scores.sort_by(|a, b| b.1.total_cmp(&a.1));
        token_scores.truncate(k);

        let tokens: Vec<LlamaToken> = token_scores
            .iter()
            .map(|(i, _)| LlamaToken(*i as i32))
            .collect();
        let log_probs: Vec<f32> = token_scores.iter().map(|(_, lp)| *lp).collect();

        (tokens, log_probs)
    }

    /// Check if a token is an EOS token.
    ///
    /// Uses the model's own EOS/EOG metadata rather than hardcoded token IDs.
    fn is_eos_token(
        &self,
        token: LlamaToken,
        eos_token_id: Option<i32>,
        model_eos: LlamaToken,
    ) -> bool {
        eos_token_id.is_some_and(|eos| token.0 == eos)
            || token == model_eos
            || self.model.is_eog_token(token)
    }

    /// Generate tokens with a custom sampler
    /// Returns the generated sequence together with the summed log-probability
    /// of the tokens it produced. The score is free: the logits it reads are
    /// the same ones the sampler is about to use at each step.
    fn generate_with_sampler(
        &self,
        input_tokens: &[LlamaToken],
        max_new_tokens: usize,
        eos_token_id: Option<i32>,
        mut sampler: LlamaSampler,
    ) -> Result<(Vec<LlamaToken>, f32)> {
        ensure_fits(input_tokens.len() + 1, self.n_ctx)?;
        let max_new_tokens = max_new_tokens.min(self.n_ctx as usize - input_tokens.len());
        let backend = get_backend()?;
        let ctx_params = self.context_params();

        let mut ctx = self
            .model
            .new_context(backend, ctx_params)
            .map_err(|e| KanjiError::Inference(e.into()))?;

        let mut batch = LlamaBatch::new(512, 1);
        let mut generated = input_tokens.to_vec();

        // Process input tokens
        for (i, token) in input_tokens.iter().enumerate() {
            let is_last = i == input_tokens.len() - 1;
            batch
                .add(*token, i as i32, &[0], is_last)
                .map_err(|e| KanjiError::Inference(e.into()))?;
        }

        ctx.decode(&mut batch)
            .map_err(|e| KanjiError::Inference(e.into()))?;

        // Get model's EOS token for comparison
        let model_eos = self.model.token_eos();

        // Generate new tokens
        let mut total_log_prob = 0.0f32;
        for n_cur in (input_tokens.len()..).take(max_new_tokens) {
            // Read the distribution before sampling: these logits are already
            // computed, so accumulating the chosen token's log-probability adds
            // no inference work.
            let step_logits = ctx.get_logits().to_vec();
            let new_token = sampler.sample(&ctx, -1);

            // Check for EOS using the provided token ID
            if eos_token_id.is_some_and(|eos| new_token.0 == eos) {
                break;
            }

            // Check against model's EOS token
            if new_token == model_eos {
                break;
            }

            // Check if model thinks it's end of generation
            if self.model.is_eog_token(new_token) {
                break;
            }

            total_log_prob += log_prob_of(&step_logits, new_token.0 as usize, step_logits.len());
            generated.push(new_token);

            // Prepare next batch with just the new token
            batch.clear();
            batch
                .add(new_token, n_cur as i32, &[0], true)
                .map_err(|e| KanjiError::Inference(e.into()))?;

            ctx.decode(&mut batch)
                .map_err(|e| KanjiError::Inference(e.into()))?;
        }

        Ok((generated, total_log_prob))
    }

    /// Get the EOS token ID from the model
    pub fn eos_token_id(&self) -> LlamaToken {
        self.model.token_eos()
    }
}

/// Log-softmax of `logits` at `index`, i.e. log P(token = index).
fn log_prob_of(logits: &[f32], index: usize, vocab_size: usize) -> f32 {
    let n = vocab_size.min(logits.len());
    if index >= n {
        return 0.0;
    }
    let max_logit = logits[..n]
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp: f32 = logits[..n]
        .iter()
        .map(|&x| (x - max_logit).exp())
        .sum::<f32>()
        .ln()
        + max_logit;
    logits[index] - log_sum_exp
}

/// Reusable NLL scorer that keeps a single llama.cpp context alive.
///
/// Creating a `LlamaContext` is expensive. This struct amortizes the cost by
/// creating one context and clearing the KV cache between calls.
/// Use one `NllScorer` per thread for parallel scoring.
pub struct NllScorer<'a> {
    model: &'a LlamaCppModel,
    ctx: llama_cpp_2::context::LlamaContext<'a>,
    vocab_size: usize,
    n_ctx: u32,
    /// Tokens the KV cache holds. Candidates scored against one prompt share
    /// it, so only each candidate's own tokens are evaluated.
    kv: Vec<LlamaToken>,
}

impl<'a> NllScorer<'a> {
    /// Create a new NLL scorer with a reusable context.
    pub fn new(model: &'a LlamaCppModel, n_ctx: u32) -> Result<Self> {
        let backend = get_backend()?;

        let ctx_params = model.context_params().with_n_ctx(Some(
            NonZeroU32::new(n_ctx).expect("n_ctx must be non-zero"),
        ));

        let ctx = model
            .model
            .new_context(backend, ctx_params)
            .map_err(|e| KanjiError::Inference(e.into()))?;

        let vocab_size = model.model.n_vocab() as usize;

        Ok(Self {
            model,
            ctx,
            vocab_size,
            n_ctx,
            kv: Vec::new(),
        })
    }

    /// Evaluate `tokens`, reusing whatever prefix the KV cache already holds
    /// and computing logits from position `first_logit` on. Returns the
    /// position of batch index 0, to map positions to logit indices.
    fn decode_from(&mut self, tokens: &[LlamaToken], first_logit: usize) -> Result<usize> {
        ensure_fits(tokens.len(), self.n_ctx)?;
        let common = self
            .kv
            .iter()
            .zip(tokens)
            .take_while(|(a, b)| a == b)
            .count();
        let keep = common.min(first_logit);
        self.ctx
            .clear_kv_cache_seq(Some(0), Some(keep as u32), None)
            .map_err(|e| KanjiError::Inference(e.into()))?;
        self.kv.truncate(keep);
        let mut batch = LlamaBatch::new(tokens.len() - keep, 1);
        for (offset, token) in tokens[keep..].iter().enumerate() {
            let pos = keep + offset;
            batch
                .add(*token, pos as i32, &[0], pos >= first_logit)
                .map_err(|e| KanjiError::Inference(e.into()))?;
        }
        self.ctx
            .decode(&mut batch)
            .map_err(|e| KanjiError::Inference(e.into()))?;
        self.kv.extend_from_slice(&tokens[keep..]);
        Ok(keep)
    }

    /// Compute per-character NLL for a single (reading, surface) pair.
    ///
    /// Reuses the internal context by clearing the KV cache between calls.
    /// `context` is the surrounding text left of the cursor, in the same slot
    /// the conversion prompt uses — scoring a candidate against the sentence it
    /// will land in is the whole point of reranking, so it must not be blank.
    pub fn compute_nll(
        &mut self,
        reading_katakana: &str,
        surface: &str,
        context: &str,
    ) -> Result<f32> {
        self.compute_nll_detail(reading_katakana, surface, context)
            .map(|d| d.mean)
    }

    /// Per-character NLL of `following` as the continuation of the context
    /// slot holding `preceding` — how naturally the text after the cursor
    /// reads once a candidate has been placed before it.
    pub fn compute_continuation_nll(&mut self, preceding: &str, following: &str) -> Result<f32> {
        let prompt = format!("{}{}", super::CONTEXT_TOKEN, preceding);
        let full_text = format!("{prompt}{following}");
        let prompt_tokens = self.model.tokenize(&prompt)?;
        let full_tokens = self.model.tokenize(&full_text)?;
        if full_tokens.len() <= prompt_tokens.len() {
            return Ok(100.0);
        }
        let base = self.decode_from(&full_tokens, prompt_tokens.len() - 1)?;
        let mut total = 0.0f32;
        for pos in prompt_tokens.len() - 1..full_tokens.len() - 1 {
            let target = full_tokens[pos + 1].0 as usize;
            if target < self.vocab_size {
                let logits = self.ctx.get_logits_ith((pos - base) as i32);
                total -= log_prob_of(logits, target, self.vocab_size);
            }
        }
        Ok(total / following.chars().count().max(1) as f32)
    }

    /// As [`Self::compute_nll`], but also reporting the single worst token.
    ///
    /// The mean alone cannot judge a long sentence: one wrong character in
    /// twenty-six barely moves it, so `…形で対欧できます` sits within any
    /// sensible margin of `…形で対応できます`. The worst token is where that
    /// damage is still visible.
    pub fn compute_nll_detail(
        &mut self,
        reading_katakana: &str,
        surface: &str,
        context: &str,
    ) -> Result<NllDetail> {
        use super::{CONTEXT_TOKEN, INPUT_START_TOKEN, OUTPUT_START_TOKEN};

        let prompt = format!(
            "{}{}{}{}{}",
            CONTEXT_TOKEN, context, INPUT_START_TOKEN, reading_katakana, OUTPUT_START_TOKEN
        );
        let full_text = format!("{}{}", prompt, surface);

        let prompt_tokens = self.model.tokenize(&prompt)?;
        let full_tokens = self.model.tokenize(&full_text)?;

        if full_tokens.len() <= prompt_tokens.len() {
            return Ok(NllDetail {
                mean: 100.0,
                worst: 100.0,
            });
        }

        let n_tokens = full_tokens.len();
        let start_pos = prompt_tokens.len() - 1;
        let base = self.decode_from(&full_tokens, start_pos)?;

        let end_pos = n_tokens - 1;
        let mut total_nll: f32 = 0.0;
        let mut worst: f32 = 0.0;
        let mut n_scored = 0;

        for pos in start_pos..end_pos {
            let logits = self.ctx.get_logits_ith((pos - base) as i32);

            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp: f32 = logits
                .iter()
                .take(self.vocab_size)
                .map(|&x| (x - max_logit).exp())
                .sum::<f32>()
                .ln()
                + max_logit;

            let target = full_tokens[pos + 1].0 as usize;
            if target < self.vocab_size {
                let nll = -(logits[target] - log_sum_exp);
                total_nll += nll;
                worst = worst.max(nll);
            }
            n_scored += 1;
        }

        if n_scored == 0 {
            return Ok(NllDetail {
                mean: 100.0,
                worst: 100.0,
            });
        }

        let n_chars = surface.chars().count().max(1);
        Ok(NllDetail {
            mean: total_nll / n_chars as f32,
            worst,
        })
    }
}

/// How well a candidate reads: the per-character average, and the single
/// character the model disbelieved most.
#[derive(Debug, Clone, Copy)]
pub struct NllDetail {
    /// Total NLL divided by the surface's character count.
    pub mean: f32,
    /// Largest single-token NLL in the surface.
    pub worst: f32,
}
