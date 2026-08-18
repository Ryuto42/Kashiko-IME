//! Display and preedit construction for the IME engine

use super::*;

impl InputMethodEngine {
    /// Build display text from the input buffer and romaji buffer
    /// Format: composed[:cursor] + romaji_buffer + composed[cursor:]
    /// In katakana mode, the composed parts are converted to katakana.
    pub(super) fn build_input_display(&self) -> String {
        let before: String = self
            .input_buf
            .text
            .chars()
            .take(self.input_buf.cursor_pos)
            .collect();
        let after: String = self
            .input_buf
            .text
            .chars()
            .skip(self.input_buf.cursor_pos)
            .collect();
        let buffer = self.converters.romaji.buffer();

        let katakana = self.input_mode == InputMode::Katakana;
        let display_before = if katakana {
            kashiko_engine::hiragana_to_katakana(&before)
        } else {
            before
        };
        let display_after = if katakana {
            kashiko_engine::hiragana_to_katakana(&after)
        } else {
            after
        };

        format!("{}{}{}", display_before, buffer, display_after)
    }

    /// Get the caret position in the display text (in characters)
    pub(super) fn display_caret_position(&self) -> usize {
        self.input_buf.cursor_pos + self.converters.romaji.buffer().chars().count()
    }

    /// Build a preedit for composing state.
    /// If live conversion text is present, shows live_text + romaji_buffer with caret at end.
    /// Otherwise shows the input buffer display with cursor-based caret.
    pub(super) fn build_composing_preedit(&self) -> Preedit {
        let (display, caret) = if !self.live.text.is_empty() {
            let buffer = self.converters.romaji.buffer();
            let display = format!("{}{}", self.live.text, buffer);
            let caret = display.chars().count();
            (display, caret)
        } else {
            (self.build_input_display(), self.display_caret_position())
        };
        let len = display.chars().count();
        let mut preedit = Preedit::with_text(&display);
        preedit.set_caret(caret);
        preedit.set_attributes(vec![PreeditAttribute::underline(0, len)]);
        preedit
    }

    /// Format an `lctx: … rctx: …` line from explicit left/right context
    /// strings, each truncated to `display_context_len` (left keeps its tail,
    /// right keeps its head). Empty when both are absent or the limit is 0.
    fn context_line(&self, left: Option<&str>, right: Option<&str>) -> String {
        let max_len = self.config.display_context_len;
        if max_len == 0 {
            return String::new();
        }
        let lctx = left.filter(|s| !s.is_empty()).map(|left| {
            let char_count = left.chars().count();
            if char_count > max_len {
                let start = char_count - max_len;
                format!("...{}", left.chars().skip(start).collect::<String>())
            } else {
                left.to_string()
            }
        });

        let rctx = right.filter(|s| !s.is_empty()).map(|right| {
            let char_count = right.chars().count();
            if char_count > max_len {
                format!("{}...", right.chars().take(max_len).collect::<String>())
            } else {
                right.to_string()
            }
        });

        match (lctx, rctx) {
            (Some(l), Some(r)) => format!("lctx: {} rctx: {}", l, r),
            (Some(l), None) => format!("lctx: {}", l),
            (None, Some(r)) => format!("rctx: {}", r),
            (None, None) => String::new(),
        }
    }

    /// Surrounding-text context line (editor left/right). Used by conversion-mode
    /// aux text, where there is no live chunking.
    #[allow(dead_code)]
    pub(super) fn display_context(&self) -> String {
        let ctx = self.surrounding_context.as_ref();
        self.context_line(
            ctx.and_then(|c| c.left.as_deref()),
            ctx.and_then(|c| c.right.as_deref()),
        )
    }

    /// Context line for live conversion (composing / auto-suggest). The single
    /// `lctx:` shown is the *current chunk's* left context — the editor
    /// surrounding text plus the converted text of the preceding chunks, derived
    /// via `chunk_lctx` — so the model context that chunk uses is what gets
    /// displayed, rather than a second redundant lctx. It already folds in the
    /// editor surrounding left context (so an empty buffer shows it as-is). The
    /// right side stays the editor surrounding right context.
    #[allow(dead_code)]
    pub(super) fn display_context_chunked(&self) -> String {
        let lctx = self.chunk_lctx(self.current_chunk_index());
        let left = (!lctx.is_empty()).then_some(lctx.as_str());
        let right = self
            .surrounding_context
            .as_ref()
            .and_then(|c| c.right.as_deref());
        self.context_line(left, right)
    }

    /// Get the current mode indicator string
    pub(super) fn mode_indicator(&self) -> String {
        let base = match self.input_mode {
            InputMode::Alphabet => "[A]",
            InputMode::Katakana => "[カ]",
            InputMode::Hiragana => "[あ]",
            // ☺ (U+263A, Unicode 1.1 / 1993) — the oldest smiley-face
            // codepoint in Unicode; gives emoji mode an unambiguous
            // glyph in the aux text that's distinct from `[A]` so the
            // user sees they're not in plain alphabet input.
            InputMode::Emoji => "[☺]",
        };
        if self.live.enabled {
            format!("⚡{}", base)
        } else {
            base.to_string()
        }
    }

    /// Format aux text for composing input mode
    pub(super) fn format_aux_composing(&self) -> String {
        String::new()
    }

    /// Get token count for a reading (returns None if converter not initialized)
    #[allow(dead_code)]
    pub(super) fn get_token_count(&self, reading: &str) -> Option<usize> {
        self.converters
            .kanji
            .as_ref()
            .and_then(|c| c.count_input_tokens(reading).ok())
    }

    /// Get the display name of the model used for the last conversion
    /// Falls back to the static model name if no conversion has happened yet
    #[allow(dead_code)]
    fn last_used_model(&self) -> String {
        if self.metrics.model_name.is_empty() {
            self.model_name()
        } else {
            self.metrics.model_name.clone()
        }
    }

    /// Format aux text for conversion mode
    /// Currently empty: the aux line in conversion mode is reserved for the
    /// deep-search progress indicator, which `start_conversion` sets directly.
    /// Kept as a hook (and to keep the action shape uniform) — it must stay
    /// cheap, since it runs on every conversion and every candidate move. It
    /// used to call `display_context`/`last_used_model`/`get_token_count` and
    /// throw the results away, which put a tokenizer pass on that hot path.
    pub(super) fn format_aux_conversion_with_page(
        &self,
        _reading: &str,
        _candidates: Option<&CandidateList>,
    ) -> String {
        String::new()
    }

    /// Format aux text for auto-suggest mode
    /// Note: token count is not shown here to avoid performance overhead on every keystroke
    /// Timing shows inference_ms/process_key_ms (process_key_ms is from previous keystroke)
    pub(super) fn format_aux_suggest(&self, _reading: &str) -> String {
        String::new()
    }

    /// Truncate context to safe size for API calls
    pub(super) fn truncate_context_for_api(&self) -> String {
        let from_document = self
            .surrounding_context
            .as_ref()
            .and_then(|ctx| ctx.left.as_deref())
            .filter(|left| !left.is_empty());
        match from_document {
            // The application told us what precedes the cursor: authoritative.
            Some(left) => self.truncate_context(left),
            // It didn't (or can't — many Electron apps and terminals don't
            // implement `string(from:)`). Fall back to what we committed
            // ourselves, which is the same text sitting left of the cursor.
            None if self.config.max_api_context_len > 0 => {
                self.truncate_context(&self.committed_tail)
            }
            None => String::new(),
        }
    }

    /// Append committed text to the fallback context tail, keeping only as much
    /// as conversion can actually use.
    pub(super) fn record_committed_context(&mut self, text: &str) {
        if text.is_empty() || self.config.max_api_context_len == 0 {
            return;
        }
        self.committed_tail.push_str(text);
        let extra = self
            .committed_tail
            .chars()
            .count()
            .saturating_sub(self.config.max_api_context_len);
        if extra > 0 {
            self.committed_tail = self.committed_tail.chars().skip(extra).collect();
        }
    }

    /// Truncate a context string to safe size for API calls
    pub(super) fn truncate_context(&self, context: &str) -> String {
        let char_count = context.chars().count();
        if char_count > self.config.max_api_context_len {
            let start = char_count - self.config.max_api_context_len;
            context.chars().skip(start).collect()
        } else {
            context.to_string()
        }
    }
}
