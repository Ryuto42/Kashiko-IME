//! Composing input handling (Empty and Composing states)

use super::*;

/// Minimum reading length (in chars) before long-sentence candidates are shown.
/// Below this threshold, candidates whose text is much longer than the reading
/// are filtered out to keep the suggestion list clean for short input.
pub(super) const LONG_INPUT_THRESHOLD: usize = 4;

/// Maximum candidate text length (in chars) allowed when the reading is short
/// (below `LONG_INPUT_THRESHOLD`). Candidates longer than this are suppressed
/// during short-text input.
pub(super) const SHORT_INPUT_MAX_CANDIDATE_LEN: usize = 6;

/// Descriptions the width-annotation rewriters attach to a plain script
/// transliteration of the reading (see `width_annotation` and
/// `HalfWidthKatakanaRewriter`). Auto-suggest hides these — they're always
/// derivable from the reading itself and just add noise to the live popup —
/// while keeping other rewriter output (symbol substitutions, emoji) intact.
const SCRIPT_VARIANT_DESCRIPTIONS: [&str; 3] = ["[全]ひらがな", "[全]カタカナ", "[半]カタカナ"];

/// Whether `text` is disproportionately longer than `reading` — the shape of a
/// model hallucination (a 1–2 char reading converting into a whole phrase).
/// Shared by the candidate filter and the live-conversion guard so both reject
/// the same thing.
fn is_overlong_for_reading(text: &str, reading: &str) -> bool {
    let reading_len = reading.chars().count();
    let text_len = text.chars().count();
    text_len > reading_len * 2 + 4
        || (text_len >= 12 && reading_len < 8)
        || (reading_len < LONG_INPUT_THRESHOLD && text_len > SHORT_INPUT_MAX_CANDIDATE_LEN)
}

/// Remove candidates that are disproportionately long relative to the reading.
/// When the reading is short (< `LONG_INPUT_THRESHOLD` chars), candidates
/// longer than `SHORT_INPUT_MAX_CANDIDATE_LEN` chars are dropped.
fn filter_long_candidates_for_short_input(candidates: &mut Vec<Candidate>, reading: &str) {
    candidates.retain(|c| !is_overlong_for_reading(&c.text, reading));
}

impl InputMethodEngine {
    /// Refresh the input state: rebuild preedit and run auto-suggest for candidates.
    pub(super) fn refresh_input_state(&mut self) -> EngineResult {
        // Alphabet mode with active live conversion but no kana left to convert:
        // preserve the existing conversion display without re-running the model.
        // (When the buffer still contains kana we fall through and reconvert below,
        // so a mixed reading like `きょうはABC` keeps live-converting.)
        if self.input_mode == InputMode::Alphabet
            && !self.live.text.is_empty()
            && !kashiko_engine::contains_kana(&self.input_buf.text)
            && !self.last_key_was_backspace
        {
            let preedit = self.set_composing_state();
            let aux = self.format_aux_composing();
            let mut result = EngineResult::consumed().with_action(EngineAction::UpdatePreedit(preedit));
            if !aux.is_empty() {
                result = result.with_action(EngineAction::UpdateAuxText(aux));
            }
            return result;
        }

        // Run auto-suggest via chunked conversion. Normally skipped in alphabet
        // mode (raw latin has no hiragana to convert), but if the buffer still
        // contains kana — e.g. the user typed hiragana, switched to alphabet mode,
        // and kept typing — keep converting the mixed reading so live conversion
        // stays alive. `chunked_auto_suggest` splits long input into
        // bounded-length chunks so per-keystroke latency stays flat; for input
        // within one chunk this is identical to a whole-buffer call.
        let convert = !self.input_buf.text.is_empty()
            && (self.input_mode != InputMode::Alphabet
                || kashiko_engine::contains_kana(&self.input_buf.text));
        let candidates = if convert {
            let reading = self.input_buf.text.clone();
            self.chunked_auto_suggest()
                .map(|converted| (vec![converted], reading))
        } else {
            self.chunks.clear();
            None
        };

        // Backspace pauses live conversion display: `chunked_auto_suggest`
        // above still ran (so the chunk cache stays correctly partitioned
        // for when typing resumes), but its result is discarded here rather
        // than shown — the user sees the raw hiragana/romaji buffer instead
        // of a re-guessed kanji conversion while backspacing. Live
        // conversion resumes on the next non-backspace key, since that path
        // clears `last_key_was_backspace` before calling this function.
        if self.last_key_was_backspace {
            self.live.text.clear();
            let preedit = self.set_composing_state();
            return EngineResult::consumed()
                .with_action(EngineAction::UpdatePreedit(preedit))
                .with_action(EngineAction::HideCandidates)
                .with_action(EngineAction::HideAuxText);
        }

        let Some((candidates, _reading)) = candidates else {
            // No kanji-conversion suggestion. Only rule-based rewriter output
            // is surfaced here — the Emoji shortcode picker (`:pien` → 🥺) and
            // live symbol substitution (`。。。` → `…`), both of which have to
            // appear before Space by design.
            //
            // Learning and dictionary lookups are deliberately NOT consulted:
            // their predictive (prefix) matches produced unsolicited
            // whole-sentence suggestions mid-typing — typing `なにかいいあ`
            // popped up a learned `なにかいい案ありますか`. Those belong to the
            // Space candidate list, where the user asked for them.
            self.live.text.clear();
            let preedit = self.set_composing_state();
            let reading = self.input_buf.text.clone();
            // Barely-typed input gets no popup at all: one or two characters
            // matched rare-kana rewriter variants (`い` → `ゐ`/`ヰ`/捨て仮名…),
            // which flashed a menu the user never asked for. Emoji mode is
            // exempt — its shortcode picker has to react from `:a` onwards.
            const MIN_POPUP_READING_CHARS: usize = 3;
            let mut all_candidates = if self.input_mode == InputMode::Emoji
                || reading.chars().count() >= MIN_POPUP_READING_CHARS
            {
                self.lookup_rewriter_variants(&reading)
            } else {
                Vec::new()
            };
            all_candidates.retain(|c| {
                !c.description
                    .as_deref()
                    .is_some_and(|d| SCRIPT_VARIANT_DESCRIPTIONS.contains(&d))
            });
            filter_long_candidates_for_short_input(&mut all_candidates, &reading);
            if all_candidates.is_empty() {
                return EngineResult::consumed()
                    .with_action(EngineAction::UpdatePreedit(preedit))
                    .with_action(EngineAction::HideCandidates)
                    .with_action(EngineAction::UpdateAuxText(self.format_aux_composing()));
            }
            return EngineResult::consumed()
                .with_action(EngineAction::UpdatePreedit(preedit))
                .with_action(EngineAction::ShowCandidates(CandidateList::new(
                    all_candidates,
                )))
                .with_action(EngineAction::UpdateAuxText(self.format_aux_composing()));
        };

        // Live conversion mode: show converted text in preedit. No candidate
        // popup here either — Space still brings up the full candidate list.
        if self.live.enabled && self.input_mode != InputMode::Katakana {
            // A latin word we're confident about (`WordPress`) beats the kana
            // the romaji converter produced from those keystrokes. Only proven
            // words drive live conversion — an unproven guess still appears in
            // the Space candidate list, but must not hijack the preedit.
            let latin = self
                .latin_word_candidate()
                .filter(|(_, score)| *score >= super::conversion::KNOWN_LATIN_BONUS)
                .map(|(text, _)| text);
            let suggestion = latin.unwrap_or_else(|| candidates[0].clone());
            // Guard against the model hallucinating a long continuation for a
            // short reading — a 1–2 char input must not live-convert into a
            // whole phrase.
            self.live.text = if is_overlong_for_reading(&suggestion, &self.input_buf.text) {
                self.input_buf.text.clone()
            } else {
                suggestion
            };
            let preedit = self.set_composing_state();
            let aux = self.format_aux_suggest(&self.input_buf.text.clone());
            return EngineResult::consumed()
                .with_action(EngineAction::UpdatePreedit(preedit))
                .with_action(EngineAction::HideCandidates)
                .with_action(EngineAction::UpdateAuxText(aux));
        }

        // Normal auto-suggest: hiragana preedit only, no candidate popup
        // (candidates appear once the user presses Space).
        self.live.text.clear();
        let preedit = self.set_composing_state();
        let aux = self.format_aux_suggest(&self.input_buf.text.clone());
        EngineResult::consumed()
            .with_action(EngineAction::UpdatePreedit(preedit))
            .with_action(EngineAction::HideCandidates)
            .with_action(EngineAction::UpdateAuxText(aux))
    }

    /// Process key in empty state
    pub(super) fn process_key_empty(&mut self, key: &KeyEvent, shift_active: bool) -> EngineResult {
        // Ctrl+Backspace right after a conversion commit: undo the commit
        // and return to conversion mode (reconversion). Any other key
        // invalidates the undo window, since the caret may move away from
        // the committed text.
        if key.is_press {
            if key.keysym == Keysym::BACKSPACE && key.modifiers.control_key {
                if let Some((reading, surface)) = self.last_commit.take() {
                    return self.reconvert_last_commit(reading, surface);
                }
                return EngineResult::not_consumed();
            }
            self.last_commit = None;
        }

        // Ctrl+Space: start input with full-width space
        if key.modifiers.control_key && key.keysym == Keysym::SPACE {
            self.converters.romaji.reset();
            self.input_buf.clear();
            self.input_buf.insert("\u{3000}");
            let preedit = self.set_composing_state();
            return EngineResult::consumed()
                .with_action(EngineAction::UpdatePreedit(preedit))
                .with_action(EngineAction::UpdateAuxText(self.format_aux_composing()));
        }

        // Bare Space from Empty state:
        //
        // * Hiragana mode → commit a full-width `　` directly, matching
        //   the Japanese-IME convention. We deliberately do NOT enter
        //   Composing here: if we did, the next Space the user typed
        //   would be interpreted by `process_key_composing` as the
        //   conversion trigger and an unwanted candidate window would
        //   appear after two spaces in a row.
        // * Any other mode → return `not_consumed` so the OS delivers
        //   a normal half-width ASCII space to the application. The
        //   user is either typing ASCII (Alphabet) or in an edge mode
        //   (Katakana / Emoji) where injecting `　` would be wrong.
        //
        // The full-width space gesture from Empty in any mode is
        // `Ctrl+Space` (above), which seeds a Composing session.
        if key.keysym == Keysym::SPACE && !key.modifiers.control_key && !key.modifiers.alt_key {
            return if self.input_mode == InputMode::Hiragana {
                if key.modifiers.shift_key {
                    EngineResult::consumed().with_action(EngineAction::Commit(" ".to_string()))
                } else {
                    EngineResult::consumed().with_action(EngineAction::Commit("\u{3000}".to_string()))
                }
            } else {
                EngineResult::not_consumed()
            };
        }

        // `:` from Empty state enters emoji shortcode mode — `:pien` stays
        // as `:pien` literally (no romaji conversion) while emoji candidates
        // are surfaced via the rewriter. The mode auto-exits back to Hiragana
        // on Escape or commit, so the user's next word lands in kana mode
        // again without an explicit toggle.
        //
        // Two keysym shapes can produce `:` depending on how fcitx5
        // resolves the layout: (a) the X11 `colon` keysym (0x003A)
        // arriving directly, or (b) the `semicolon` keysym (0x003B)
        // with shift held. Accept both so we don't depend on which
        // shape the upstream stack happens to emit.
        let typed_colon =
            key.to_char() == Some(':') || (shift_active && key.keysym == Keysym(b';' as u32));
        if typed_colon
            && !key.modifiers.control_key
            && !key.modifiers.alt_key
            && self.input_mode != InputMode::Alphabet
        {
            return self.start_emoji_mode();
        }

        // Only handle printable characters without modifiers (except shift)
        if let Some(ch) = key.to_char()
            && !key.modifiers.control_key
            && !key.modifiers.alt_key
        {
            // Detect Shift+letter: shift modifier with alphabetic, OR uppercase keysym.
            // fcitx5 may resolve Shift into the keysym (sending 'A' instead of 'a'+shift),
            // so we must also check for uppercase to handle both cases.
            let is_shift_alpha =
                ch.is_ascii_uppercase() || (shift_active && ch.is_ascii_alphabetic());

            if is_shift_alpha && self.input_mode != InputMode::Alphabet {
                self.input_mode = InputMode::Alphabet;
                self.shift_alphabet_active = true;
            } else if !is_shift_alpha && self.input_mode == InputMode::Alphabet && self.shift_alphabet_active && ch.is_ascii_alphabetic() {
                self.input_mode = InputMode::Hiragana;
                self.shift_alphabet_active = false;
            }
            let ch = if self.input_mode == InputMode::Alphabet && is_shift_alpha {
                ch.to_ascii_uppercase()
            } else {
                ch
            };
            return self.start_input(ch);
        }
        EngineResult::not_consumed()
    }

    /// Start input with a character (first character of a new input session).
    /// In alphabet mode, inserts directly; otherwise goes through romaji conversion.
    pub(super) fn start_input(&mut self, ch: char) -> EngineResult {
        self.last_key_was_backspace = false;
        self.converters.romaji.reset();
        self.input_buf.clear();
        self.input_buf.raw_keys.push(ch);

        if self.input_mode == InputMode::Alphabet {
            self.input_buf.insert(&ch.to_string());
        } else {
            let prev_output_len = 0;
            let _event = self.converters.romaji.push(ch);
            let romaji_buffer = self.converters.romaji.buffer().to_string();

            // PassThrough chars (no romaji rule, e.g. `'`, `;`, `<`, `(`) used to
            // auto-commit immediately, but that prevented users from composing
            // sequences like `「」` or getting symbol variants. Treat them like
            // digits — let them enter Composing and accumulate in the preedit.

            if self.converters.romaji.output().is_empty() && romaji_buffer.is_empty() {
                return EngineResult::not_consumed();
            }

            // Consume new converter output into composed_hiragana
            let new_output_len = self.converters.romaji.output().chars().count();
            if new_output_len > prev_output_len {
                let new_chars: String = self
                    .converters
                    .romaji
                    .output()
                    .chars()
                    .skip(prev_output_len)
                    .collect();
                self.input_buf.insert(&new_chars);
            }
        }

        let preedit = self.set_composing_state();

        EngineResult::consumed()
            .with_action(EngineAction::UpdatePreedit(preedit))
            .with_action(EngineAction::UpdateAuxText(self.format_aux_composing()))
    }

    /// Insert a full-width space (U+3000) at cursor position
    pub(super) fn input_fullwidth_space(&mut self) -> EngineResult {
        self.input_buf.insert("\u{3000}");
        self.refresh_input_state()
    }

    /// Process key in hiragana input state
    pub(super) fn process_key_composing(
        &mut self,
        key: &KeyEvent,
        shift_active: bool,
    ) -> EngineResult {
        // Handle Ctrl+key shortcuts
        if key.modifiers.control_key {
            match key.keysym {
                // Ctrl+Space: insert full-width space (U+3000)
                Keysym::SPACE => return self.input_fullwidth_space(),
                // Ctrl+K: enter katakana mode
                Keysym::KEY_K | Keysym::KEY_K_UPPER => return self.enter_katakana_mode(),
                // Ctrl+A: move to beginning (Emacs-style Home)
                Keysym::KEY_A | Keysym::KEY_A_UPPER => return self.move_caret_home(),
                // Ctrl+B: move left (Emacs-style Left)
                Keysym::KEY_B | Keysym::KEY_B_UPPER => return self.move_caret_left(),
                // Ctrl+E: move to end (Emacs-style End)
                Keysym::KEY_E | Keysym::KEY_E_UPPER => return self.move_caret_end(),
                // Ctrl+F: move right (Emacs-style Right)
                Keysym::KEY_F | Keysym::KEY_F_UPPER => return self.move_caret_right(),
                _ => {}
            }
        }

        match key.keysym {
            Keysym::RETURN => self.commit_composing(),
            Keysym::ESCAPE => self.cancel_composing(),
            Keysym::BACKSPACE => self.backspace_composing(),
            Keysym::DELETE => {
                self.last_key_was_backspace = false;
                self.delete_composing()
            }
            Keysym::SPACE if self.input_mode == InputMode::Alphabet => {
                self.last_key_was_backspace = false;
                self.input_char(' ')
            }
            Keysym::SPACE if key.modifiers.shift_key => {
                self.last_key_was_backspace = false;
                self.input_char(' ')
            }
            // Tab triggers conversion that bypasses the learning cache, so users
            // can escape stale or unwanted learned entries (mozc binds Tab to a
            // different conversion path — PredictAndConvert — in the same spirit).
            Keysym::TAB => self.start_conversion(true, None, 0),
            // `1` here means "advance past a live-conversion duplicate" (see
            // `start_conversion`'s doc comment) rather than an unconditional
            // move to the next candidate.
            Keysym::SPACE | Keysym::DOWN => self.start_conversion(false, None, 1),
            Keysym::UP => self.start_conversion(false, None, -1),
            Keysym::LEFT => self.move_caret_left(),
            Keysym::RIGHT => self.move_caret_right(),
            Keysym::HOME => self.move_caret_home(),
            Keysym::END => self.move_caret_end(),
            _ => {
                if let Some(ch) = key.to_char()
                    && !key.modifiers.control_key
                    && !key.modifiers.alt_key
                {
                    // Detect Shift+letter: shift modifier with alphabetic, OR uppercase keysym.
                    // fcitx5 may resolve Shift into the keysym (sending 'A' instead of 'a'+shift).
                    let is_shift_alpha =
                        ch.is_ascii_uppercase() || (shift_active && ch.is_ascii_alphabetic());

                    if is_shift_alpha && self.input_mode != InputMode::Alphabet {
                        // Bake katakana before switching so preedit doesn't revert
                        if self.input_mode == InputMode::Katakana {
                            self.bake_katakana();
                        }
                        self.input_mode = InputMode::Alphabet;
                        self.shift_alphabet_active = true;
                        self.flush_romaji_to_composed();
                        self.live.text.clear();
                    } else if !is_shift_alpha && self.input_mode == InputMode::Alphabet && self.shift_alphabet_active && ch.is_ascii_alphabetic() {
                        self.input_mode = InputMode::Hiragana;
                        self.shift_alphabet_active = false;
                    }
                    let ch = if self.input_mode == InputMode::Alphabet && is_shift_alpha {
                        ch.to_ascii_uppercase()
                    } else {
                        ch
                    };
                    return self.input_char(ch);
                }
                EngineResult::not_consumed()
            }
        }
    }

    /// Begin a new emoji-shortcode composing session.
    ///
    /// Resets any leftover state, switches `input_mode` to
    /// [`InputMode::Emoji`], seeds the buffer with `:`, and refreshes
    /// the candidate list so the user sees emoji suggestions appear
    /// the moment they press `:`.
    pub(super) fn start_emoji_mode(&mut self) -> EngineResult {
        self.converters.romaji.reset();
        self.input_buf.clear();
        self.live.text.clear();
        // Remember where the user was so commit/cancel/erase-to-empty
        // can drop them back into the same mode (e.g. Katakana stays
        // Katakana). Guard against clobbering on re-entry just in case
        // start_emoji_mode is ever called while already in Emoji mode.
        if self.input_mode != InputMode::Emoji {
            self.pre_emoji_mode = Some(self.input_mode);
        }
        self.input_mode = InputMode::Emoji;
        self.input_buf.insert(":");
        self.refresh_input_state()
    }

    /// First emoji candidate the rewriter would surface for `reading`,
    /// or `None` if none match. Used by Enter in emoji mode so committing
    /// `:smile` produces 😄 directly rather than the literal `:smile`.
    fn first_emoji_candidate(&self, reading: &str) -> Option<String> {
        self.converters
            .rewriters
            .rewrite_all(&[reading.to_string()])
            .into_iter()
            .map(|(text, _desc)| text)
            .next()
    }

    /// Input a character during composing.
    /// In alphabet mode, inserts directly; otherwise goes through romaji conversion.
    pub(super) fn input_char(&mut self, ch: char) -> EngineResult {
        self.last_key_was_backspace = false;
        self.input_buf.raw_keys.push(ch);
        if matches!(self.input_mode, InputMode::Alphabet | InputMode::Emoji) {
            self.input_buf.insert(&ch.to_string());
            return self.refresh_input_state();
        }

        let prev_output_len = self.converters.romaji.output().chars().count();
        let _event = self.converters.romaji.push(ch);
        let curr_output_len = self.converters.romaji.output().chars().count();

        // Consume ALL new converter output into composed_hiragana at cursor position.
        // The converter may recursively pass through multiple chars (e.g., "thx" →
        // output="th", buffer="x"), so capture all of them via delta detection.
        // PassThrough chars are already included in the converter output.
        if curr_output_len > prev_output_len {
            let new_chars: String = self
                .converters
                .romaji
                .output()
                .chars()
                .skip(prev_output_len)
                .collect();
            self.input_buf.insert(&new_chars);
        }

        // PassThrough chars no longer auto-commit. They accumulate in the preedit
        // alongside hiragana, allowing users to compose `「」`, type `'word'`,
        // and access symbol variants from the candidate list.

        if let Some(result) = self.try_reset_if_empty() {
            return result;
        }

        self.refresh_input_state()
    }

    /// Commit the current hiragana input (or katakana if in katakana mode)
    /// In live conversion mode, commits the converted text instead of hiragana.
    pub(super) fn commit_composing(&mut self) -> EngineResult {
        // Flush any pending romaji into composed_hiragana
        self.flush_romaji_to_composed();

        let reading = self.input_buf.text.clone();
        let text = if self.input_mode == InputMode::Emoji {
            // Emoji mode: Enter should select the first emoji candidate the
            // EmojiRewriter would surface, not commit the literal `:smile`.
            // Falls back to the literal buffer when nothing matches (e.g.
            // `:xyz`) so the user still sees what they typed.
            self.first_emoji_candidate(&reading)
                .unwrap_or_else(|| reading.clone())
        } else if self.input_mode == InputMode::Katakana {
            // Katakana mode always commits katakana, ignoring live conversion
            kashiko_engine::hiragana_to_katakana(&reading)
        } else if !self.live.text.is_empty() {
            // Live conversion active: commit converted text
            self.live.text.clone()
        } else {
            reading.clone()
        };

        if text.is_empty() {
            self.state = InputState::Empty;
            self.input_buf.clear();
            self.live.text.clear();
            self.chunks.clear();
            return EngineResult::consumed()
                .with_action(EngineAction::HideCandidates)
                .with_action(EngineAction::HideAuxText);
        }

        // Record live conversion result in learning cache.
        // Skip the learning record for emoji mode — the buffer holds
        // a Slack-style query like `:smile`, not a hiragana reading,
        // so storing it would corrupt the kana-keyed learning cache.
        if self.input_mode != InputMode::Emoji {
            self.record_learning(&reading, &text);
            // Remember the commit for one key so Ctrl+Backspace can take it
            // back into conversion mode — live-conversion Enter commits are
            // the most common commit path on macOS.
            self.last_commit = Some((reading.clone(), text.clone()));
        }

        self.converters.romaji.reset();
        self.input_buf.clear();
        self.live.text.clear();
        self.chunks.clear();
        self.state = InputState::Empty;
        self.exit_emoji_mode();

        // HideCandidates is required here: the auto-suggest/live-conversion
        // window may be open while Composing, and the macOS frontend's
        // NSPanel only closes on an explicit hide (fcitx5 resets its panel
        // on commit implicitly, which masked this on Linux).
        EngineResult::consumed()
            .with_action(EngineAction::UpdatePreedit(Preedit::new()))
            .with_action(EngineAction::Commit(text))
            .with_action(EngineAction::HideCandidates)
            .with_action(EngineAction::HideAuxText)
    }

    /// Cancel the current input
    /// In live conversion mode: first Escape clears live conversion and shows hiragana,
    /// second Escape cancels input entirely.
    pub(super) fn cancel_composing(&mut self) -> EngineResult {
        // Escape abandons the conversion, so any segments settled by sliding
        // the focus are dropped with it.
        self.decided.clear();
        self.tail_text = None;
        self.segment_bounds.clear();
        // If live conversion is active, first Escape returns to hiragana display
        if !self.live.text.is_empty() {
            self.live.text.clear();
            let preedit = self.set_composing_state();
            return EngineResult::consumed()
                .with_action(EngineAction::UpdatePreedit(preedit))
                .with_action(EngineAction::HideCandidates)
                .with_action(EngineAction::UpdateAuxText(self.format_aux_composing()));
        }

        // Emoji mode: Escape closes the picker but commits the literal
        // buffer (the typed `:smile` or `:xyz`) — Slack-style escape.
        // The user is saying "abandon the emoji lookup but keep what I
        // typed as plain text". Without this, Escape would silently
        // discard the typed characters which is surprising when the
        // user just wanted to dismiss the candidate list.
        let emoji_literal =
            if self.input_mode == InputMode::Emoji && !self.input_buf.text.is_empty() {
                Some(self.input_buf.text.clone())
            } else {
                None
            };

        self.converters.romaji.reset();
        self.input_buf.clear();
        self.live.text.clear();
        self.chunks.clear();
        self.state = InputState::Empty;
        // Emoji mode is per-session: leaving it returns the user to
        // whatever mode they were in before typing `:` so their next
        // word doesn't unexpectedly stay in ASCII-passthrough mode.
        self.exit_emoji_mode();

        let mut result = EngineResult::consumed()
            .with_action(EngineAction::UpdatePreedit(Preedit::new()))
            .with_action(EngineAction::HideCandidates)
            .with_action(EngineAction::HideAuxText);
        if let Some(literal) = emoji_literal {
            result = result.with_action(EngineAction::Commit(literal));
        }
        result
    }
}
