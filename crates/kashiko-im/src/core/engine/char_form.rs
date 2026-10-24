//! F6–F10: show the reading in another character form (hiragana, katakana,
//! half-width katakana, full-/half-width alphanumerics) and commit it as shown.
//!
//! The converted text rides in `live.text`, so the preedit, Escape (back to the
//! reading), caret moves and Enter already behave as they do for live
//! conversion. `char_form` only remembers what is on screen for the F9/F10 cycle.

use kashiko_engine::{
    ascii_to_fullwidth, fullwidth_to_ascii, hiragana_to_half_katakana, hiragana_to_katakana,
    katakana_to_hiragana,
};

use super::*;

/// The form currently shown by an F6–F10 press. The `u8` is the position in
/// the lower → upper → capitalized cycle of F9/F10.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CharForm {
    Hiragana,
    Katakana,
    HalfKatakana,
    FullAlnum(u8),
    HalfAlnum(u8),
}

pub(super) fn is_char_form_key(keysym: Keysym) -> bool {
    matches!(
        keysym,
        Keysym::F6 | Keysym::F7 | Keysym::F8 | Keysym::F9 | Keysym::F10
    )
}

fn cased(raw: &str, step: u8) -> String {
    match step {
        0 => raw.to_lowercase(),
        1 => raw.to_uppercase(),
        _ => {
            let mut chars = raw.chars();
            let first: String = chars
                .next()
                .into_iter()
                .flat_map(char::to_uppercase)
                .collect();
            first + &chars.as_str().to_lowercase()
        }
    }
}

impl InputMethodEngine {
    /// The keystrokes of this composition when they still describe the buffer.
    fn usable_raw_keys(&self) -> Option<&str> {
        let buf = &self.input_buf;
        let usable = !buf.raw_keys.is_empty() && (buf.raw_synced || buf.raw_keys == buf.text);
        usable.then_some(buf.raw_keys.as_str())
    }

    /// Text for `form`, or `None` when the reading can't take that form.
    fn render_char_form(&self, form: CharForm) -> Option<String> {
        let reading = &self.input_buf.text;
        let text = match form {
            CharForm::Hiragana => katakana_to_hiragana(reading),
            CharForm::Katakana => hiragana_to_katakana(reading),
            CharForm::HalfKatakana => hiragana_to_half_katakana(reading),
            CharForm::FullAlnum(step) | CharForm::HalfAlnum(step) => {
                let full = matches!(form, CharForm::FullAlnum(_));
                let width = |t: &str| {
                    if full {
                        ascii_to_fullwidth(t)
                    } else {
                        fullwidth_to_ascii(t)
                    }
                };
                match self.usable_raw_keys() {
                    Some(raw) => width(&cased(raw, step)),
                    // No keystrokes to go back to: only the width of what is
                    // already there can change, and kana have no such form.
                    None => {
                        let has_alnum = reading.chars().any(|c| {
                            c.is_ascii_graphic() || ('\u{FF01}'..='\u{FF5E}').contains(&c)
                        });
                        if !has_alnum {
                            return None;
                        }
                        width(reading)
                    }
                }
            }
        };
        (!text.is_empty()).then_some(text)
    }

    /// Handle F6–F10 while composing. A form the reading can't take (F9/F10
    /// on plain kana whose keystrokes are gone) leaves the display as it was.
    pub(super) fn apply_char_form(&mut self, keysym: Keysym) -> EngineResult {
        self.last_key_was_backspace = false;
        if self.input_mode == InputMode::Emoji {
            return EngineResult::consumed();
        }
        self.flush_romaji_to_composed();

        let form = match (keysym, self.char_form) {
            (Keysym::F6, _) => CharForm::Hiragana,
            (Keysym::F7, _) => CharForm::Katakana,
            (Keysym::F8, _) => CharForm::HalfKatakana,
            (Keysym::F9, Some(CharForm::FullAlnum(n))) => CharForm::FullAlnum((n + 1) % 3),
            (Keysym::F9, _) => CharForm::FullAlnum(0),
            (_, Some(CharForm::HalfAlnum(n))) => CharForm::HalfAlnum((n + 1) % 3),
            _ => CharForm::HalfAlnum(0),
        };

        if let Some(text) = self.render_char_form(form) {
            self.live.text = text;
            self.char_form = Some(form);
        }

        let preedit = self.set_composing_state();
        EngineResult::consumed()
            .with_action(EngineAction::UpdatePreedit(preedit))
            .with_action(EngineAction::HideCandidates)
            .with_action(EngineAction::UpdateAuxText(self.format_aux_composing()))
    }

    /// F6–F10 during conversion: back to composing on the whole reading, then
    /// show it in the requested form.
    pub(super) fn apply_char_form_in_conversion(&mut self, keysym: Keysym) -> EngineResult {
        self.cancel_deep_search();
        self.phrases.clear();
        self.converters.romaji.reset();
        self.input_buf.cursor_pos = self.input_buf.text.chars().count();
        self.live.text.clear();
        self.char_form = None;
        self.state = InputState::Composing {
            preedit: Preedit::new(),
            romaji_buffer: String::new(),
        };
        self.apply_char_form(keysym)
    }
}
