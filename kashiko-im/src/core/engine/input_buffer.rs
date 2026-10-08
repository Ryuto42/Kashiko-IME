//! InputBuffer: composed hiragana text with cursor.
//!
//! This struct bundles `text` and `cursor_pos`
//! which are always operated on together.

/// Composed input buffer with cursor.
pub(super) struct InputBuffer {
    /// Composed hiragana text (source of truth)
    pub text: String,
    /// Cursor position (in characters, not bytes)
    pub cursor_pos: usize,
    /// Raw printable keystrokes of this composition, before romaji→kana
    /// conversion. Typing `WordPress` leaves `text` as `Wおrdpレッ`, which no
    /// kana-keyed lookup can recover; keeping the keystrokes lets the engine
    /// offer the latin spelling back as a candidate. Lives here so it is
    /// cleared together with the text it belongs to.
    pub raw_keys: String,
    /// `text` is still exactly what `raw_keys` produce when typed at the end,
    /// so the text can be rebuilt from them (English words kept as typed —
    /// see `mixed_input`). Lost on any edit that breaks the correspondence.
    pub raw_synced: bool,
}

impl InputBuffer {
    /// Create a new empty buffer.
    pub fn new() -> Self {
        Self {
            text: String::new(),
            cursor_pos: 0,
            raw_keys: String::new(),
            raw_synced: true,
        }
    }

    /// Clear the buffer (text, cursor, raw keystrokes).
    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor_pos = 0;
        self.raw_keys.clear();
        self.raw_synced = true;
    }

    /// Replace the whole buffer with `text`, dropping the keystroke history.
    ///
    /// Use this instead of assigning `text` directly whenever the buffer stops
    /// corresponding to what the user typed — a partial commit leaving only
    /// the remainder, a katakana bake, and so on. A stale `raw_keys` would
    /// otherwise make the engine offer the *original* keystrokes as a latin
    /// word for text they no longer cover.
    pub fn set_text(&mut self, text: String) {
        self.text = text;
        self.cursor_pos = self.text.chars().count();
        self.raw_keys.clear();
        self.raw_synced = false;
    }

    /// Replace the text with one rebuilt from `raw_keys`, caret at the end.
    pub fn set_rendered(&mut self, text: String) {
        self.text = text;
        self.cursor_pos = self.text.chars().count();
    }

    /// Insert text at the current cursor position.
    pub fn insert(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        if self.cursor_pos != self.text.chars().count() {
            self.raw_synced = false;
        }
        let byte_pos = self
            .text
            .char_indices()
            .nth(self.cursor_pos)
            .map(|(i, _)| i)
            .unwrap_or(self.text.len());
        self.text.insert_str(byte_pos, text);
        let char_count = text.chars().count();
        self.cursor_pos += char_count;
    }

    /// Remove the character at the given character position.
    pub fn remove_char_at(&mut self, char_pos: usize) -> Option<char> {
        self.raw_synced = false;
        let (byte_start, removed) = self.text.char_indices().nth(char_pos)?;
        let byte_end = self
            .text
            .char_indices()
            .nth(char_pos + 1)
            .map(|(i, _)| i)
            .unwrap_or(self.text.len());
        self.text.replace_range(byte_start..byte_end, "");
        Some(removed)
    }

    /// Remove the character before the cursor.
    pub fn remove_char_before_cursor(&mut self) -> Option<char> {
        if self.cursor_pos == 0 {
            return None;
        }
        self.cursor_pos -= 1;
        self.remove_char_at(self.cursor_pos)
    }

    /// Remove the character at the cursor position (delete key).
    pub fn remove_char_at_cursor(&mut self) -> Option<char> {
        self.remove_char_at(self.cursor_pos)
    }
}
