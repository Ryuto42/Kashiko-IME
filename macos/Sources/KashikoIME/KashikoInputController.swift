import Carbon
import Cocoa
import InputMethodKit

/// Thin InputMethodKit adapter for the kashiko engine.
///
/// All IME state (Empty → Composing → Conversion, romaji conversion,
/// candidates, learning) lives in Kashiko IME Engine; this controller only
/// translates key events and applies the resulting UI actions.
@objc(KashikoInputController)
class KashikoInputController: IMKInputController {
    static let candidateWindow = CandidateWindowController()

    /// Apply an asynchronous `update_candidates` push from the engine (deep
    /// typo-search results). Runs without an IMK client, so it only touches
    /// the candidate window — the panel redraws at its last anchor position.
    /// Ignored when the panel has already been dismissed by a later key.
    static func applyAsyncCandidateUpdate(_ actions: [EngineAction]) {
        // A refined live conversion: redraw the marked text in the client that
        // is composing it.
        let preedits = actions.filter {
            if case .updatePreedit = $0 { return true }
            return false
        }
        if !preedits.isEmpty, let controller = focused, let client = controller.focusedClient {
            controller.apply(actions: preedits, client: client)
        }
        guard candidateWindow.isVisible else { return }
        for action in actions {
            switch action {
            case .showCandidates(let candidates, let cursor, let page, let totalPages):
                candidateWindow.show(
                    candidates: candidates,
                    cursor: cursor,
                    page: page,
                    totalPages: totalPages,
                    cursorRect: nil,
                    liveMode: false
                )
            case .updateAux(let text, let button):
                candidateWindow.setAux(text, isButton: button)
            case .hideAux:
                candidateWindow.setAux(nil)
            default:
                break  // preedit updates are applied above; commits are never pushed
            }
        }
    }

    /// The controller that last handled a key, and its client: where an
    /// asynchronous preedit update is drawn.
    private static weak var focused: KashikoInputController?
    private weak var focusedClient: (any IMKTextInput)?

    /// Mirrors whether the engine currently shows a preedit (updated from
    /// engine actions). Used to decide when to refresh surrounding text.
    private var hasPreedit = false
    private var isConversionMode = false

    // MARK: - Event handling

    override func recognizedEvents(_ sender: Any!) -> Int {
        Int(NSEvent.EventTypeMask.keyDown.rawValue)
    }

    override func handle(_ event: NSEvent!, client sender: Any!) -> Bool {
        guard let event else { return false }
        guard event.type == .keyDown else { return false }
        guard let client = sender as? (any IMKTextInput) else { return false }
        Self.focused = self
        focusedClient = client

        // Password fields and secure keyboard entry: nothing typed may be
        // learned or read back as context.
        let secureInput = IsSecureEventInputEnabled()
        engineClient.setSecureInput(secureInput)

        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        // Never swallow Command shortcuts.
        if flags.contains(.command) { return false }

        // JIS かな key (and Karabiner right-Command tap → かな): always
        // consume so the system doesn't process keyCode 104 after the engine
        // returns not_consumed (already in hiragana mode).
        if event.keyCode == KeyCodeMap.kanaKeyCode {
            let key = EngineKeyEvent(keysym: KeyCodeMap.superRKeysym, modifiers: KeyModifiers())
            if let result = engineClient.processKeySync(key) {
                apply(actions: result.actions, client: client)
            }
            return true
        }

        // JIS 英数 key: flush pending composition so preedit doesn't linger
        // after macOS switches to the English input source.
        if event.keyCode == KeyCodeMap.eisuKeyCode {
            flushComposition(client: client)
            return false
        }

        guard let key = KeyCodeMap.translate(event: event) else { return false }

        // Refresh the conversion context while no composition is active
        // (captures surrounding text in the Empty state). Queued before process_key on the same pipe, so the
        // engine sees it first. Skipped for function/navigation keysyms
        // (0xff00 range): they can't start a composition, and the three
        // synchronous client IPCs in sendSurroundingText would otherwise
        // fire on every arrow-key repeat.
        if !secureInput && !hasPreedit && key.keysym < 0xff00 {
            sendSurroundingText(client: client)
        }

        guard let result = engineClient.processKeySync(key) else {
            // Engine busy or dead: let the key pass through rather than
            // freezing input.
            return false
        }
        apply(actions: result.actions, client: client)
        return result.consumed
    }

    // MARK: - Lifecycle

    override func activateServer(_ sender: Any!) {
        super.activateServer(sender)
        Updater.shared.checkIfDue()
    }

    override func deactivateServer(_ sender: Any!) {
        if Self.focused === self {
            Self.focused = nil
        }
        // Mozc-style: commit the pending preedit on focus loss, then
        // persist what the user taught us.
        if let client = sender as? (any IMKTextInput) {
            flushComposition(client: client)
        } else {
            Self.candidateWindow.hide()
        }
        engineClient.saveLearningAsync()
        super.deactivateServer(sender)
    }

    override func commitComposition(_ sender: Any!) {
        if let client = sender as? (any IMKTextInput) {
            flushComposition(client: client)
        } else {
            Self.candidateWindow.hide()
        }
    }

    /// Commit any pending composition via the engine and apply the cleanup
    /// actions it emits (clear preedit, hide candidates/aux).
    private func flushComposition(client: any IMKTextInput) {
        if let result = engineClient.commitSync() {
            apply(actions: result.actions, client: client)
        } else {
            // Engine unavailable: still drop any stale candidate panel.
            Self.candidateWindow.hide()
        }
    }

    // MARK: - Input menu

    override func menu() -> NSMenu! {
        let menu = NSMenu()
        menu.addItem(
            NSMenuItem(title: "設定…", action: #selector(openSettings(_:)), keyEquivalent: ""))
        menu.addItem(.separator())
        let privateItem = NSMenuItem(
            title: "プライベートモード（学習しない）", action: #selector(togglePrivateMode(_:)),
            keyEquivalent: "")
        privateItem.state = engineClient.privateMode ? .on : .off
        menu.addItem(privateItem)
        menu.addItem(
            NSMenuItem(
                title: "学習履歴を消去…", action: #selector(clearLearning(_:)), keyEquivalent: ""))
        menu.addItem(.separator())
        menu.addItem(
            NSMenuItem(
                title: "アップデートを確認…", action: #selector(checkForUpdates(_:)), keyEquivalent: ""))
        let auto = NSMenuItem(
            title: "自動でアップデートを確認", action: #selector(toggleAutoUpdateCheck(_:)),
            keyEquivalent: "")
        auto.state = Updater.shared.autoCheckEnabled ? .on : .off
        menu.addItem(auto)
        menu.addItem(.separator())
        menu.addItem(
            NSMenuItem(
                title: "Kashiko について", action: #selector(showAbout(_:)), keyEquivalent: ""))
        return menu
    }

    @objc func openSettings(_ sender: Any?) {
        Task { @MainActor in SettingsWindowController.shared.show() }
    }

    @objc func togglePrivateMode(_ sender: Any?) {
        engineClient.setPrivateMode(!engineClient.privateMode)
    }

    @objc func clearLearning(_ sender: Any?) {
        let alert = NSAlert()
        alert.messageText = "学習履歴を消去しますか？"
        alert.informativeText = "これまでに学習した変換・英単語の判定がすべて消えます。元に戻せません。"
        alert.addButton(withTitle: "消去")
        alert.addButton(withTitle: "キャンセル")
        NSApp.activate(ignoringOtherApps: true)
        if alert.runModal() == .alertFirstButtonReturn {
            engineClient.clearLearningAsync()
        }
    }

    @objc func checkForUpdates(_ sender: Any?) {
        Updater.shared.check(manual: true)
    }

    @objc func toggleAutoUpdateCheck(_ sender: Any?) {
        Updater.shared.autoCheckEnabled.toggle()
    }

    @objc func showAbout(_ sender: Any?) {
        Updater.shared.showAbout()
    }

    // MARK: - Applying engine actions

    private func apply(actions: [EngineAction], client: any IMKTextInput) {
        let updatesCandidates = actions.contains {
            switch $0 {
            case .showCandidates, .hideCandidates: return true
            default: return false
            }
        }

        for action in actions {
            switch action {
            case .updateAux(let text, let button):
                Self.candidateWindow.setAux(
                    text, deferRender: updatesCandidates, isButton: button)
            case .hideAux:
                Self.candidateWindow.setAux(nil, deferRender: updatesCandidates)
            default:
                break
            }
        }

        // Set by deleteSurrounding: the range of just-committed text the next
        // setMarkedText should replace (undo commit / reconversion).
        var pendingReplacement: NSRange?

        for action in actions {
            switch action {
            case .commit(let text):
                client.insertText(text, replacementRange: NSRange(location: NSNotFound, length: 0))

            case .deleteSurrounding(let text):
                let sel = client.selectedRange()
                let len = text.utf16.count
                if sel.location != NSNotFound && sel.location >= len {
                    pendingReplacement = NSRange(location: sel.location - len, length: len)
                }

            case .updatePreedit(let text, let caret, let attributes):
                isConversionMode = attributes.contains { $0.style == "highlight" }
                hasPreedit = !text.isEmpty
                setMarkedText(
                    text: text, caret: caret, attributes: attributes, client: client,
                    replacementRange: pendingReplacement)
                pendingReplacement = nil

            case .showCandidates(let candidates, let cursor, let page, let totalPages):
                // Always re-query the cursor rect (cheap client IPC), rather than
                // reusing the last one while the panel is visible: the caret can
                // move (line wrap, newline) mid-composition in multi-line text
                // fields, and reusing a stale rect leaves the panel anchored to
                // where composition started instead of where the caret now is.
                var lineHeightRect = NSRect.zero
                client.attributes(forCharacterIndex: 0, lineHeightRectangle: &lineHeightRect)
                Self.candidateWindow.show(
                    candidates: candidates,
                    cursor: cursor,
                    page: page,
                    totalPages: totalPages,
                    cursorRect: lineHeightRect,
                    liveMode: !isConversionMode
                )

            case .hideCandidates:
                Self.candidateWindow.hide()

            case .updateAux, .hideAux:
                break  // applied above
            }
        }
    }


    /// Send the text around the cursor to the engine as conversion
    /// context. Gated on `selectedRange` only: `client.length()` is the
    /// least-implemented part of IMKTextInput (it returns 0 even in apps
    /// whose `attributedSubstring` works fine), and the request below is
    /// capped to 80 UTF-16 units each side, so document size doesn't matter.
    /// Whether a client supports this at all is app-dependent (Cocoa text
    /// views do; Electron/Chromium/terminals mostly don't), so the skip
    /// reasons are logged for dogfooding visibility.
    private func sendSurroundingText(client: any IMKTextInput) {
        // When capture isn't possible, CLEAR the engine's context rather
        // than skipping: leaving the context from a previous cursor
        // position in place makes the engine condition on (and display)
        // text that is no longer left of the cursor. No context beats a
        // wrong one. selectedRange flakiness is per-keystroke in some
        // apps, so this also self-heals on the next successful capture.
        let selected = client.selectedRange()
        guard selected.location != NSNotFound else {
            NSLog("Kashiko IME: surrounding text cleared (no usable selection)")
            engineClient.setSurroundingTextAsync(text: "", cursorPos: 0)
            return
        }

        // Must exceed the engine's own `max_context_length` (32 chars) so the
        // engine has enough to truncate from; disambiguating homophones depends
        // on the preceding sentence, so a window shorter than one sentence
        // silently defeats context-aware conversion.
        let maxContextUTF16 = 80
        // string(from:actualRange:) rather than attributedSubstring(from:):
        // it's the IMKTextInput document-access method clients actually
        // implement (azooKey-Desktop settled on the same call).
        var actualRange = NSRange()
        let start = max(0, selected.location - maxContextUTF16)
        let leftContext = selected.location > 0
            ? client.string(
                from: NSRange(location: start, length: selected.location - start),
                actualRange: &actualRange) ?? ""
            : ""
        // Text after the cursor (past any selection, which typing replaces).
        // Clients clamp the range at the end of the document.
        let rightContext = client.string(
            from: NSRange(location: NSMaxRange(selected), length: maxContextUTF16),
            actualRange: &actualRange) ?? ""
        guard !leftContext.isEmpty || !rightContext.isEmpty else {
            NSLog("Kashiko IME: surrounding text cleared (string(from:) unavailable)")
            engineClient.setSurroundingTextAsync(text: "", cursorPos: 0)
            return
        }

        NSLog("Kashiko IME: surrounding text captured (\(leftContext.count) + \(rightContext.count) chars)")
        engineClient.setSurroundingTextAsync(
            text: leftContext + rightContext,
            cursorPos: leftContext.unicodeScalars.count
        )
    }

    private func setMarkedText(
        text: String, caret: Int, attributes: [PreeditAttr], client: any IMKTextInput,
        replacementRange: NSRange? = nil
    ) {
        // Default: insert at the caret. A concrete range (from
        // deleteSurrounding) replaces just-committed document text with the
        // marked text instead — the undo-commit / reconversion path.
        let replacement = replacementRange ?? NSRange(location: NSNotFound, length: 0)
        guard !text.isEmpty else {
            client.setMarkedText(
                NSAttributedString(string: ""),
                selectionRange: NSRange(location: 0, length: 0),
                replacementRange: replacement
            )
            return
        }

        let attributed = NSMutableAttributedString(
            string: text,
            attributes: [.underlineStyle: NSUnderlineStyle.single.rawValue]
        )
        for attr in attributes {
            guard let range = utf16Range(of: attr.start..<attr.end, in: text) else { continue }
            let style: NSUnderlineStyle
            switch attr.style {
            // The focused/highlighted segment is drawn with a thick
            // underline (the convention azooKey/mac-akaza use for marked
            // text, since background colors are unreliable across apps).
            case "underline_double", "highlight", "reverse":
                style = .thick
            default:
                style = .single
            }
            attributed.addAttribute(.underlineStyle, value: style.rawValue, range: range)
        }

        let caretUTF16 = utf16Offset(ofScalarOffset: caret, in: text)
        client.setMarkedText(
            attributed,
            selectionRange: NSRange(location: caretUTF16, length: 0),
            replacementRange: replacement
        )
    }

}

// MARK: - Unicode scalar → UTF-16 offset conversion

/// The engine reports positions in Unicode scalar values; IMK APIs take
/// UTF-16 offsets.
func utf16Offset(ofScalarOffset offset: Int, in text: String) -> Int {
    let scalars = text.unicodeScalars
    let clamped = min(max(offset, 0), scalars.count)
    let index = scalars.index(scalars.startIndex, offsetBy: clamped)
    return text.utf16.distance(from: text.utf16.startIndex, to: index)
}

func utf16Range(of scalarRange: Range<Int>, in text: String) -> NSRange? {
    guard scalarRange.lowerBound >= 0, scalarRange.lowerBound <= scalarRange.upperBound else {
        return nil
    }
    let start = utf16Offset(ofScalarOffset: scalarRange.lowerBound, in: text)
    let end = utf16Offset(ofScalarOffset: scalarRange.upperBound, in: text)
    return NSRange(location: start, length: end - start)
}
