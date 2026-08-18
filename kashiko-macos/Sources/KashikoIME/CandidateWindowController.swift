import Cocoa

/// A candidate row that reveals a delete button while the pointer is over it.
///
/// The button stays hidden until hover so the list looks clean, and is drawn
/// small and dim so it reads as an affordance rather than part of the
/// candidate. Clicking it asks the engine to stop offering that candidate.
final class CandidateRowView: NSView {
    /// Row index within the visible page, as the engine numbers them.
    let pageIndex: Int
    private let deleteButton: NSButton
    private var onDelete: ((Int) -> Void)?
    private var trackingAreaAdded = false

    init(pageIndex: Int, onDelete: ((Int) -> Void)?) {
        self.pageIndex = pageIndex
        self.onDelete = onDelete
        deleteButton = NSButton(frame: .zero)
        super.init(frame: .zero)

        wantsLayer = true
        deleteButton.title = "✕"
        deleteButton.font = NSFont.systemFont(ofSize: 9)
        deleteButton.isBordered = false
        deleteButton.contentTintColor = NSColor.tertiaryLabelColor
        deleteButton.alphaValue = 0.55
        deleteButton.isHidden = true
        deleteButton.target = self
        deleteButton.action = #selector(deleteClicked)
        deleteButton.translatesAutoresizingMaskIntoConstraints = false
        addSubview(deleteButton)
        NSLayoutConstraint.activate([
            deleteButton.trailingAnchor.constraint(equalTo: trailingAnchor),
            deleteButton.centerYAnchor.constraint(equalTo: centerYAnchor),
            deleteButton.widthAnchor.constraint(equalToConstant: 14),
        ])
    }

    required init?(coder: NSCoder) { fatalError("not supported") }

    /// Whether this row shows a ✕ on hover.
    var isDeletable: Bool = false

    @objc private func deleteClicked() {
        onDelete?(pageIndex)
    }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        for area in trackingAreas { removeTrackingArea(area) }
        addTrackingArea(
            NSTrackingArea(
                rect: bounds,
                options: [.mouseEnteredAndExited, .activeAlways],
                owner: self,
                userInfo: nil
            ))
        trackingAreaAdded = true
    }

    override func mouseEntered(with event: NSEvent) {
        if isDeletable { deleteButton.isHidden = false }
    }

    override func mouseExited(with event: NSEvent) {
        deleteButton.isHidden = true
    }
}

/// Custom candidate window (borderless non-activating NSPanel).
///
/// The engine pre-paginates: `show` receives only the visible page plus
/// page metadata, so this controller just renders rows. An optional aux
/// line (reading hint / model info from the engine) is shown as a footer.
class CandidateWindowController {
    // Visual scale of the panel. Candidate rows use a larger type size
    // than the footers (page indicator / aux line), matching the system
    // Japanese IME's proportions.
    private static let candidateFontSize: CGFloat = 18
    private static let footerFontSize: CGFloat = 13
    private static let minPanelWidth: CGFloat = 160
    /// Hard cap on panel width so a long candidate cannot stretch the window
    /// across (or off) the screen. Fallback used when no screen is available.
    private static let maxPanelWidth: CGFloat = 640

    private let panel: NSPanel
    private let stackView: NSStackView
    private var rowViews: [NSView] = []
    private var auxText: String?

    private struct PageState {
        let candidates: [CandidateItem]
        let cursor: Int
        let page: Int
        let totalPages: Int
    }
    private var pageState: PageState?
    var liveMode: Bool = false
    /// Called with a page row index when its ✕ is clicked, to stop the engine
    /// offering the candidate on that row.
    var onDeleteCandidate: ((Int) -> Void)?

    init() {
        panel = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: 200, height: 100),
            styleMask: [.nonactivatingPanel, .borderless],
            backing: .buffered,
            defer: true
        )
        panel.level = .popUpMenu
        panel.hidesOnDeactivate = false
        panel.isOpaque = false
        panel.backgroundColor = .clear
        // Mouse events are accepted so a row can reveal its delete button on
        // hover. The panel is non-activating, so this never steals focus from
        // the app being typed into.
        panel.ignoresMouseEvents = false
        panel.contentView?.wantsLayer = true
        panel.contentView?.layer?.backgroundColor = NSColor.windowBackgroundColor.cgColor
        panel.contentView?.layer?.cornerRadius = 16
        panel.contentView?.layer?.masksToBounds = true

        stackView = NSStackView()
        stackView.orientation = .vertical
        // `.width` makes every row as wide as the widest one, so a row's
        // hover area covers the full panel width instead of just its text —
        // without it the ✕ could only be reached by pointing exactly at the
        // characters.
        stackView.alignment = .width
        stackView.spacing = 2
        stackView.edgeInsets = NSEdgeInsets(top: 6, left: 12, bottom: 6, right: 12)
        stackView.translatesAutoresizingMaskIntoConstraints = false

        panel.contentView?.addSubview(stackView)
        if let contentView = panel.contentView {
            NSLayoutConstraint.activate([
                stackView.topAnchor.constraint(equalTo: contentView.topAnchor),
                stackView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor),
                stackView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor),
                // No bottom constraint: let stackView height be driven by its content
            ])
        }
    }

    var isVisible: Bool { panel.isVisible }

    /// `cursorRect: nil` reuses the rect from the previous `show` — the
    /// caller can skip its (synchronous, per-keystroke) client IPC while
    /// the panel is already on screen, since the composition anchor
    /// doesn't move mid-composition.
    func show(
        candidates: [CandidateItem], cursor: Int, page: Int, totalPages: Int, cursorRect: NSRect?, liveMode: Bool = false
    ) {
        self.liveMode = liveMode
        pageState = PageState(
            candidates: candidates, cursor: cursor, page: page, totalPages: totalPages)
        render(cursorRect: cursorRect)
    }

    /// Update the aux footer; re-renders in place if the window is visible.
    /// Pass `deferRender: true` when a `show`/`hide` follows in the same
    /// action batch, so the panel is rendered once per batch instead of
    /// once for the aux change and again for the candidates.
    func setAux(_ text: String?, deferRender: Bool = false) {
        auxText = text
        if !deferRender, panel.isVisible, pageState != nil {
            render(cursorRect: nil)
        }
    }

    func hide() {
        pageState = nil
        panel.orderOut(nil)
    }

    private func render(cursorRect: NSRect?) {
        clearRows()
        guard let state = pageState, !state.candidates.isEmpty else {
            hide()
            return
        }

        for (index, candidate) in state.candidates.enumerated() {
            addCandidateRow(candidate, number: liveMode ? nil : index + 1, selected: index == state.cursor)
        }
        if !liveMode && state.totalPages > 1 {
            addFooterLabel("[\(state.page + 1)/\(state.totalPages)]")
        }
        if let aux = auxText, !aux.isEmpty {
            addFooterLabel(aux)
        }

        positionPanel(cursorRect: cursorRect)
    }

    private func clearRows() {
        for view in rowViews {
            stackView.removeArrangedSubview(view)
            view.removeFromSuperview()
        }
        rowViews.removeAll()
    }

    /// Longest kana row shown in full; beyond this the middle is elided.
    private static let kanaDisplayLimit = 14

    /// What a row shows, which is not always the whole candidate.
    ///
    /// The hiragana and katakana rows are the reading typed back verbatim, so on
    /// a long sentence they add two full-width lines that say nothing the user
    /// doesn't already know — and they set the panel's width for every other
    /// row. Only these two are elided (they are the ones carrying a width
    /// annotation); a real conversion is always shown in full, because there the
    /// characters are the information. Display only — committing uses
    /// `candidate.text`.
    private static func displayText(for candidate: CandidateItem) -> String {
        guard candidate.description != nil else { return candidate.text }
        let chars = Array(candidate.text)
        guard chars.count > kanaDisplayLimit else { return candidate.text }
        // Head and tail kept, so both ends of the reading stay recognisable.
        let head = kanaDisplayLimit / 2
        let tail = kanaDisplayLimit - head - 1
        return String(chars.prefix(head)) + "…" + String(chars.suffix(tail))
    }

    private func addCandidateRow(_ candidate: CandidateItem, number: Int?, selected: Bool) {
        // A small negative baseline offset nudges the glyphs themselves down
        // within the label's frame. Container padding alone (below) can't
        // fully center CJK text vertically — the font's line-height reserves
        // descent space the glyphs never use, so text still sits high even
        // with zero bottom padding.
        let text = NSMutableAttributedString()
        // The number prefix is deliberately smaller and dimmer than the
        // candidate text — it's a selection shortcut, not content.
        if let num = number {
            text.append(
                NSAttributedString(
                    string: "\(num). ",
                    attributes: [
                        .font: NSFont.systemFont(ofSize: Self.footerFontSize),
                        .foregroundColor: selected
                            ? NSColor.white.withAlphaComponent(0.6)
                            : NSColor.tertiaryLabelColor,
                        .baselineOffset: -1.5,
                    ]
                ))
        }
        text.append(
            NSAttributedString(
                string: Self.displayText(for: candidate),
                attributes: [
                    .font: NSFont.systemFont(ofSize: Self.candidateFontSize),
                    .foregroundColor: selected ? NSColor.white : NSColor.labelColor,
                    .baselineOffset: -1.5,
                ]
            ))
        if let description = candidate.description {
            text.append(
                NSAttributedString(
                    string: "  \(description)",
                    attributes: [
                        .font: NSFont.systemFont(ofSize: Self.footerFontSize),
                        .foregroundColor: selected
                            ? NSColor.white.withAlphaComponent(0.8)
                            : NSColor.secondaryLabelColor,
                        .baselineOffset: -1.5,
                    ]
                ))
        }

        let label = NSTextField(labelWithAttributedString: text)
        label.translatesAutoresizingMaskIntoConstraints = false
        // Keep each candidate on one line; long candidates truncate rather
        // than wrap (which would inflate the panel height/width).
        label.lineBreakMode = .byTruncatingTail
        label.maximumNumberOfLines = 1
        label.cell?.usesSingleLineMode = true
        label.backgroundColor = .clear
        label.drawsBackground = false

        // The highlight is drawn on a container view (not the text field
        // itself) with explicit symmetric padding, so the selected text sits
        // exactly where the unselected text does — NSTextField's own
        // drawsBackground fill has asymmetric internal cell padding that
        // makes the text look shifted within the highlight.
        let container = CandidateRowView(pageIndex: rowViews.count, onDelete: onDeleteCandidate)
        // Any candidate can be dismissed: the engine forgets a learned entry
        // and records a suppression for anything else, so the ✕ always means
        // "stop showing me this".
        container.isDeletable = true
        container.translatesAutoresizingMaskIntoConstraints = false
        container.wantsLayer = true
        container.layer?.backgroundColor =
            (selected ? NSColor.selectedContentBackgroundColor : .clear).cgColor
        container.layer?.cornerRadius = 4
        container.addSubview(label)
        // Top/bottom padding is intentionally asymmetric (not 2/2): CJK glyphs
        // reserve little space below the baseline compared to Latin descenders,
        // so a symmetric split leaves visible text sitting high inside the
        // highlight. Shifting more padding to the top (same total, so row
        // height is unaffected) pulls the text down to the highlight's center.
        NSLayoutConstraint.activate([
            label.topAnchor.constraint(equalTo: container.topAnchor, constant: 4),
            label.bottomAnchor.constraint(equalTo: container.bottomAnchor, constant: 0),
            label.leadingAnchor.constraint(equalTo: container.leadingAnchor, constant: 4),
            // Room on the right for the hover ✕ so it never overlaps the text.
            label.trailingAnchor.constraint(
                lessThanOrEqualTo: container.trailingAnchor, constant: -18),
        ])

        stackView.addArrangedSubview(container)
        rowViews.append(container)
    }

    private func addFooterLabel(_ text: String) {
        let label = NSTextField(labelWithString: text)
        label.font = NSFont.systemFont(ofSize: Self.footerFontSize)
        label.textColor = NSColor.secondaryLabelColor
        label.translatesAutoresizingMaskIntoConstraints = false
        stackView.addArrangedSubview(label)
        rowViews.append(label)
    }

    private var lastCursorRect: NSRect = .zero

    private func positionPanel(cursorRect: NSRect?) {
        if let rect = cursorRect {
            lastCursorRect = rect
        }
        let cursorRect = lastCursorRect

        stackView.layoutSubtreeIfNeeded()
        let contentSize = stackView.fittingSize
        let screen = NSScreen.main
        // Cap width at 80% of the screen (or the fallback constant) so a long
        // candidate cannot stretch the panel across the display.
        let widthCap = screen.map { $0.visibleFrame.width * 0.8 } ?? Self.maxPanelWidth
        let panelWidth = min(max(contentSize.width + 16, Self.minPanelWidth), widthCap)
        let panelHeight = contentSize.height + 8

        guard cursorRect != .zero else {
            panel.setFrame(
                NSRect(x: 100, y: 100, width: panelWidth, height: panelHeight), display: true)
            panel.orderFront(nil)
            return
        }

        // Flip above the cursor when the panel would fall off the bottom of
        // the screen.
        let showAbove: Bool
        if let screen = screen {
            showAbove = cursorRect.origin.y - panelHeight < screen.visibleFrame.origin.y
        } else {
            showAbove = false
        }

        let originY: CGFloat
        if showAbove {
            originY = cursorRect.origin.y + cursorRect.size.height
        } else {
            originY = cursorRect.origin.y - panelHeight
        }

        // Clamp X so the panel stays fully on-screen even for wide candidates.
        var originX = cursorRect.origin.x
        if let screen = screen {
            let frame = screen.visibleFrame
            originX = min(originX, frame.maxX - panelWidth)
            originX = max(originX, frame.minX)
        }

        panel.setFrame(
            NSRect(x: originX, y: originY, width: panelWidth, height: panelHeight),
            display: true)
        panel.orderFront(nil)
    }
}
