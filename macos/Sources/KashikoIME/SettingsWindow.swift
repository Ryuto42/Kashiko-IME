import Cocoa
import SwiftUI

/// The single settings window. IME is a background (LSUIElement) app, so the
/// window activates the app itself and is kept around after closing.
@MainActor
final class SettingsWindowController {
    static let shared = SettingsWindowController()

    private var window: NSWindow?
    private let model = SettingsViewModel()

    func show() {
        if window == nil {
            let window = NSWindow(
                contentRect: NSRect(x: 0, y: 0, width: 520, height: 680),
                styleMask: [.titled, .closable],
                backing: .buffered, defer: false)
            window.title = "Kashiko の設定"
            window.isReleasedWhenClosed = false
            window.contentView = NSHostingView(rootView: SettingsView(model: model))
            window.center()
            self.window = window
        }
        model.reload()
        NSApp.activate(ignoringOtherApps: true)
        window?.makeKeyAndOrderFront(nil)
    }
}

@MainActor
final class SettingsViewModel: ObservableObject {
    @Published var draft: EngineSettings?
    @Published var models: [ModelInfo] = []
    @Published var loadError: String?
    @Published var isApplying = false
    @Published var statusMessage: String?
    @Published var errorMessage: String?
    @Published var privateMode = false
    @Published var autoCheck = Updater.shared.autoCheckEnabled {
        didSet { Updater.shared.autoCheckEnabled = autoCheck }
    }

    private(set) var defaults: EngineSettings?
    private var saved: EngineSettings?

    var isDirty: Bool { draft != saved }

    func reload() {
        privateMode = engineClient.privateMode
        autoCheck = Updater.shared.autoCheckEnabled
        statusMessage = nil
        engineClient.getSettings { [weak self] result in
            DispatchQueue.main.async {
                guard let self else { return }
                switch result {
                case .success(let value):
                    self.draft = value.settings
                    self.saved = value.settings
                    self.defaults = value.defaults
                    self.models = value.models
                    self.loadError = nil
                case .failure(let error):
                    NSLog("Kashiko IME: get_settings failed: \(error.message)")
                    if self.draft == nil { self.loadError = error.message }
                }
            }
        }
    }

    func resetToDefaults() {
        if let defaults { draft = defaults }
    }

    func setPrivateMode(_ active: Bool) {
        engineClient.setPrivateMode(active)
        privateMode = active
    }

    func apply() {
        guard let draft, !isApplying else { return }
        isApplying = true
        statusMessage = nil
        engineClient.updateSettings(draft) { [weak self] result in
            DispatchQueue.main.async {
                guard let self else { return }
                self.isApplying = false
                switch result {
                case .success:
                    self.saved = draft
                    self.statusMessage = "適用しました。エンジンを再起動しています。"
                    engineProcess.restart()
                case .failure(let error):
                    NSLog("Kashiko IME: update_settings failed: \(error.message)")
                    self.errorMessage = error.message
                }
            }
        }
    }
}

private let userDictionaryHelpURL = URL(
    string: "https://github.com/Ryuto42/Kashiko-IME/blob/main/docs/user-dictionary.md")!

struct SettingsView: View {
    @ObservedObject var model: SettingsViewModel
    @State private var confirmClearLearning = false

    var body: some View {
        VStack(spacing: 0) {
            if model.draft != nil {
                form
                Divider()
                footer
            } else if let error = model.loadError {
                VStack(spacing: 12) {
                    Text("設定を読み込めませんでした").font(.headline)
                    Text(error).foregroundColor(.secondary)
                    Button("再試行") { model.reload() }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .frame(minWidth: 520, minHeight: 560)
        .background(Color(nsColor: .windowBackgroundColor))
        .alert(
            "設定を適用できませんでした",
            isPresented: Binding(
                get: { model.errorMessage != nil },
                set: { if !$0 { model.errorMessage = nil } })
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(model.errorMessage ?? "")
        }
    }

    private func bind<T>(_ keyPath: WritableKeyPath<EngineSettings, T>) -> Binding<T> {
        Binding(
            get: { model.draft![keyPath: keyPath] },
            set: { model.draft?[keyPath: keyPath] = $0 })
    }

    private func clamped(_ binding: Binding<Int>, _ range: ClosedRange<Int>) -> Binding<Int> {
        Binding(
            get: { min(max(binding.wrappedValue, range.lowerBound), range.upperBound) },
            set: { binding.wrappedValue = $0 })
    }

    private func modelBinding(_ keyPath: WritableKeyPath<EngineSettings, String?>) -> Binding<String>
    {
        Binding(
            get: {
                model.draft?[keyPath: keyPath] ?? model.defaults?[keyPath: keyPath] ?? ""
            },
            set: { model.draft?[keyPath: keyPath] = $0 })
    }

    private func modelOptions(including current: String) -> [ModelInfo] {
        if current.isEmpty || model.models.contains(where: { $0.id == current }) {
            return model.models
        }
        return model.models + [ModelInfo(id: current, displayName: current)]
    }

    private func modelPicker(_ title: String, _ keyPath: WritableKeyPath<EngineSettings, String?>)
        -> some View
    {
        let selection = modelBinding(keyPath)
        return Picker(title, selection: selection) {
            ForEach(modelOptions(including: selection.wrappedValue), id: \.id) { info in
                Text(info.displayName).tag(info.id)
            }
        }
    }

    private var form: some View {
        let candidates = clamped(bind(\.conversion.numCandidates), 1...20)
        let threads = clamped(bind(\.conversion.nThreads), 1...8)
        let maxEntries = clamped(bind(\.learning.maxEntries), 1000...100_000)
        return Form {
            Section("変換") {
                Toggle("ライブ変換", isOn: bind(\.conversion.liveConversion))
                Toggle("前後の文脈を使う", isOn: bind(\.conversion.useContext))
                Stepper(value: candidates, in: 1...20) {
                    LabeledContent("候補の数", value: "\(candidates.wrappedValue)")
                }
                modelPicker("変換モデル", \.conversion.model)
                modelPicker("補助モデル", \.conversion.lightModel)
                VStack(alignment: .leading, spacing: 2) {
                    Stepper(value: threads, in: 1...8) {
                        LabeledContent("推論スレッド数", value: "\(threads.wrappedValue)")
                    }
                    Text("増やすと速くなりますが、CPU と電力を多く使います")
                        .font(.caption).foregroundColor(.secondary)
                }
            }
            Section("学習") {
                Toggle("学習する", isOn: bind(\.learning.enabled))
                Stepper(value: maxEntries, in: 1000...100_000, step: 1000) {
                    LabeledContent("学習の上限件数", value: "\(maxEntries.wrappedValue)")
                }
                Toggle(
                    "プライベートモード（学習しない）",
                    isOn: Binding(
                        get: { model.privateMode }, set: { model.setPrivateMode($0) }))
                Button("学習履歴を消去…") { confirmClearLearning = true }
                    .alert("学習履歴を消去しますか？", isPresented: $confirmClearLearning) {
                        Button("消去", role: .destructive) { engineClient.clearLearningAsync() }
                        Button("キャンセル", role: .cancel) {}
                    } message: {
                        Text("これまでに学習した変換・英単語の判定がすべて消えます。元に戻せません。")
                    }
            }
            Section("辞書") {
                Button("ユーザー辞書フォルダを開く") { openUserDictionaryFolder() }
                Link("ユーザー辞書の書き方", destination: userDictionaryHelpURL)
            }
            Section("アップデート") {
                Toggle("自動で確認", isOn: $model.autoCheck)
                LabeledContent("現在の版", value: Updater.shared.currentVersion)
                Button("今すぐ確認") { Updater.shared.check(manual: true) }
            }
        }
        .formStyle(.grouped)
    }

    private var footer: some View {
        HStack {
            Button("既定に戻す") { model.resetToDefaults() }
            if let message = model.statusMessage {
                Text(message).font(.caption).foregroundColor(.secondary)
            }
            Spacer()
            Button("適用") { model.apply() }
                .keyboardShortcut(.defaultAction)
                .disabled(!model.isDirty || model.isApplying)
        }
        .padding(12)
    }

    private func openUserDictionaryFolder() {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/Kashiko/user_dicts")
        do {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        } catch {
            NSLog("Kashiko IME: cannot create user_dicts: \(error)")
        }
        NSWorkspace.shared.open(dir)
    }
}
