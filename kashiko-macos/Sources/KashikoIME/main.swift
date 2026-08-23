import Cocoa
import Darwin
import InputMethodKit

// Writing to a dead kashiko-imserver must not kill the IME with SIGPIPE.
signal(SIGPIPE, SIG_IGN)

private func setupApplicationMenu() {
    // NSApp must be initialized before touching mainMenu, or this crashes.
    let app = NSApplication.shared

    let mainMenu = NSMenu()
    let appMenuItem = NSMenuItem()
    mainMenu.addItem(appMenuItem)

    let editMenuItem = NSMenuItem(title: "Edit", action: nil, keyEquivalent: "")
    let editMenu = NSMenu(title: "Edit")
    editMenu.addItem(NSMenuItem(title: "Undo", action: Selector(("undo:")), keyEquivalent: "z"))
    editMenu.addItem(NSMenuItem(title: "Redo", action: Selector(("redo:")), keyEquivalent: "Z"))
    editMenu.addItem(NSMenuItem.separator())
    editMenu.addItem(NSMenuItem(title: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x"))
    editMenu.addItem(
        NSMenuItem(title: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c"))
    editMenu.addItem(
        NSMenuItem(title: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v"))
    editMenu.addItem(
        NSMenuItem(title: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a"))
    editMenuItem.submenu = editMenu
    mainMenu.addItem(editMenuItem)

    app.mainMenu = mainMenu
}

private func setupLogging() {
    let logDir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Logs/KashikoIME")
    try? FileManager.default.createDirectory(at: logDir, withIntermediateDirectories: true)

    let logFile = logDir.appendingPathComponent("kashiko-ime.log")
    if !FileManager.default.fileExists(atPath: logFile.path) {
        FileManager.default.createFile(atPath: logFile.path, contents: nil)
    }

    if let handle = FileHandle(forWritingAtPath: logFile.path) {
        handle.seekToEndOfFile()
        // stderr (NSLog and kashiko-imserver tracing) goes to the log file.
        dup2(handle.fileDescriptor, STDERR_FILENO)
    }
}

private func connectionName() -> String {
    if let name = Bundle.main.object(forInfoDictionaryKey: "InputMethodConnectionName") as? String {
        return name
    }
    return (Bundle.main.bundleIdentifier ?? "dev.togatoga.inputmethod.Kashiko") + "_Connection"
}

setupLogging()
setupApplicationMenu()
NSLog("KashikoIME: starting")

guard
    let server = IMKServer(name: connectionName(), bundleIdentifier: Bundle.main.bundleIdentifier)
else {
    NSLog("KashikoIME: failed to create IMKServer")
    exit(1)
}
_ = server  // keep the IMKServer alive

let engineProcess = EngineProcess()
let engineClient = EngineClient(serverProcess: engineProcess)

// Deep typo-search results are pushed by the engine after the synchronous
// key response; apply them to the candidate window on the main thread.
engineClient.onUpdateCandidates = { actions in
    DispatchQueue.main.async {
        KashikoInputController.applyAsyncCandidateUpdate(actions)
    }
}

// Clicking a row's ✕ asks the engine to forget that learned candidate; the
// refreshed list comes back in the response.
KashikoInputController.candidateWindow.onDeleteCandidate = { pageIndex in
    guard let result = engineClient.forgetCandidateSync(pageIndex: pageIndex) else { return }
    KashikoInputController.applyAsyncCandidateUpdate(result.actions)
}

// Clicking the aux-line button asks for another inference pass; the extra
// candidates arrive later as an `update_candidates` notification.
KashikoInputController.candidateWindow.onDeepenInference = {
    guard let result = engineClient.deepenInferenceSync() else { return }
    KashikoInputController.applyAsyncCandidateUpdate(result.actions)
}

engineProcess.start()
engineClient.startReaderLoop()
// Model loading takes seconds (and downloads from HuggingFace on the very
// first run), so initialize in the background instead of on the first key.
engineClient.initAsync()

NSLog("KashikoIME: IMKServer created")

// macOS can discard pipe connections during sleep; restart the engine
// process on wake to recover.
NSWorkspace.shared.notificationCenter.addObserver(
    forName: NSWorkspace.didWakeNotification,
    object: nil,
    queue: .main
) { _ in
    NSLog("KashikoIME: wake from sleep — restarting kashiko-imserver")
    engineProcess.restart()
}

NSApplication.shared.run()
