import CryptoKit
import Cocoa

enum UpdateLogic {
    struct Asset: Decodable {
        let name: String
        let browserDownloadURL: String

        enum CodingKeys: String, CodingKey {
            case name
            case browserDownloadURL = "browser_download_url"
        }
    }

    struct Release: Decodable {
        let tagName: String
        let htmlURL: String?
        let assets: [Asset]

        enum CodingKeys: String, CodingKey {
            case tagName = "tag_name"
            case htmlURL = "html_url"
            case assets
        }

        var version: String { UpdateLogic.normalizedVersion(tagName) }
    }

    static func normalizedVersion(_ raw: String) -> String {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.hasPrefix("v") || s.hasPrefix("V") { s.removeFirst() }
        return s
    }

    static func versionComponents(_ raw: String) -> [Int]? {
        let parts = normalizedVersion(raw).split(separator: ".", omittingEmptySubsequences: false)
        guard !parts.isEmpty else { return nil }
        var result: [Int] = []
        for part in parts {
            let digits = part.prefix { $0.isASCII && $0.isNumber }
            guard let value = Int(digits) else { return nil }
            result.append(value)
        }
        return result
    }

    /// True when `remote` is strictly newer than `local`. Unparseable input is never newer.
    static func isNewer(_ remote: String, than local: String) -> Bool {
        guard var r = versionComponents(remote), var l = versionComponents(local) else {
            return false
        }
        let count = max(r.count, l.count)
        r += Array(repeating: 0, count: count - r.count)
        l += Array(repeating: 0, count: count - l.count)
        return r.lexicographicallyPrecedes(l) == false && r != l
    }

    static func assetName(version: String) -> String {
        "Kashiko-\(normalizedVersion(version))-macos-arm64.zip"
    }

    static let checksumsAssetName = "SHA256SUMS"

    /// Parses `<hex>  <filename>` lines (a leading `*` on the filename marks binary mode).
    static func sha256(forFile filename: String, inSums text: String) -> String? {
        for line in text.split(whereSeparator: \.isNewline) {
            let fields = line.split(
                maxSplits: 1, omittingEmptySubsequences: true, whereSeparator: { $0 == " " || $0 == "\t" })
            guard fields.count == 2 else { continue }
            var name = fields[1].trimmingCharacters(in: .whitespaces)
            if name.hasPrefix("*") { name.removeFirst() }
            let hash = String(fields[0]).lowercased()
            guard name == filename, hash.count == 64, hash.allSatisfy(\.isHexDigit) else {
                continue
            }
            return hash
        }
        return nil
    }

    static func releaseAPIURL(repository: String) -> URL? {
        let parts = repository.split(separator: "/", omittingEmptySubsequences: false)
        let allowed = CharacterSet(
            charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        guard parts.count == 2,
            parts.allSatisfy({
                !$0.isEmpty && $0 != "." && $0 != ".." && $0.unicodeScalars.allSatisfy(allowed.contains)
            })
        else { return nil }
        return URL(string: "https://api.github.com/repos/\(repository)/releases/latest")
    }
}

private enum UpdateError: LocalizedError {
    case unsupportedURL
    case httpStatus(Int)
    case missingAsset(String)
    case checksumMissing
    case checksumMismatch
    case extractFailed
    case invalidBundle(String)
    case signatureInvalid
    case installFailed(String)

    var errorDescription: String? {
        switch self {
        case .unsupportedURL: return "HTTPS 以外の URL は使用できません。"
        case .httpStatus(let code): return "サーバーがエラーを返しました（HTTP \(code)）。"
        case .missingAsset(let name): return "リリースに \(name) がありません。"
        case .checksumMissing: return "SHA256SUMS に対象ファイルの記載がありません。"
        case .checksumMismatch: return "チェックサムが一致しません。ダウンロードを破棄しました。"
        case .extractFailed: return "アーカイブの展開に失敗しました。"
        case .invalidBundle(let reason): return "新しいアプリを検証できません（\(reason)）。"
        case .signatureInvalid: return "コード署名の検証に失敗しました。"
        case .installFailed(let reason): return "アプリの入れ替えに失敗しました（\(reason)）。"
        }
    }
}

private final class HTTPSOnlyRedirect: NSObject, URLSessionTaskDelegate {
    func urlSession(
        _ session: URLSession, task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        completionHandler(request.url?.scheme == "https" ? request : nil)
    }
}

final class Updater {
    static let shared = Updater()

    private enum Key {
        static let autoCheck = "KashikoUpdateAutoCheck"
        static let lastCheck = "KashikoUpdateLastCheck"
        static let skippedVersion = "KashikoUpdateSkippedVersion"
    }

    private static let checkInterval: TimeInterval = 24 * 60 * 60

    private enum CheckResult {
        case available(UpdateLogic.Release)
        case upToDate
        case failed
    }

    private let redirectGuard = HTTPSOnlyRedirect()
    private lazy var session = URLSession(
        configuration: .ephemeral, delegate: redirectGuard, delegateQueue: nil)
    // Main-thread only.
    private var busy = false

    private let defaults = UserDefaults.standard

    private init() {
        defaults.register(defaults: [Key.autoCheck: true])
    }

    var currentVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0"
    }

    var autoCheckEnabled: Bool {
        get { defaults.bool(forKey: Key.autoCheck) }
        set { defaults.set(newValue, forKey: Key.autoCheck) }
    }

    private var repository: String {
        Bundle.main.object(forInfoDictionaryKey: "KashikoUpdateRepository") as? String
            ?? "Ryuto42/Kashiko-IME"
    }

    func checkIfDue() {
        guard autoCheckEnabled, !busy else { return }
        if let last = defaults.object(forKey: Key.lastCheck) as? Date,
            Date().timeIntervalSince(last) < Self.checkInterval
        {
            return
        }
        check(manual: false)
    }

    func check(manual: Bool) {
        guard !busy else { return }
        busy = true
        defaults.set(Date(), forKey: Key.lastCheck)
        Task.detached { [self] in
            let result = await fetchLatest()
            await MainActor.run {
                busy = false
                handle(result, manual: manual)
            }
        }
    }

    func showAbout() {
        let alert = NSAlert()
        alert.messageText = "Kashiko \(currentVersion)"
        alert.informativeText = "https://github.com/\(repository)"
        alert.addButton(withTitle: "OK")
        present(alert)
    }

    // MARK: - Check

    private func fetchLatest() async -> CheckResult {
        guard let url = UpdateLogic.releaseAPIURL(repository: repository) else {
            NSLog("Kashiko IME: update check: invalid repository \(repository)")
            return .failed
        }
        var request = URLRequest(url: url, timeoutInterval: 15)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue("Kashiko/\(currentVersion)", forHTTPHeaderField: "User-Agent")
        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
                NSLog("Kashiko IME: update check: unexpected response")
                return .failed
            }
            let release = try JSONDecoder().decode(UpdateLogic.Release.self, from: data)
            return UpdateLogic.isNewer(release.version, than: currentVersion)
                ? .available(release) : .upToDate
        } catch {
            NSLog("Kashiko IME: update check failed: \(error)")
            return .failed
        }
    }

    private func handle(_ result: CheckResult, manual: Bool) {
        switch result {
        case .failed:
            guard manual else { return }
            let alert = NSAlert()
            alert.messageText = "アップデートを確認できませんでした"
            alert.informativeText = "ネットワーク接続を確認して、もう一度お試しください。"
            alert.addButton(withTitle: "OK")
            present(alert)
        case .upToDate:
            guard manual else { return }
            let alert = NSAlert()
            alert.messageText = "最新です"
            alert.informativeText = "Kashiko \(currentVersion) は最新の版です。"
            alert.addButton(withTitle: "OK")
            present(alert)
        case .available(let release):
            if !manual, defaults.string(forKey: Key.skippedVersion) == release.version { return }
            promptForUpdate(release)
        }
    }

    private func promptForUpdate(_ release: UpdateLogic.Release) {
        let alert = NSAlert()
        alert.messageText = "Kashiko \(release.version) が利用できます（現在 \(currentVersion)）"
        #if arch(arm64)
        alert.informativeText = "アップデートするとアプリを入れ替えます。"
        alert.addButton(withTitle: "アップデート")
        #else
        alert.informativeText = "この環境では自動更新に対応していません。リリースページを開きます。"
        alert.addButton(withTitle: "リリースページを開く")
        #endif
        alert.addButton(withTitle: "後で")
        alert.addButton(withTitle: "この版をスキップ")
        switch present(alert) {
        case .alertFirstButtonReturn:
            startUpdate(release)
        case .alertThirdButtonReturn:
            defaults.set(release.version, forKey: Key.skippedVersion)
        default:
            break
        }
    }

    // MARK: - Update

    private func startUpdate(_ release: UpdateLogic.Release) {
        #if !arch(arm64)
        let page = release.htmlURL.flatMap(URL.init(string:))
            ?? URL(string: "https://github.com/\(repository)/releases/latest")
        if let page, page.scheme == "https" { NSWorkspace.shared.open(page) }
        #else
        guard !busy else { return }
        busy = true
        Task.detached { [self] in
            do {
                try await install(release)
                await MainActor.run {
                    let alert = NSAlert()
                    alert.messageText = "アップデートしました"
                    alert.informativeText = "次の入力から新しい版が使われます。"
                    alert.addButton(withTitle: "OK")
                    present(alert)
                    engineProcess.stop()
                    exit(0)
                }
            } catch {
                NSLog("Kashiko IME: update failed: \(error)")
                await MainActor.run {
                    busy = false
                    let alert = NSAlert()
                    alert.alertStyle = .warning
                    alert.messageText = "アップデートに失敗しました"
                    alert.informativeText = error.localizedDescription
                    alert.addButton(withTitle: "OK")
                    present(alert)
                }
            }
        }
        #endif
    }

    private func install(_ release: UpdateLogic.Release) async throws {
        let zipName = UpdateLogic.assetName(version: release.version)
        guard let zipAsset = release.assets.first(where: { $0.name == zipName }) else {
            throw UpdateError.missingAsset(zipName)
        }
        guard
            let sumsAsset = release.assets.first(where: {
                $0.name == UpdateLogic.checksumsAssetName
            })
        else { throw UpdateError.missingAsset(UpdateLogic.checksumsAssetName) }

        let fm = FileManager.default
        let workDir = fm.temporaryDirectory.appendingPathComponent("kashiko-update-\(UUID().uuidString)")
        try fm.createDirectory(at: workDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: workDir) }

        let zipFile = try await download(zipAsset.browserDownloadURL, to: workDir.appendingPathComponent(zipName))
        let sumsFile = try await download(
            sumsAsset.browserDownloadURL, to: workDir.appendingPathComponent("SHA256SUMS"))

        let sums = (try? String(contentsOf: sumsFile, encoding: .utf8)) ?? ""
        guard let expected = UpdateLogic.sha256(forFile: zipName, inSums: sums) else {
            throw UpdateError.checksumMissing
        }
        guard try Self.sha256Hex(of: zipFile) == expected else {
            throw UpdateError.checksumMismatch
        }

        let extractDir = workDir.appendingPathComponent("extracted")
        try fm.createDirectory(at: extractDir, withIntermediateDirectories: true)
        guard try Self.run("/usr/bin/ditto", ["-x", "-k", zipFile.path, extractDir.path]) == 0 else {
            throw UpdateError.extractFailed
        }

        let newApp = extractDir.appendingPathComponent("Kashiko.app")
        guard fm.fileExists(atPath: newApp.path) else {
            throw UpdateError.invalidBundle("Kashiko.app がありません")
        }
        let plist = NSDictionary(contentsOf: newApp.appendingPathComponent("Contents/Info.plist"))
        guard let newID = plist?["CFBundleIdentifier"] as? String,
            newID == Bundle.main.bundleIdentifier
        else { throw UpdateError.invalidBundle("Bundle ID が一致しません") }
        guard try Self.run(
            "/usr/bin/codesign", ["--verify", "--deep", "--strict", newApp.path]) == 0
        else { throw UpdateError.signatureInvalid }

        if try Self.run("/usr/bin/xattr", ["-dr", "com.apple.quarantine", newApp.path]) != 0 {
            NSLog("Kashiko IME: update: xattr -dr com.apple.quarantine failed")
        }

        let current = Bundle.main.bundleURL
        let backupDir = workDir.appendingPathComponent("old")
        let backup = backupDir.appendingPathComponent(current.lastPathComponent)
        do {
            try fm.createDirectory(at: backupDir, withIntermediateDirectories: true)
            try fm.moveItem(at: current, to: backup)
        } catch {
            throw UpdateError.installFailed(error.localizedDescription)
        }
        do {
            try fm.moveItem(at: newApp, to: current)
        } catch {
            do {
                try fm.moveItem(at: backup, to: current)
            } catch {
                NSLog("Kashiko IME: update: rollback failed: \(error)")
            }
            throw UpdateError.installFailed(error.localizedDescription)
        }
    }

    private func download(_ urlString: String, to destination: URL) async throws -> URL {
        guard let url = URL(string: urlString), url.scheme == "https" else {
            throw UpdateError.unsupportedURL
        }
        var request = URLRequest(url: url, timeoutInterval: 60)
        request.setValue("Kashiko/\(currentVersion)", forHTTPHeaderField: "User-Agent")
        let (tmp, response) = try await session.download(for: request)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw UpdateError.httpStatus((response as? HTTPURLResponse)?.statusCode ?? 0)
        }
        guard response.url?.scheme == "https" else { throw UpdateError.unsupportedURL }
        try FileManager.default.moveItem(at: tmp, to: destination)
        return destination
    }

    private static func sha256Hex(of file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: 1 << 20), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func run(_ path: String, _ arguments: [String]) throws -> Int32 {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = arguments
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()
        return process.terminationStatus
    }

    @discardableResult
    private func present(_ alert: NSAlert) -> NSApplication.ModalResponse {
        NSApp.activate(ignoringOtherApps: true)
        return alert.runModal()
    }
}
