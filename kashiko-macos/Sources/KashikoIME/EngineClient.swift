import Foundation
import os

/// Newline-delimited JSON-RPC 2.0 client for kashiko-imserver.
///
/// Requests are written to the child's stdin; a dedicated reader queue
/// splits stdout on 0x0A and dispatches responses to pending completions.
/// Key processing uses the synchronous API (the IMK `handle` callback must
/// answer "consumed?" synchronously, the same trade-off Mozc makes); slow
/// or fire-and-forget calls use the async API.
class EngineClient {
    private let serverProcess: EngineProcess
    private var nextID = 1
    private let requestQueue = DispatchQueue(label: "io.github.ryuto42.kashiko.jsonrpc.request")

    private let logger = Logger(
        subsystem: Bundle.main.bundleIdentifier ?? "KashikoIME", category: "EngineClient")

    private let lock = NSLock()
    private var pendingRequests: [Int: (Data?) -> Void] = [:]

    /// Called (on the reader queue) when the server pushes an
    /// `update_candidates` notification — deep-search results arriving after
    /// the synchronous key response.
    var onUpdateCandidates: (([EngineAction]) -> Void)?

    /// `autoInit` re-sends `init` whenever the server (re)starts. Tests
    /// disable it to avoid loading models.
    init(serverProcess: EngineProcess, autoInit: Bool = true) {
        self.serverProcess = serverProcess
        self.serverProcess.onRestart = { [weak self] in
            // A fresh server starts with secure input off.
            self?.secureInputActive = false
            self?.startReaderLoop()
            if autoInit {
                self?.initAsync()
            }
        }
    }

    // MARK: - Engine methods

    func initAsync() {
        sendRequest(method: "init", params: [:]) { [weak self] data in
            guard let self else { return }
            guard let data,
                let result = try? makeProtocolDecoder().decode(InitResult.self, from: data)
            else {
                NSLog("KashikoIME: engine init failed")
                return
            }
            self.serverProcess.resetBackoff()
            NSLog(
                "KashikoIME: engine initialized (protocol v\(result.protocolVersion), model=\(result.modelName))"
            )
        }
    }

    func processKeySync(_ key: EngineKeyEvent, isRelease: Bool = false) -> KeyResult? {
        let params: [String: Any] = [
            "keysym": key.keysym,
            "modifiers": key.modifiers.jsonObject,
            "is_release": isRelease,
        ]
        return keyResultSync(method: "process_key", params: params, timeout: 3.0)
    }

    func commitSync() -> KeyResult? {
        keyResultSync(method: "commit", params: [:], timeout: 1.0)
    }

    /// Forget the learned candidate at `pageIndex` (the ✕ button on a row).
    func forgetCandidateSync(pageIndex: Int) -> KeyResult? {
        keyResultSync(
            method: "forget_candidate", params: ["page_index": pageIndex], timeout: 1.0)
    }

    /// Run another, wider inference pass over the conversion on screen (the
    /// aux-line button). Findings arrive later as an `update_candidates`
    /// notification, so the response only carries the progress message.
    func deepenInferenceSync() -> KeyResult? {
        keyResultSync(method: "deepen_inference", params: [:], timeout: 1.0)
    }

    func saveLearningAsync() {
        sendRequest(method: "save_learning", params: [:]) { _ in }
    }

    func setSurroundingTextAsync(text: String, cursorPos: Int) {
        sendRequest(
            method: "set_surrounding_text",
            params: ["text": text, "cursor_pos": cursorPos]
        ) { _ in }
    }

    /// Last secure-input state sent to the server. Main queue only.
    private var secureInputActive = false

    /// Tell the server to stop (or resume) learning and context capture.
    /// Sent only on change; queued ahead of the key that triggered it.
    func setSecureInput(_ active: Bool) {
        guard active != secureInputActive else { return }
        secureInputActive = active
        sendRequest(method: "set_secure_input", params: ["active": active]) { _ in }
    }

    private func keyResultSync(method: String, params: [String: Any], timeout: TimeInterval)
        -> KeyResult?
    {
        guard let data = sendRequestSync(method: method, params: params, timeout: timeout) else {
            return nil
        }
        do {
            return try makeProtocolDecoder().decode(KeyResult.self, from: data)
        } catch {
            logger.error("failed to decode \(method, privacy: .public) result: \(String(describing: error), privacy: .private)")
            return nil
        }
    }

    // MARK: - JSON-RPC transport

    func startReaderLoop() {
        guard let stdout = serverProcess.stdoutPipe else { return }

        let queue = DispatchQueue(label: "io.github.ryuto42.kashiko.jsonrpc.reader")
        queue.async { [weak self] in
            let handle = stdout.fileHandleForReading
            var buffer = Data()

            while true {
                let chunk = handle.availableData
                if chunk.isEmpty {
                    // EOF: server terminated
                    self?.failAllPending()
                    break
                }
                buffer.append(chunk)

                while let newlineRange = buffer.range(of: Data([0x0A])) {
                    let lineData = buffer.subdata(in: buffer.startIndex..<newlineRange.lowerBound)
                    buffer.removeSubrange(buffer.startIndex...newlineRange.lowerBound)
                    guard !lineData.isEmpty else { continue }
                    self?.handleResponse(lineData)
                }
            }
        }
    }

    @discardableResult
    func sendRequest(
        method: String, params: [String: Any], completion: @escaping (Data?) -> Void
    ) -> Int {
        lock.lock()
        let id = nextID
        nextID += 1
        pendingRequests[id] = completion
        lock.unlock()

        let request: [String: Any] = [
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        ]

        requestQueue.async { [weak self] in
            guard let self,
                let stdin = self.serverProcess.stdinPipe,
                var data = try? JSONSerialization.data(withJSONObject: request)
            else {
                self?.takePending(id: id)?(nil)
                return
            }
            data.append(0x0A)
            do {
                try stdin.fileHandleForWriting.write(contentsOf: data)
            } catch {
                self.logger.error("failed to write request: \(error.localizedDescription, privacy: .public)")
                self.takePending(id: id)?(nil)
            }
        }
        return id
    }

    func sendRequestSync(method: String, params: [String: Any], timeout: TimeInterval) -> Data? {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Data?
        let id = sendRequest(method: method, params: params) { data in
            result = data
            semaphore.signal()
        }
        if semaphore.wait(timeout: .now() + timeout) == .timedOut {
            self.logger.error("\(method, privacy: .public) timed out after \(timeout)s")
            takePending(id: id)?(nil)
            return nil
        }
        return result
    }

    private func handleResponse(_ lineData: Data) {
        guard
            let json = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any]
        else {
            self.logger.error("unparsable response line (\(lineData.count) bytes)")
            return
        }
        guard let id = json["id"] as? Int else {
            // Server-initiated notification (no id).
            if json["method"] as? String == "update_candidates",
                let params = json["params"],
                let data = try? JSONSerialization.data(withJSONObject: params),
                let decoded = try? makeProtocolDecoder().decode(
                    UpdateCandidatesParams.self, from: data)
            {
                onUpdateCandidates?(decoded.actions)
                return
            }
            self.logger.error("response without id (\(lineData.count) bytes)")
            return
        }
        if let error = json["error"] as? [String: Any] {
            let code = error["code"] as? Int ?? 0
            self.logger.error("engine error for request \(id): code \(code)")
            takePending(id: id)?(nil)
            return
        }
        guard let result = json["result"] else {
            self.logger.error("no result in response for \(id)")
            takePending(id: id)?(nil)
            return
        }
        guard let data = try? JSONSerialization.data(withJSONObject: result, options: .fragmentsAllowed) else {
            self.logger.error("JSONSerialization.data failed for \(id)")
            takePending(id: id)?(nil)
            return
        }
        takePending(id: id)?(data)
    }

    private func takePending(id: Int) -> ((Data?) -> Void)? {
        lock.lock()
        defer { lock.unlock() }
        return pendingRequests.removeValue(forKey: id)
    }

    private func failAllPending() {
        lock.lock()
        let pending = pendingRequests
        pendingRequests.removeAll()
        lock.unlock()
        for (_, completion) in pending {
            completion(nil)
        }
    }
}
