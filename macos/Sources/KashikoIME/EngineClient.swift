import Foundation
import os

/// Newline-delimited JSON-RPC 2.0 client for Kashiko IME Engine.
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
    private var pendingRequests: [Int: (Result<Data, EngineRPCError>) -> Void] = [:]

    /// Called (on the reader queue) when the server pushes an
    /// `update_candidates` notification — deep-search results arriving after
    /// the synchronous key response.
    var onUpdateCandidates: (([EngineAction]) -> Void)?

    /// `autoInit` re-sends `init` whenever the server (re)starts. Tests
    /// disable it to avoid loading models.
    init(serverProcess: EngineProcess, autoInit: Bool = true) {
        self.serverProcess = serverProcess
        self.serverProcess.onRestart = { [weak self] in
            // A fresh server starts with secure input and private mode off.
            self?.secureInputActive = false
            if let self, self.privateMode {
                self.sendRequest(method: "set_private_mode", params: ["active": true]) { _ in }
            }
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
                NSLog("Kashiko IME: engine init failed")
                return
            }
            self.serverProcess.resetBackoff()
            NSLog(
                "Kashiko IME: engine initialized (protocol v\(result.protocolVersion), model=\(result.modelName))"
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

    /// Off at every launch, like a browser's private window.
    private(set) var privateMode = false

    func setPrivateMode(_ active: Bool) {
        privateMode = active
        sendRequest(method: "set_private_mode", params: ["active": active]) { _ in }
    }

    func clearLearningAsync() {
        sendRequest(method: "clear_learning", params: [:]) { _ in }
    }

    // MARK: - Settings

    func getSettings(completion: @escaping (Result<GetSettingsResult, EngineRPCError>) -> Void) {
        sendRequestResult(method: "get_settings", params: [:]) { result in
            completion(
                result.flatMap { data in
                    do {
                        return .success(
                            try makeProtocolDecoder().decode(GetSettingsResult.self, from: data))
                    } catch {
                        return .failure(EngineRPCError(message: "設定を読み取れませんでした"))
                    }
                })
        }
    }

    func updateSettings(
        _ settings: EngineSettings,
        completion: @escaping (Result<Void, EngineRPCError>) -> Void
    ) {
        guard let data = try? makeProtocolEncoder().encode(settings),
            let object = try? JSONSerialization.jsonObject(with: data)
        else {
            completion(.failure(EngineRPCError(message: "設定を送信できませんでした")))
            return
        }
        sendRequestResult(method: "update_settings", params: ["settings": object]) { result in
            completion(result.map { _ in })
        }
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
        sendRequestResult(method: method, params: params) { completion(try? $0.get()) }
    }

    @discardableResult
    func sendRequestResult(
        method: String, params: [String: Any],
        completion: @escaping (Result<Data, EngineRPCError>) -> Void
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
                self?.takePending(id: id)?(.failure(.unavailable))
                return
            }
            data.append(0x0A)
            do {
                try stdin.fileHandleForWriting.write(contentsOf: data)
            } catch {
                self.logger.error("failed to write request: \(error.localizedDescription, privacy: .public)")
                self.takePending(id: id)?(.failure(.unavailable))
            }
        }
        return id
    }

    func sendRequestSync(method: String, params: [String: Any], timeout: TimeInterval) -> Data? {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Data?
        let id = sendRequestResult(method: method, params: params) { response in
            result = try? response.get()
            semaphore.signal()
        }
        if semaphore.wait(timeout: .now() + timeout) == .timedOut {
            self.logger.error("\(method, privacy: .public) timed out after \(timeout)s")
            takePending(id: id)?(.failure(.unavailable))
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
            let message = error["message"] as? String ?? "エンジンがエラーを返しました（\(code)）"
            takePending(id: id)?(.failure(EngineRPCError(message: message)))
            return
        }
        guard let result = json["result"] else {
            self.logger.error("no result in response for \(id)")
            takePending(id: id)?(.failure(.unavailable))
            return
        }
        guard let data = try? JSONSerialization.data(withJSONObject: result, options: .fragmentsAllowed) else {
            self.logger.error("JSONSerialization.data failed for \(id)")
            takePending(id: id)?(.failure(.unavailable))
            return
        }
        takePending(id: id)?(.success(data))
    }

    private func takePending(id: Int) -> ((Result<Data, EngineRPCError>) -> Void)? {
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
            completion(.failure(.unavailable))
        }
    }
}
