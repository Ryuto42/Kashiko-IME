import Foundation

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
    private let requestQueue = DispatchQueue(label: "dev.togatoga.kashiko.jsonrpc.request")

    private func debugLog(_ msg: String) {
        let text = "\(Date()): \(msg)\n"
        if let handle = FileHandle(forWritingAtPath: "/tmp/kashiko_debug.log") {
            handle.seekToEndOfFile()
            handle.write(text.data(using: .utf8)!)
            handle.closeFile()
        } else {
            try? text.write(toFile: "/tmp/kashiko_debug.log", atomically: false, encoding: .utf8)
        }
    }

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

    func saveLearningAsync() {
        sendRequest(method: "save_learning", params: [:]) { _ in }
    }

    func setSurroundingTextAsync(text: String, cursorPos: Int) {
        sendRequest(
            method: "set_surrounding_text",
            params: ["text": text, "cursor_pos": cursorPos]
        ) { _ in }
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
            debugLog("failed to decode \(method) result: \(error)")
            NSLog("KashikoIME: failed to decode \(method) result: \(error)")
            return nil
        }
    }

    // MARK: - JSON-RPC transport

    func startReaderLoop() {
        guard let stdout = serverProcess.stdoutPipe else { return }

        let queue = DispatchQueue(label: "dev.togatoga.kashiko.jsonrpc.reader")
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
                self.debugLog("failed to write request: \(error)")
                NSLog("KashikoIME: failed to write request: \(error)")
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
            self.debugLog("\(method) timed out after \(timeout)s")
            NSLog("KashikoIME: \(method) timed out after \(timeout)s")
            takePending(id: id)?(nil)
            return nil
        }
        return result
    }

    private func handleResponse(_ lineData: Data) {
        guard
            let json = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any]
        else {
            self.debugLog("unparsable response line: \(String(data: lineData, encoding: .utf8) ?? "")")
            NSLog("KashikoIME: unparsable response line")
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
            self.debugLog("response without id: \(json)")
            NSLog("KashikoIME: response without id: \(json)")
            return
        }
        if let error = json["error"] as? [String: Any] {
            self.debugLog("engine error for request \(id): \(error)")
            NSLog("KashikoIME: engine error for request \(id): \(error)")
            takePending(id: id)?(nil)
            return
        }
        guard let result = json["result"] else {
            self.debugLog("no result in response for \(id): \(json)")
            takePending(id: id)?(nil)
            return
        }
        guard let data = try? JSONSerialization.data(withJSONObject: result, options: .fragmentsAllowed) else {
            self.debugLog("JSONSerialization.data failed for \(id): \(result)")
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
