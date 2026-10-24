import Foundation

// Swift-side mirror of crates/kashiko-im/src/server/protocol.rs.
// All positions (caret, attribute start/end) are in Unicode scalar values;
// convert to UTF-16 offsets before passing to IMK APIs.

struct KeyResult: Decodable {
    let consumed: Bool
    let actions: [EngineAction]
    let conversionMs: UInt64?
    let processKeyMs: UInt64?
}

struct PreeditAttr: Decodable {
    let start: Int
    let end: Int
    let style: String
}

struct CandidateItem: Decodable {
    let text: String
    let description: String?
    /// Came from the learning cache, so the user may delete it (hover ✕).
    /// Absent in the payload for non-learned candidates.
    private let isLearnedFlag: Bool?

    var isLearned: Bool { isLearnedFlag ?? false }

    private enum CodingKeys: String, CodingKey {
        case text
        case description
        case isLearnedFlag = "is_learned"
    }
}

struct InitResult: Decodable {
    let protocolVersion: Int
    let modelName: String
}

/// Params of the server-initiated `update_candidates` notification: pushed
/// after a key response when the deep typo search finds extra candidates.
struct UpdateCandidatesParams: Decodable {
    let actions: [EngineAction]
}

enum EngineAction: Decodable {
    case updatePreedit(text: String, caret: Int, attributes: [PreeditAttr])
    case showCandidates(candidates: [CandidateItem], cursor: Int, page: Int, totalPages: Int)
    case hideCandidates
    case commit(text: String)
    /// Undo commit (reconversion): replace this just-committed text, which
    /// sits immediately before the caret, with the marked text from the
    /// following updatePreedit in the same batch.
    case deleteSurrounding(text: String)
    case updateAux(text: String, button: Bool)
    case hideAux

    private enum CodingKeys: String, CodingKey {
        case type, text, caret, attributes, candidates, cursor, page, totalPages, button
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type {
        case "update_preedit":
            self = .updatePreedit(
                text: try container.decode(String.self, forKey: .text),
                caret: try container.decode(Int.self, forKey: .caret),
                attributes: try container.decodeIfPresent([PreeditAttr].self, forKey: .attributes)
                    ?? []
            )
        case "show_candidates":
            self = .showCandidates(
                candidates: try container.decode([CandidateItem].self, forKey: .candidates),
                cursor: try container.decode(Int.self, forKey: .cursor),
                page: try container.decode(Int.self, forKey: .page),
                totalPages: try container.decode(Int.self, forKey: .totalPages)
            )
        case "hide_candidates":
            self = .hideCandidates
        case "commit":
            self = .commit(text: try container.decode(String.self, forKey: .text))
        case "delete_surrounding":
            self = .deleteSurrounding(text: try container.decode(String.self, forKey: .text))
        case "update_aux":
            self = .updateAux(
                text: try container.decode(String.self, forKey: .text),
                button: try container.decodeIfPresent(Bool.self, forKey: .button) ?? false)
        case "hide_aux":
            self = .hideAux
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type, in: container, debugDescription: "unknown action type: \(type)")
        }
    }
}

struct EngineRPCError: Error, LocalizedError {
    let message: String

    static let unavailable = EngineRPCError(message: "エンジンに接続できません")

    var errorDescription: String? { message }
}

/// Mirror of the server's `Settings`. Every field is kept (even those the
/// settings screen does not show) because `update_settings` replaces the
/// whole config.toml.
struct EngineSettings: Codable, Equatable {
    struct Conversion: Codable, Equatable {
        var strategy: String
        var numCandidates: Int
        var useContext: Bool
        var maxContextLength: Int
        var composingChunkLen: Int
        var dictPath: String?
        var model: String?
        var lightModel: String?
        var shortInputThreshold: Int
        var beamWidth: Int
        var maxLatencyMs: UInt64
        var nThreads: Int
        var liveConversion: Bool
    }

    struct Learning: Codable, Equatable {
        var enabled: Bool
        var maxEntries: Int
    }

    var conversion: Conversion
    var learning: Learning
}

struct ModelInfo: Decodable, Equatable {
    let id: String
    let displayName: String
}

/// Result of `get_settings`.
struct GetSettingsResult: Decodable {
    let settings: EngineSettings
    let defaults: EngineSettings
    let models: [ModelInfo]
}

/// JSONDecoder configured for the engine protocol (snake_case keys).
func makeProtocolDecoder() -> JSONDecoder {
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    return decoder
}

func makeProtocolEncoder() -> JSONEncoder {
    let encoder = JSONEncoder()
    encoder.keyEncodingStrategy = .convertToSnakeCase
    return encoder
}
