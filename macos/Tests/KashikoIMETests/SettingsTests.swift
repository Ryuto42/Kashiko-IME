import XCTest

@testable import KashikoIME

final class SettingsTests: XCTestCase {
    private let json = """
        {"settings":{"conversion":{"strategy":"adaptive","num_candidates":9,"use_context":true,"max_context_length":32,"composing_chunk_len":30,"dict_path":null,"model":"jinen-v2-small-q5","light_model":"jinen-v2-xsmall-q5","short_input_threshold":10,"beam_width":3,"max_latency_ms":100,"n_threads":2,"live_conversion":true},"learning":{"enabled":true,"max_entries":10000}},
        "defaults":{"conversion":{"strategy":"adaptive","num_candidates":9,"use_context":true,"max_context_length":32,"composing_chunk_len":30,"short_input_threshold":10,"beam_width":3,"max_latency_ms":100,"n_threads":2,"live_conversion":true},"learning":{"enabled":true,"max_entries":10000}},
        "models":[{"id":"jinen-v2-small-q5","display_name":"Jinen v2 Small"}]}
        """

    func testDecodeGetSettingsResult() throws {
        let result = try makeProtocolDecoder().decode(GetSettingsResult.self, from: Data(json.utf8))
        XCTAssertEqual(result.settings.conversion.numCandidates, 9)
        XCTAssertEqual(result.settings.conversion.maxLatencyMs, 100)
        XCTAssertEqual(result.settings.conversion.nThreads, 2)
        XCTAssertNil(result.settings.conversion.dictPath)
        XCTAssertEqual(result.settings.conversion.lightModel, "jinen-v2-xsmall-q5")
        XCTAssertNil(result.defaults.conversion.model)
        XCTAssertEqual(result.settings.learning.maxEntries, 10000)
        XCTAssertEqual(result.models, [ModelInfo(id: "jinen-v2-small-q5", displayName: "Jinen v2 Small")])
    }

    func testEncodeUsesSnakeCaseAndRoundTrips() throws {
        var settings = try makeProtocolDecoder().decode(
            GetSettingsResult.self, from: Data(json.utf8)
        ).settings
        settings.conversion.numCandidates = 5
        settings.learning.enabled = false

        let data = try makeProtocolEncoder().encode(settings)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let conversion = try XCTUnwrap(object["conversion"] as? [String: Any])
        XCTAssertEqual(conversion["num_candidates"] as? Int, 5)
        XCTAssertEqual(conversion["max_latency_ms"] as? Int, 100)
        XCTAssertEqual(conversion["composing_chunk_len"] as? Int, 30)
        XCTAssertEqual(conversion["light_model"] as? String, "jinen-v2-xsmall-q5")
        XCTAssertNil(conversion["dict_path"])
        let learning = try XCTUnwrap(object["learning"] as? [String: Any])
        XCTAssertEqual(learning["max_entries"] as? Int, 10000)

        let decoded = try makeProtocolDecoder().decode(EngineSettings.self, from: data)
        XCTAssertEqual(decoded, settings)
    }
}
