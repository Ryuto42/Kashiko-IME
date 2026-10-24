import XCTest

@testable import KashikoIME

final class UpdaterTests: XCTestCase {
    func testVersionComparison() {
        XCTAssertTrue(UpdateLogic.isNewer("v0.2.0", than: "0.1.0"))
        XCTAssertTrue(UpdateLogic.isNewer("0.10.0", than: "0.9.1"))
        XCTAssertTrue(UpdateLogic.isNewer("1.0.1", than: "1.0"))
        XCTAssertFalse(UpdateLogic.isNewer("1.0.0", than: "1.0"))
        XCTAssertFalse(UpdateLogic.isNewer("v0.1.0", than: "0.1.0"))
        XCTAssertFalse(UpdateLogic.isNewer("0.9.1", than: "0.10.0"))
        XCTAssertFalse(UpdateLogic.isNewer("latest", than: "0.1.0"))
        XCTAssertFalse(UpdateLogic.isNewer("", than: "0.1.0"))
    }

    func testAssetName() {
        XCTAssertEqual(UpdateLogic.assetName(version: "v0.2.0"), "Kashiko-0.2.0-macos-arm64.zip")
        XCTAssertEqual(UpdateLogic.assetName(version: "0.2.0"), "Kashiko-0.2.0-macos-arm64.zip")
    }

    func testChecksumParsing() {
        let hashA = String(repeating: "a", count: 64)
        let hashB = String(repeating: "B", count: 64)
        let sums = """
            \(hashA)  Kashiko-0.2.0-macos-arm64.zip
            \(hashB) *other.zip
            not-a-hash  broken.zip

            """
        XCTAssertEqual(
            UpdateLogic.sha256(forFile: "Kashiko-0.2.0-macos-arm64.zip", inSums: sums), hashA)
        XCTAssertEqual(
            UpdateLogic.sha256(forFile: "other.zip", inSums: sums), hashB.lowercased())
        XCTAssertNil(UpdateLogic.sha256(forFile: "broken.zip", inSums: sums))
        XCTAssertNil(UpdateLogic.sha256(forFile: "missing.zip", inSums: sums))
    }

    func testReleaseAPIURL() {
        XCTAssertEqual(
            UpdateLogic.releaseAPIURL(repository: "Ryuto42/Kashiko-IME")?.absoluteString,
            "https://api.github.com/repos/Ryuto42/Kashiko-IME/releases/latest")
        XCTAssertNil(UpdateLogic.releaseAPIURL(repository: "a/b/c"))
        XCTAssertNil(UpdateLogic.releaseAPIURL(repository: "../x"))
        XCTAssertNil(UpdateLogic.releaseAPIURL(repository: "a b/c"))
        XCTAssertNil(UpdateLogic.releaseAPIURL(repository: ""))
    }

    func testReleaseDecoding() throws {
        let json = """
            {"tag_name":"v0.2.0","html_url":"https://github.com/o/r/releases/tag/v0.2.0",
             "assets":[{"name":"SHA256SUMS","browser_download_url":"https://example.com/s"}]}
            """
        let release = try JSONDecoder().decode(UpdateLogic.Release.self, from: Data(json.utf8))
        XCTAssertEqual(release.version, "0.2.0")
        XCTAssertEqual(release.assets.first?.name, "SHA256SUMS")
    }
}
