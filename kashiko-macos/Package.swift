// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "KashikoIME",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "KashikoIME",
            path: "Sources/KashikoIME",
            linkerSettings: [
                .linkedFramework("InputMethodKit")
            ]
        ),
        .testTarget(
            name: "KashikoIMETests",
            dependencies: ["KashikoIME"],
            path: "Tests/KashikoIMETests"
        ),
    ]
)
