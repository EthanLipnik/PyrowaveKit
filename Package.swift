// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "PyrowaveKit",
    platforms: [
        .macOS(.v14),
        .iOS("17.4"),
        .visionOS("1.1")
    ],
    products: [
        .library(name: "PyrowaveKit", targets: ["PyrowaveKit"]),
        .executable(name: "pyrowave-swift-bench", targets: ["pyrowave-swift-bench"])
    ],
    targets: [
        .target(
            name: "PyrowaveKit",
            resources: [.process("Metal")]
        ),
        .executableTarget(
            name: "pyrowave-swift-bench",
            dependencies: ["PyrowaveKit"]
        ),
        .testTarget(
            name: "PyrowaveKitTests",
            dependencies: ["PyrowaveKit"]
        )
    ]
)
