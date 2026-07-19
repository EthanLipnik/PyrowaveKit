import Foundation
import PyrowaveKit

do {
    if let reportURL = try await PyrowaveBenchmarkCLI.run() {
        print("Wrote \(reportURL.path)")
    }
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}
