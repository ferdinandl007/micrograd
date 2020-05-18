import Foundation

public struct MoonsData {
    public let label: Int
    public let imageData: [Double]
    public init(label: Int, imageData: [Double]) {
        self.label = label
        self.imageData = imageData.map { i in
            if i != 0 {
                return i
            }
            return i
        }
    }
}

public func getMoonsData() -> [MoonsData] {
    if let fileURL = Bundle.main.url(forResource: "out", withExtension: "csv") {
        do {
            let csvText = try! String(contentsOf: fileURL, encoding: String.Encoding.utf8)

            let pattern = "[ \t]*(?:\"((?:[^\"]|\"\")*)\"|([^,\"\r\\n]*))[ \t]*(,|\r\\n?|\\n|$)"
            let regex = try! NSRegularExpression(pattern: pattern)

            var result: [[String]] = []
            var record: [String] = []
            regex.enumerateMatches(in: csvText, options: .anchored, range: NSRange(0 ..< csvText.utf16.count)) { match, _, stop in
                guard let match = match else { fatalError() }
                if let quotedRange = Range(match.range(at: 1), in: csvText) {
                    let field = csvText[quotedRange].replacingOccurrences(of: "\"\"", with: "\"")
                    record.append(field)
                } else if let range = Range(match.range(at: 2), in: csvText) {
                    let field = csvText[range].trimmingCharacters(in: .whitespaces)
                    record.append(field)
                }
                let separator = csvText[Range(match.range(at: 3), in: csvText)!]
                switch separator {
                case "": // end of text
                    // Ignoring empty last line...
                    if record.count > 1 || (record.count == 1 && !record[0].isEmpty) {
                        result.append(record)
                    }
                    stop.pointee = true
                case ",": // comma
                    break
                default: // newline
                    result.append(record)
                    record = []
                }
            }

            return result.map { (arr) -> MoonsData in

                let label = arr[arr.count - 1]
                var data = [Double]()
                for i in 0 ..< (arr.count - 1) {
                    data.append(Double(arr[i]) ?? 1000)
                }
                return MoonsData(label: Int(label) ?? -1, imageData: data)
            }

        }

    } else {
        print("No such file URL.")
        return []
    }
}
