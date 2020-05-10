import Foundation

public struct IrisData {
    public let label: Int
    public let id: String
    public let imageData: [Double]
    init(id: String, imageData: [Double]) {
        self.id = id
        switch id {
        case "Setosa":
            label = 1
        case "Versicolor":
            label = 2
        case "Virginica":
            label = 3
        default:
            label = -1
        }
        self.imageData = imageData.map { i in
            if i != 0 {
                return i / 10.0
            }
            return i
        }
    }
}

public struct MLPerformance {
    public let loss: Decimal
    public let accuracy: Decimal
    public init(loss: Decimal, accuracy: Decimal) {
        self.loss = loss
        self.accuracy = accuracy
    }
}

public func getIrisData() -> [IrisData] {
    var text = ""

    if let fileURL = Bundle.main.url(forResource: "iris", withExtension: "csv") {
        do {
            text = try String(contentsOf: fileURL, encoding: String.Encoding.utf8)
        } catch {
            print("Error: \(error.localizedDescription)")
        }
    } else {
        print("No such file URL.")
    }

    let csvText = text as? String ?? ""

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

    return result.map { (arr) -> IrisData in

        let label = arr[arr.count - 1]
        var data = [Double]()
        for i in 0 ..< (arr.count - 1) {
            data.append(Double(arr[i]) ?? 0.0)
        }
        return IrisData(id: label, imageData: data)
    }
}

public func getMNIST() -> [[Double]] {
    var text = ""

    if let fileURL = Bundle.main.url(forResource: "mnist_test", withExtension: "csv") {
        do {
            text = try String(contentsOf: fileURL, encoding: String.Encoding.utf8)
        } catch {
            print("Error: \(error.localizedDescription)")
        }
    } else {
        print("No such file URL.")
    }

    let csvText = text as? String ?? ""

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

    return result.map { (arr) -> [Double] in
        arr.map { Double($0) ?? 0.0 }
    }
}

public struct MNISTdata {
    public let label: Int
    public let imageData: [Double]
    init(label: Int, imageData: [Double]) {
        self.label = label
        self.imageData = imageData.map { i in
            if i != 0 {
                return i / 255
            }
            return i
        }
    }
}

public func getPreprocessedMNIST(data: [[Double]]) -> [MNISTdata] {
    return data.map { MNISTdata(label: Int($0[0]), imageData: Array($0[1...])) }
}
