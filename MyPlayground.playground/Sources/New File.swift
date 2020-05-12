import Foundation

public struct IrisData {
    public let label: [Float]
    public let id: String
    public let imageData: [Float]
    init(id: String, imageData: [Float]) {
        self.id = id
        switch id {
        case "Setosa":
            label = [1,0,0]
        case "Versicolor":
            label = [0,1,0]
        case "Virginica":
             label = [0,0,1]
        default:
             label = [0,0,0]
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
    public let loss: Float
    public let accuracy: Float
    public init(loss: Float, accuracy: Float) {
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
        var data = [Float]()
        for i in 0 ..< (arr.count - 1) {
            data.append(Float(arr[i]) ?? 0.0)
        }
        return IrisData(id: label, imageData: data)
    }
}

public func getMNIST() -> [[Float]] {
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

    return result.map { (arr) -> [Float] in
        arr.map { Float($0) ?? 0.0 }
    }
}

public struct MNISTdata {
    public let label: Int
    public let imageData: [Float]
    init(label: Int, imageData: [Float]) {
        self.label = label
        self.imageData = imageData.map { i in
            if i != 0 {
                return i / 255
            }
            return i
        }
    }
}

public func getPreprocessedMNIST(data: [[Float]]) -> [MNISTdata] {
    return data.map { MNISTdata(label: Int($0[0]), imageData: Array($0[1...])) }
}
