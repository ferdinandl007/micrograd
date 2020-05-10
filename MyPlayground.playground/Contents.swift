import Foundation
import PlaygroundSupport

class Value: Hashable, CustomStringConvertible {
    var data: Decimal
    var grad: Decimal

    var _backward: () -> Void
    var _op: String
    var _children: Set<Value>

    init(data: Decimal, _children: Set<Value> = [], _op: String = "") {
        self.data = data
        grad = 0

        // internal variables used for autograd graph construction
        _backward = { () -> Void in }
        self._children = _children
        self._op = _op // the op that produced this node, for graphviz / debugging / etc
    }

    func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self))
    }
    public var description: String { return "Value(data: \(data), grad: \(grad), op: \(_op))" }

    func backward() {
        var topo = [Value]()
        var visited = Set<Value>()

        func build_topo(v: Value) {
            if !visited.contains(v) {
                visited.insert(v)
                v._children.forEach { build_topo(v: $0) }
                topo.append(v)
            }
        }

        build_topo(v: self)

        grad = 1
        for v in topo.reversed() {
            v._backward()
        }
    }

    func relu() -> Value {
        var val: Decimal = 0.0
        if data >= 0 {
            val = data
        }

        var out = Value(data: val, _children: [self], _op: "ReLU")

        func _backward() {
            if out.data > 0 {
                grad += out.grad
            }
        }
        out._backward = _backward
        return out
    }
}

func + (lhs: Value, rhs: Value) -> Value {
    var out = Value(data: lhs.data + rhs.data, _children: [lhs, rhs], _op: "+")

    func _backward() {
        lhs.grad += out.grad
        rhs.grad += out.grad
    }
    out._backward = _backward
    return out
}

func + (lhs: Value, rhs: Decimal) -> Value {
    let val2 = Value(data: rhs)
    return lhs + val2
}

func + (lhs: Decimal, rhs: Value) -> Value {
    let val2 = Value(data: lhs)
    return rhs + val2
}

func * (lhs: Value, rhs: Value) -> Value {
    var out = Value(data: lhs.data * rhs.data, _children: [lhs, rhs], _op: "*")

    func _backward() {
        lhs.grad += rhs.data * out.grad
        rhs.grad += lhs.data * out.grad
    }
    out._backward = _backward
    return out
}

func * (lhs: Value, rhs: Decimal) -> Value {
    let val2 = Value(data: rhs)
    return lhs * val2
}

func * (lhs: Decimal, rhs: Value) -> Value {
    let val2 = Value(data: lhs)
    return rhs * val2
}

precedencegroup ExponentiationPrecedence {
    associativity: right
    higherThan: MultiplicationPrecedence
}

infix operator **: ExponentiationPrecedence
func ** (lhs: Value, rhs: Int) -> Value {
    var out = Value(data: pow(lhs.data, rhs), _children: [lhs], _op: "+")

    func _backward() {
        lhs.grad += (Decimal(rhs) * pow(lhs.data, rhs - 1)) * out.grad
    }
    out._backward = _backward

    return out
}

// Complete the operator set

prefix func - (lhs: Value) -> Value {
    return lhs * -1
}

func - (lhs: Value, rhs: Value) -> Value {
    return lhs + (-rhs)
}

func - (lhs: Value, rhs: Decimal) -> Value {
    let val2 = Value(data: rhs)
    return lhs - val2
}

func - (lhs: Decimal, rhs: Value) -> Value {
    let val2 = Value(data: lhs)
    return rhs - val2
}

func / (lhs: Value, rhs: Value) -> Value {
    return lhs * (rhs ** -1)
}

func / (lhs: Value, rhs: Decimal) -> Value {
    let val2 = Value(data: rhs)
    return lhs / val2
}

func / (lhs: Decimal, rhs: Value) -> Value {
    let val2 = Value(data: lhs)
    return rhs / val2
}

// For Hashing
func == (lhs: Value, rhs: Value) -> Bool {
    return ObjectIdentifier(lhs) == ObjectIdentifier(rhs)
}

// Ugly, please check if there's a better way
func castToDouble<T: Numeric>(val: T) -> Double {
    switch val {
    case let i as Int: return Double(i)
    case let f as Float: return Double(f)
    case let d as Double: return d
    default: fatalError()
    }
}

func castToFloat<T: Numeric>(val: T) -> Float {
    switch val {
    case let i as Int: return Float(i)
    case let f as Float: return Float(f)
    case let d as Double: return Float(d)
    default: fatalError()
    }
}

// Define a simple non-generic tree
final class Node {
    let value: Value

    private(set) weak var parent: Node?
    private(set) var children: [Node] = []

    init(value: Value) {
        self.value = value
        self.value._children.forEach { self.addChild(node: Node(value: $0)) }
    }

    func addChild(node: Node) {
        children.append(node)
        node.parent = self
    }

    func treeLines(sep: String = "|--- ") -> [String] {
        return ["Data: \(value.data) & Grad: \(value.grad) & op: \(value._op)"] + children.flatMap { $0.treeLines(sep: sep) }.map { sep + $0 }
    }

    func printTree(sep: String = "|--- ") {
        let text = treeLines(sep: sep).joined(separator: "\n")
        print(text)
    }
}

class Module {
    func zero_grad() {
        parameters().forEach { $0.grad = 0 }
    }

    func parameters() -> [Value] {
        return []
    }
}

class Neuron: Module, CustomStringConvertible {
    var w: [Value]
    var b: Value
    var nonlin: Bool

    init(nin: Int, nonlin: Bool = true) {
        w = []
        for _ in 0 ..< nin {
            let randn = Double.random(in: -1 ... 1)
            w.append(Value(data: Decimal(randn)))
        }
        b = Value(data: 0)
        self.nonlin = nonlin
    }

    func eval(x: [Value]) -> Value {
        var act = Value(data: 0)

        for (wi, xi) in zip(w, x) {
            act = act + (wi * xi)
        }

        act = act + b

        if nonlin {
            return act.relu()
        } else {
            return act
        }
    }

    override func parameters() -> [Value] {
        return w + [b]
    }

    public var description: String { return "Neuron(\(w.count))" }
}

class Layer: Module, CustomStringConvertible {
    var neurons: [Neuron]

    init(nin: Int, nout: Int, nonlin: Bool = true) {
        neurons = Array(repeating: Neuron(nin: nin, nonlin: nonlin), count: nout)
    }

    func eval(x: [Value]) -> [Value] {
        return neurons.map { $0.eval(x: x) }
    }

    override func parameters() -> [Value] {
        return neurons.flatMap { $0.parameters() }
    }

    public func descr_fn() -> String {
        var descr = ""
        neurons.forEach { descr += $0.description + ", " }
        return "Layer of: [\(descr)]"
    }

    public var description: String { return descr_fn() }
}

class MLP: Module, CustomStringConvertible {
    var layers: [Layer] = []

    init(nin: Int, nouts: [Int]) {
        let sz = [nin] + nouts // nouts.insert(nin, at: 0)

        for i in 0 ..< nouts.count {
            var nonlin = true
            if i == (nouts.count - 1) {
                nonlin = false
            }
            layers.append(Layer(nin: sz[i], nout: sz[i + 1], nonlin: nonlin))
        }
    }

    func eval(x: [Value]) -> [Value] {
        var out = x
        layers.forEach { out = $0.eval(x: out) }
        return out
    }

    override func parameters() -> [Value] {
        return layers.flatMap { $0.parameters() }
    }

    public func descr_fn() -> String {
        var descr = ""
        layers.forEach { descr += $0.description + ", " }
        return "Model of: [\(descr)]"
    }

    public var description: String { return descr_fn() }
}


var model = MLP(nin: 2, nouts: [4, 3, 3]) // 2-layer neural network
print(model)
print("number of parameters \(model.parameters().count)")

// this processes a lot of understated can take up to 30 seconds
// iris
let result = getIrisData()
// MNIST
// let result = getMNIST()
// let proses = getPreprocessedMNIST(data: result)
//
let data: [[Double]] = result.map { $0.imageData }
let labels: [Int] = result.map { $0.label }

func loss() -> (Value,Decimal) {
    let inputs = data.map { x -> [Value] in
        let t = x.map { Value(data: Decimal($0)) }
        return t
    }

    let scores = inputs.map { model.eval(x: $0) }

    // svm "max-margin" loss
    let losses = zip(labels, scores).map { (yi, scoresi) -> [Value] in
        scoresi.map { (1 + (Decimal(yi) * $0)).relu() }
    }.flatMap { $0 }

    let dataLoss = losses.reduce(Value(data: 0), +) * (1.0 / Decimal(losses.count))

    // L2 regularization
    let alpha: Decimal = 0.0001
    let p = model.parameters().map { $0 * $0 }
    let regLoss = alpha * p.reduce(Value(data: 0), +)
    let total_loss = dataLoss + regLoss
    // Also get accuracy
    let accuracy = zip(labels, scores).map({ (yi, scoresi) -> [Decimal] in
        scoresi.map { s in
            if (yi > 0) == (s.data > 0) {
                return 1
            }
            return 0
        }
    }
    ).flatMap { $0 }.reduce(0.0, +)

    return  (total_loss, accuracy / Decimal(labels.count))
}

let ml = loss()
print(ml)

var a = Value(data: -4.0)
var b = Value(data: 2.0)
print((1 + 5 * a).relu())


for k in 0...100 {
   // forward
   let (total_loss, acc) = loss()

    // backward
    model.zero_grad()
    total_loss.backward()

    // update (sgd)
    let learning_rate = 1.0 - 0.9*Decimal(k)/100
    for p in model.parameters() {
        p.data -= learning_rate * p.grad
    }
    if k % 1 == 0 {
        print("step \(k) loss \(total_loss.data), accuracy \(acc)%")
    }


}

