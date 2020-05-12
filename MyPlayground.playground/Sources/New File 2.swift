import Foundation

public class Value: Hashable, CustomStringConvertible {
    public var data: Float
    public var grad: Float

    public var _backward: () -> Void
    public var _op: String
    public var _children: Set<Value>

    public init(data: Float, _children: Set<Value> = [], _op: String = "") {
        self.data = data
        grad = 0

        // internal variables used for autograd graph construction
        _backward = { () -> Void in }
        self._children = _children
        self._op = _op // the op that produced this node, for graphviz / debugging / etc
    }

    public func hash(into hasher: inout Hasher) {
        hasher.combine(ObjectIdentifier(self))
    }
    public var description: String { return "Value(data: \(data), grad: \(grad), op: \(_op))" }

    public func backward() {
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

    public func relu() -> Value {
        var val: Float = 0.0
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
    
     public func sigmoid() -> Value {
        var val: Float =  0.0
        let sig: Float = 1.0 / (1.0 + exp(data))
        if sig >= 0 {
            val = sig
        }
        var out = Value(data: val, _children: [self], _op: "sigmoid")
        
        func _backward() {
            if out.data > 0 {
                grad += out.grad
            }
        }
        
        out._backward = _backward
        
        return out
    }
}

public func + (lhs: Value, rhs: Value) -> Value {
    var out = Value(data: lhs.data + rhs.data, _children: [lhs, rhs], _op: "+")

    func _backward() {
        lhs.grad += out.grad
        rhs.grad += out.grad
    }
    out._backward = _backward
    return out
}

public func + (lhs: Value, rhs: Float) -> Value {
    let val2 = Value(data: rhs)
    return lhs + val2
}

public func + (lhs: Float, rhs: Value) -> Value {
    let val2 = Value(data: lhs)
    return rhs + val2
}

public func * (lhs: Value, rhs: Value) -> Value {
    var out = Value(data: lhs.data * rhs.data, _children: [lhs, rhs], _op: "*")

    func _backward() {
        lhs.grad += rhs.data * out.grad
        rhs.grad += lhs.data * out.grad
    }
    out._backward = _backward
    return out
}

public func * (lhs: Value, rhs: Float) -> Value {
    let val2 = Value(data: rhs)
    return lhs * val2
}

public func * (lhs: Float, rhs: Value) -> Value {
    let val2 = Value(data: lhs)
    return rhs * val2
}

precedencegroup ExponentiationPrecedence {
    associativity: right
    higherThan: MultiplicationPrecedence
}

infix operator **: ExponentiationPrecedence
public func ** (lhs: Value, rhs: Int) -> Value {
    var out = Value(data: pow(lhs.data, Float(rhs)), _children: [lhs], _op: "+")

    func _backward() {
        lhs.grad += (Float(rhs) * pow(lhs.data, Float(rhs - 1))) * out.grad
    }
    out._backward = _backward

    return out
}

// Complete the operator set

public prefix func - (lhs: Value) -> Value {
    return lhs * -1
}

public func - (lhs: Value, rhs: Value) -> Value {
    return lhs + (-rhs)
}

public func - (lhs: Value, rhs: Float) -> Value {
    let val2 = Value(data: rhs)
    return lhs - val2
}

public func - (lhs: Float, rhs: Value) -> Value {
    let val2 = Value(data: lhs)
    return rhs - val2
}

public func / (lhs: Value, rhs: Value) -> Value {
    return lhs * (rhs ** -1)
}

public func / (lhs: Value, rhs: Float) -> Value {
    let val2 = Value(data: rhs)
    return lhs / val2
}

public func / (lhs: Float, rhs: Value) -> Value {
    let val2 = Value(data: lhs)
    return rhs / val2
}

// For Hashing
public func == (lhs: Value, rhs: Value) -> Bool {
    return ObjectIdentifier(lhs) == ObjectIdentifier(rhs)
}

// Ugly, please check if there's a better way
public func castToDouble<T: Numeric>(val: T) -> Double {
    switch val {
    case let i as Int: return Double(i)
    case let f as Float: return Double(f)
    case let d as Double: return d
    default: fatalError()
    }
}

public func castToFloat<T: Numeric>(val: T) -> Float {
    switch val {
    case let i as Int: return Float(i)
    case let f as Float: return Float(f)
    case let d as Double: return Float(d)
    default: fatalError()
    }
}

// Define a simple non-generic tree
public final class Node {
public let value: Value

    private(set) weak var parent: Node?
    private(set) var children: [Node] = []

    init(value: Value) {
        self.value = value
        self.value._children.forEach { self.addChild(node: Node(value: $0)) }
    }

public func addChild(node: Node) {
        children.append(node)
        node.parent = self
    }

public func treeLines(sep: String = "|--- ") -> [String] {
        return ["Data: \(value.data) & Grad: \(value.grad) & op: \(value._op)"] + children.flatMap { $0.treeLines(sep: sep) }.map { sep + $0 }
    }

public func printTree(sep: String = "|--- ") {
        let text = treeLines(sep: sep).joined(separator: "\n")
        print(text)
    }
}

public class Module {
    public func zero_grad() {
        parameters().forEach { $0.grad = 0 }
    }

    public func parameters() -> [Value] {
        return []
    }
}

public class Neuron: Module, CustomStringConvertible {
    public var w: [Value]
    public var b: Value
    public var nonlin: Bool

    public init(nin: Int, nonlin: Bool = true) {
        w = []
        for _ in 0 ..< nin {
            let randn = Float.random(in: -1 ... 1)
            w.append(Value(data: randn))
        }
        b = Value(data: 0)
        self.nonlin = nonlin
    }

    public func eval(x: [Value]) -> Value {
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

    public override func parameters() -> [Value] {
        return w + [b]
    }

    public var description: String { return "Neuron(\(w.count))" }
}

public class Layer: Module, CustomStringConvertible {
    public var neurons: [Neuron]

    public init(nin: Int, nout: Int, nonlin: Bool = true) {
        neurons = Array(repeating: Neuron(nin: nin, nonlin: nonlin), count: nout)
    }

    public func eval(x: [Value]) -> [Value] {
        return neurons.map { $0.eval(x: x) }
    }

    public override func parameters() -> [Value] {
        return neurons.flatMap { $0.parameters() }
    }

    public func descr_fn() -> String {
        var descr = ""
        neurons.forEach { descr += $0.description + ", " }
        return "Layer of: [\(descr)]"
    }

    public var description: String { return descr_fn() }
}

public class MLP: Module, CustomStringConvertible {
    public var layers: [Layer] = []

    public init(nin: Int, nouts: [Int]) {
        let sz = [nin] + nouts // nouts.insert(nin, at: 0)

        for i in 0 ..< nouts.count {
            var nonlin = true
            if i == (nouts.count - 1) {
                nonlin = false
            }
            layers.append(Layer(nin: sz[i], nout: sz[i + 1], nonlin: nonlin))
        }
    }

    public func eval(x: [Value]) -> [Value] {
        var out = x
        layers.forEach { out = $0.eval(x: out) }
        return out
    }

    public override func parameters() -> [Value] {
        return layers.flatMap { $0.parameters() }
    }

    public func descr_fn() -> String {
        var descr = ""
        layers.forEach { descr += $0.description + ", " }
        return "Model of: [\(descr)]"
    }

    public var description: String { return descr_fn() }
}


public func max(_ x: [Value]) -> Int {
    var v: Float = 0.0
    var index: Int = 0
    for i in 0 ..< x.count {
        if v < x[i].data {
            v = x[i].data
            index = i
        }
    }
    return index + 1
}
