import Foundation

infix operator **: ExponentiationPrecedence
precedencegroup ExponentiationPrecedence {
    associativity: right
    higherThan: MultiplicationPrecedence
}

public class Value: Hashable, CustomStringConvertible {
    public var data: Decimal
    public var grad: Decimal

    // var _backward: () -> Void
    public var _op: String
    public var _children_lhs: Value?
    public var _children_rhs: Value?

    // Store the computational graph
    public var topo = [Value]()

    public init(data: Decimal, _children_lhs: Value? = nil, _children_rhs: Value? = nil, _op: String = "") {
        self.data = data
        grad = 0

        // internal variables used for autograd graph construction
        self._children_lhs = _children_lhs
        self._children_rhs = _children_rhs
        self._op = _op // the op that produced this node, for graphviz / debugging / etc
    }

    public func hash(into hasher: inout Hasher) {
        return hasher.combine(ObjectIdentifier(self))
    }

    public var description: String { return "Value(data: \(data), grad: \(grad), op: \(_op))" }

    public static func == (lhs: Value, rhs: Value) -> Bool {
        return ObjectIdentifier(lhs) == ObjectIdentifier(rhs)
    }

    public func _backward(lhs _: Value?, rhs _: Value?) {}

    public func forward() {
        var visited = Set<Value>()

        func build_topo(v: Value) {
            if !visited.contains(v) {
                visited.insert(v)

                if let lhs_safe = v._children_lhs { build_topo(v: lhs_safe) }
                if let rhs_safe = v._children_rhs { build_topo(v: rhs_safe) }
                // v._children.forEach { build_topo(v: $0) }
                topo.append(v)
            }
        }

        build_topo(v: self)
    }

    public func backward(force_build_graph: Bool = false) {
        if (topo.count == 0) || force_build_graph {
            forward()
        }

        grad = 1
        for v in topo.reversed() {
            v._backward(lhs: v._children_lhs, rhs: v._children_rhs)
        }
    }

    public func relu() -> Value {
        var val: Decimal = 0.0
        if data >= 0 {
            val = data
        }

        let out = ReluValue(data: val, _children_lhs: self, _op: "ReLU")

        return out
    }
}

public class ReluValue: Value {
    public override func _backward(lhs: Value?, rhs _: Value?) {
        if let lhs_safe = lhs {
            if data > 0 {
                lhs_safe.grad += grad
            }
        }
    }
}

public class AddValue: Value {
    public override func _backward(lhs: Value?, rhs: Value?) {
        if let lhs_safe = lhs, let rhs_safe = rhs {
            lhs_safe.grad += grad
            rhs_safe.grad += grad
        }
    }
}

public func + (lhs: Value, rhs: Value) -> Value {
    let out = AddValue(data: lhs.data + rhs.data, _children_lhs: lhs, _children_rhs: rhs, _op: "+")
    return out
}

public func + (lhs: Value, rhs: Decimal) -> Value {
    let val2 = Value(data: rhs)
    return lhs + val2
}

public func + (lhs: Decimal, rhs: Value) -> Value {
    let val2 = Value(data: lhs)
    return rhs + val2
}

public class MultiplyValue: Value {
    public override func _backward(lhs: Value?, rhs: Value?) {
        if let lhs_safe = lhs, let rhs_safe = rhs {
            lhs_safe.grad += rhs_safe.data * grad
            rhs_safe.grad += lhs_safe.data * grad
        }
    }
}

public func * (lhs: Value, rhs: Value) -> Value {
    return MultiplyValue(data: lhs.data * rhs.data, _children_lhs: lhs, _children_rhs: rhs, _op: "*")
}

public func * (lhs: Value, rhs: Decimal) -> Value {
    let val2 = Value(data: rhs)
    return lhs * val2
}

public func * (lhs: Decimal, rhs: Value) -> Value {
    let val2 = Value(data: lhs)
    return rhs * val2
}

public class ExponentialValue: Value {
    public var exponent: Int

    public init(data: Decimal, _children_lhs: Value? = nil, _children_rhs: Value? = nil, _op: String = "", exponent: Int) {
        self.exponent = exponent
        super.init(data: data, _children_lhs: _children_lhs, _children_rhs: _children_rhs, _op: _op)
    }

    public override func _backward(lhs: Value?, rhs _: Value?) {
        if let lhs_safe = lhs {
            if exponent >= 1 {
                lhs_safe.grad += (Decimal(exponent) * pow(lhs_safe.data, exponent - 1)) * grad
            } else if exponent <= -1 {
                lhs_safe.grad += (Decimal(exponent) / pow(lhs_safe.data, -exponent + 1)) * grad
            } else {
                print("Not currently supported!")
            }
        }
    }
}

public func ** (lhs: Value, rhs: Int) -> Value {
    if rhs >= 1 {
        let out = ExponentialValue(data: pow(lhs.data, rhs), _children_lhs: lhs, _op: "**", exponent: rhs)
        return out
    } else if rhs <= -1 {
        let out = ExponentialValue(data: 1.0 / pow(lhs.data, -rhs), _children_lhs: lhs, _op: "**", exponent: rhs)
        return out
    } else {
        print("Not supported!")
        let out = ExponentialValue(data: 0.0, _children_lhs: lhs, _op: "DEAD", exponent: rhs)
        return out
    }
}

// Complete the operator set

public prefix func - (lhs: Value) -> Value {
    return lhs * (-1)
}

public func - (lhs: Value, rhs: Value) -> Value {
    return lhs + (-rhs)
}

public func - (lhs: Value, rhs: Decimal) -> Value {
    let val2 = Value(data: rhs)
    return lhs - val2
}

public func - (lhs: Decimal, rhs: Value) -> Value {
    let val2 = Value(data: lhs)
    return val2 - rhs
}

public func / (lhs: Value, rhs: Value) -> Value {
    return lhs * (rhs ** -1)
}

public func / (lhs: Value, rhs: Decimal) -> Value {
    let val2 = Value(data: rhs)
    return lhs / val2
}

public func / (lhs: Decimal, rhs: Value) -> Value {
    let val2 = Value(data: lhs)
    return val2 / rhs
}

// Define a simple non-generic tree
public final class Node {
    public let value: Value

    private(set) weak var parent: Node?
    private(set) var children: [Node] = []

    public init(value: Value) {
        self.value = value

        if let lhs_safe = self.value._children_lhs { addChild(node: Node(value: lhs_safe)) }
        if let rhs_safe = self.value._children_rhs { addChild(node: Node(value: rhs_safe)) }
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
            let randn = Double.random(in: -1 ... 1)
            w.append(Value(data: Decimal(randn)))
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

    public var description: String {
        if nonlin { return "ReLUNeuron(\(w.count))" }
        else { return "LinearNeuron(\(w.count))" }
    }
}

public class Layer: Module, CustomStringConvertible {
    public var neurons: [Neuron]

    public init(nin: Int, nout: Int, nonlin: Bool = true) {
        // neurons = Array(repeating: Neuron(nin: nin, nonlin: nonlin), count: nout)
        neurons = []
        for _ in 1 ... nout {
            neurons.append(Neuron(nin: nin, nonlin: nonlin))
        }
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
