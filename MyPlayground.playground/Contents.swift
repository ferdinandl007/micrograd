import Foundation

var model = MLP(nin: 2, nouts: [4 ,4, 3]) // 2-layer neural network
print(model)
print("number of parameters \(model.parameters().count)")

// this processes a lot of understated can take up to 30 seconds
// iris
var result = getIrisData()
result.remove(at: 0)
// MNIST
// let result = getMNIST()
// let proses = getPreprocessedMNIST(data: result)
//
let t = Float()
let data: [[Float]] = result.map { $0.imageData }
let labels: [[Float]] = result.map { $0.label }



func loss() -> (Value,Float) {
    let inputs = data.map { x -> [Value] in
        let t = x.map { Value(data: $0) }
        return t
    }

    let scores = inputs.map { model.eval(x: $0) }

    // svm "max-margin" loss
    let losses = zip(labels, scores).map { (yi, scoresi) -> [Value] in
        zip(yi, scoresi).map { (1 + (-$0 * $1)).relu()}
    }.flatMap { $0 }

    let dataLoss = losses.reduce(Value(data: 0), +) * (1.0 / Float(losses.count))

    // L2 regularization
    let alpha: Float = 0.0001
    let p = model.parameters().map { $0 * $0 }
    let regLoss = alpha * p.reduce(Value(data: 0), +)
    let total_loss = dataLoss + regLoss
    // Also get accuracy
//    let accuracy = zip(labels, scores).map({ (yi, scoresi) -> Float in
//            let accuracy = zip(labels, scores).map({ (yi, scoresi) -> [Float] in
//            return zip(yi,scoresi).map {
//                if ($0 > 0) == ($1.data > 0) {
//                    return 1
//                }
//                return 0
//            }
//        })
//    }).reduce(0.0, +)


    return  (total_loss, 100 / Float(labels.count))
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
    let learning_rate = 1.0 - 0.9*Float(k)/100
    for p in model.parameters() {
        p.data -= learning_rate * p.grad
    }
    if k % 1 == 0 {
        print("step \(k) loss \(total_loss.data), accuracy \(acc)%")
    }
}
print(result[15].label)
print(model.eval(x: (result[15].imageData.map({Value(data: $0)})) ))
