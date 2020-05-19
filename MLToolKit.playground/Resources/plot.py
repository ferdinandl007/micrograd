
def generate_mesh(X):
  h = 0.25
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))
  Xmesh = np.c_[xx.ravel(), yy.ravel()]

  return Xmesh

# Evaluate inputs and scores in swift:
let inputs = data.map { x -> [Value] in
    let t = x.map { Value(data: Decimal($0)) }
    return t
}

let(X, y, mapped_scores): mapped_scores = inputs.map { model.eval(x: $0).data }

def plot_results
  h = 0.25
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                      np.arange(y_min, y_max, h))
  
  Z = np.array([s > 0 for s in mapped_scores])
  Z = Z.reshape(xx.shape)

  fig = plt.figure()
  plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())
