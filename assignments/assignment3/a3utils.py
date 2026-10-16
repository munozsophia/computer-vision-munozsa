"""
a3utils.py  --  provided code for Assignment 3. You do not need to edit this file.

Data and checking
    load_mnist()        loads MNIST from the bundled copy, returns normalised arrays
    batches()           yields shuffled minibatches
    grad_check()        compares one layer's backward() against finite differences
    grad_check_model()  compares a whole network, loss included
    evaluate()          test accuracy, computed in batches

Reference implementations from Assignment 2
    Linear, ReLU, SoftmaxCrossEntropy, MLP, train

    These are the pieces you wrote in Assignment 2, provided here so that this
    assignment stands on its own. Use them; do not rewrite them.

    One change from Assignment 2: `Linear` now applies to the LAST axis of an array
    with any number of leading axes, so the same class serves a batch of vectors,
    shape (B, n_in), and a batch of token sequences, shape (B, N, n_in). For 2-D
    input it is identical to the version you wrote.

Named in Part B
    LayerNorm, Residual, MeanPool

    Part B asks about a vision transformer without building one, and these are the
    pieces of it that the questions refer to by name. Nothing in the notebook calls
    them; they are here so that you can read what they actually do rather than take
    the questions' word for it. `Residual.backward` is the one to read -- it is a
    two-line illustration of Question A2(a) from Assignment 2.
"""
import gzip, os, struct, urllib.request
import numpy as np

# =================================================================================
#  Data
# =================================================================================

_MIRRORS = [
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]
_FILES = {"xtr": "train-images-idx3-ubyte.gz", "ytr": "train-labels-idx1-ubyte.gz",
          "xte": "t10k-images-idx3-ubyte.gz",  "yte": "t10k-labels-idx1-ubyte.gz"}


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        dims = struct.unpack(">" + "I" * ndim, f.read(4 * ndim))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def _ssl_contexts():
    """Certificate stores to try, in order. First the system's own, which works almost
    everywhere, including behind university proxies that add their own certificate
    authority. Then certifi's bundle, which rescues Python installed from python.org on
    macOS, where the system store is not used and downloads fail with
    CERTIFICATE_VERIFY_FAILED."""
    contexts = [None]
    try:
        import ssl, certifi
        contexts.append(ssl.create_default_context(cafile=certifi.where()))
    except ImportError:
        pass
    return contexts


_HELP = """
Could not download MNIST ({err}).

The assignment ships with its own copy of MNIST in the mnist_data folder, so you
should never reach this message. If you do, the folder has gone missing. Either
re-extract the assignment archive, or download the four files by hand in a web
browser from
    {url}
        train-images-idx3-ubyte.gz   train-labels-idx1-ubyte.gz
        t10k-images-idx3-ubyte.gz    t10k-labels-idx1-ubyte.gz
and put them, still compressed, in this folder:
    {folder}

If the message mentions CERTIFICATE_VERIFY_FAILED, your Python cannot check website
certificates. This is common with Python installed from python.org on a Mac. Fix it with
ONE of these, then rerun the cell:

  1. Run the certificate installer that came with Python, in a Terminal:
         /Applications/Python\\ 3.X/Install\\ Certificates.command
     (replace 3.X with your version, e.g. 3.10 -- or double-click it in Finder)
  2. Or:  pip install certifi   (this file uses it automatically if it is installed)
"""


def _fetch(name, folder):
    path = os.path.join(folder, name)
    if os.path.exists(path):
        return path
    last = None
    for base in _MIRRORS:
        print(f"  downloading {name} from {base.split('/')[2]} ...")
        for ctx in _ssl_contexts():      # system certificates first, then certifi's
            try:
                with urllib.request.urlopen(base + name, context=ctx, timeout=60) as r, \
                     open(path, "wb") as f:
                    f.write(r.read())
                return path
            except Exception as e:
                last = e
                if os.path.exists(path):
                    os.remove(path)
    raise RuntimeError(_HELP.format(err=last, url=_MIRRORS[0],
                                    folder=os.path.abspath(folder))) from None


def load_mnist(folder="mnist_data", flat=False):
    """
    Returns (x_train, y_train, x_test, y_test).

    Images are float64 in [0, 1]. Labels are integers 0..9.
      flat=False : images have shape (N, 1, 28, 28)  -- for the CNN, Part A
      flat=True  : images have shape (N, 784)        -- for the MLP baseline, A5
    """
    os.makedirs(folder, exist_ok=True)
    cache = os.path.join(folder, "mnist.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        xtr, ytr, xte, yte = d["xtr"], d["ytr"], d["xte"], d["yte"]
    else:
        raw = {k: _read_idx(_fetch(v, folder)) for k, v in _FILES.items()}
        xtr, ytr, xte, yte = raw["xtr"], raw["ytr"], raw["xte"], raw["yte"]
        np.savez_compressed(cache, xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    xtr = xtr.astype(np.float64) / 255.0
    xte = xte.astype(np.float64) / 255.0
    if flat:
        xtr, xte = xtr.reshape(len(xtr), -1), xte.reshape(len(xte), -1)
    else:
        xtr, xte = xtr[:, None], xte[:, None]
    return xtr, ytr.astype(np.int64), xte, yte.astype(np.int64)


def batches(x, y, batch_size, rng):
    """Yield (x_batch, y_batch) over one shuffled pass of the data."""
    order = rng.permutation(len(x))
    for i in range(0, len(x), batch_size):
        idx = order[i:i + batch_size]
        yield x[idx], y[idx]


# =================================================================================
#  Gradient checking
# =================================================================================

def grad_check(layer, x, eps=1e-6, rng=None, verbose=True):
    """
    Check layer.backward() against central finite differences.

    Uses the scalar loss  J = sum(r * layer.forward(x))  for a fixed random r,
    so that dJ/d(output) = r and backward(r) should return dJ/dx.

    Checks the input gradient, and the gradient of every parameter listed in
    layer.params() (if the layer has any). Returns True if all pass.

    A relative error below about 1e-6 means your backward() is correct.
    Around 1e-2 or worse means it is wrong.
    """
    rng = rng or np.random.default_rng(0)
    out = layer.forward(x)
    r = rng.standard_normal(out.shape)

    def J():
        return float(np.sum(r * layer.forward(x)))

    layer.forward(x)
    g_in = layer.backward(r)

    checks = [("input", x, g_in)]
    for name, p, g in getattr(layer, "params", lambda: [])():
        checks.append((name, p, g))

    ok = True
    for name, arr, analytic in checks:
        numeric = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"])
        # checking every entry of a big array is slow; sample at most 200
        idxs = [it.multi_index for _ in it]
        if len(idxs) > 200:
            pick = rng.choice(len(idxs), 200, replace=False)
            idxs = [idxs[i] for i in pick]
        for i in idxs:
            old = arr[i]
            arr[i] = old + eps; jp = J()
            arr[i] = old - eps; jm = J()
            arr[i] = old
            numeric[i] = (jp - jm) / (2 * eps)
        a = np.array([analytic[i] for i in idxs])
        n = np.array([numeric[i] for i in idxs])
        rel = np.max(np.abs(a - n)) / max(1e-12, np.max(np.abs(a) + np.abs(n)))
        passed = rel < 1e-5
        ok &= passed
        if verbose:
            print(f"  {name:>8s}   relative error {rel:.2e}   {'PASS' if passed else 'FAIL'}")
    layer.forward(x)          # leave the layer's cache consistent
    return ok


def evaluate(net, x, y, batch_size=500):
    """Test accuracy, computed in batches so a big model does not run out of memory."""
    correct = 0
    for i in range(0, len(x), batch_size):
        correct += int((net.forward(x[i:i + batch_size]).argmax(axis=1) == y[i:i + batch_size]).sum())
    return correct / len(x)


def grad_check_model(net, x, y, eps=1e-6, n_samples=30, rng=None, verbose=True):
    """
    Check a whole network, loss included, against finite differences.
    Samples a few entries of every parameter array. Returns True if all pass.
    """
    rng = rng or np.random.default_rng(0)

    def J():
        return float(net.loss.forward(net.forward(x), y))

    J(); net.backward(net.loss.backward())
    ok = True
    for li, (name, p, g) in enumerate(net.params()):
        g = g.copy()
        flat = rng.choice(p.size, min(n_samples, p.size), replace=False)
        a, n = [], []
        for f in flat:
            i = np.unravel_index(f, p.shape)
            old = p[i]
            p[i] = old + eps; jp = J()
            p[i] = old - eps; jm = J()
            p[i] = old
            a.append(g[i]); n.append((jp - jm) / (2 * eps))
        a, n = np.array(a), np.array(n)
        rel = np.max(np.abs(a - n)) / max(1e-12, np.max(np.abs(a) + np.abs(n)))
        passed = rel < 1e-5
        ok &= passed
        if verbose:
            print(f"  param {li:2d} ({name}, shape {str(p.shape):>14s})   relative error {rel:.2e}   "
                  f"{'PASS' if passed else 'FAIL'}")
    return ok


# =================================================================================
#  Reference implementations from Assignment 2
# =================================================================================

class Linear:
    """
    Applies  x_out = x_in @ W.T + b  to the last axis of x.

      x_in : (..., n_in)    W : (n_out, n_in)    b : (n_out,)    x_out : (..., n_out)

    With x of shape (B, n_in) this is exactly the layer you wrote in Assignment 2.
    With x of shape (B, N, n_in) the same weights are applied to each of the N tokens
    independently -- which is what a transformer needs, and is the "processed
    identically" property of Chapter 24 applied to tokens instead of pixels.
    """
    def __init__(self, n_in, n_out, rng):
        self.W = rng.standard_normal((n_out, n_in)) * np.sqrt(2.0 / n_in)   # He initialisation
        self.b = np.zeros(n_out)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x                        # save the operating point for backward
        return x @ self.W.T + self.b

    def backward(self, g_out):
        xf = self.x.reshape(-1, self.x.shape[-1])     # fold every leading axis into one
        gf = g_out.reshape(-1, g_out.shape[-1])
        self.dW = gf.T @ xf               # (n_out, M) @ (M, n_in) -> (n_out, n_in)
        self.db = gf.sum(axis=0)          # the bias is added to every row, so sum over them all
        return g_out @ self.W             # (..., n_out) @ (n_out, n_in) -> (..., n_in)

    def params(self):
        return [("W", self.W, self.dW), ("b", self.b, self.db)]


class ReLU:
    def forward(self, x):
        self.mask = x > 0                 # the gate: which units were active
        return x * self.mask

    def backward(self, g_out):
        return g_out * self.mask          # gradient passes where the unit was active

    def params(self):
        return []


class SoftmaxCrossEntropy:
    def forward(self, z, y):
        """z: (B, K) logits.  y: (B,) integer labels.  Returns the mean loss (a float)."""
        zs = z - z.max(axis=1, keepdims=True)                       # subtract the max
        logp = zs - np.log(np.exp(zs).sum(axis=1, keepdims=True))   # log-sum-exp, never log(0)
        self.p = np.exp(logp)                                       # probabilities, for backward
        self.y = y
        return -logp[np.arange(len(y)), y].mean()

    def backward(self):
        """Returns dJ/dz, shape (B, K)."""
        B = len(self.y)
        g = self.p.copy()
        g[np.arange(B), self.y] -= 1.0    # yhat - y, with y one-hot
        return g / B                      # the 1/B from averaging over the batch


class MLP:
    """A plain stack of layers with a loss on the end.

    The CNN of Part A inherits from this class, so it gets `backward`, `params` and
    `step` unchanged and only replaces `self.layers`.
    """
    def __init__(self, sizes, rng):
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1], rng))
            if i < len(sizes) - 2:
                self.layers.append(ReLU())
        self.loss = SoftmaxCrossEntropy()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, g):
        """g is dJ/d(output of the last layer). Propagate it back through every layer."""
        for layer in reversed(self.layers):
            g = layer.backward(g)
        return g

    def params(self):
        out = []
        for layer in self.layers:
            out += layer.params()
        return out

    def step(self, lr):
        """One step of gradient descent on every parameter."""
        for name, p, g in self.params():
            p -= lr * g


def train(net, x_train, y_train, x_test, y_test, epochs, lr, batch_size, rng,
          momentum=0.0):
    """
    Returns a list of per-batch training losses.

    With momentum = 0 this is exactly the loop you wrote in Assignment 2, calling
    net.step(lr).

    With momentum > 0 the update instead keeps a running average of recent gradients,

        v  <-  momentum * v + g,        p  <-  p - lr * v,

    so that consistent directions build up speed and directions that keep changing
    sign cancel out. Nothing in this assignment needs it -- the CNN trains fine
    without -- but it is here because a transformer trained with plain gradient
    descent learns very slowly, and you may want it later. The velocity lives here
    rather than in `step`, so the layer code from Assignment 2 is untouched.
    """
    history = []
    vel = [np.zeros_like(p) for _, p, _ in net.params()] if momentum else None
    for epoch in range(epochs):
        for xb, yb in batches(x_train, y_train, batch_size, rng):
            loss = net.loss.forward(net.forward(xb), yb)
            net.backward(net.loss.backward())
            if momentum:
                for v, (_, p, g) in zip(vel, net.params()):
                    v *= momentum
                    v += g
                    p -= lr * v
            else:
                net.step(lr)
            history.append(float(loss))
        print(f"epoch {epoch + 1}:  last batch loss {history[-1]:.4f},  "
              f"test accuracy {evaluate(net, x_test, y_test):.4f}")
    return history


# =================================================================================
#  Extra plumbing for the transformer (Part B)
# =================================================================================

class LayerNorm:
    """
    Normalises each token separately to zero mean and unit variance over its d features,
    then rescales by learned gamma and shifts by learned beta.

        y = gamma * (x - mu) / sqrt(var + eps) + beta

    x : (..., d).  The statistics are taken over the last axis only, so every token is
    normalised on its own and no information crosses between tokens or examples.
    """
    def __init__(self, d, eps=1e-5):
        self.gamma = np.ones(d)
        self.beta = np.zeros(d)
        self.dgamma = np.zeros(d)
        self.dbeta = np.zeros(d)
        self.eps = eps

    def forward(self, x):
        self.mu = x.mean(axis=-1, keepdims=True)
        self.var = x.var(axis=-1, keepdims=True)
        self.istd = 1.0 / np.sqrt(self.var + self.eps)
        self.xhat = (x - self.mu) * self.istd
        return self.gamma * self.xhat + self.beta

    def backward(self, g_out):
        d = self.xhat.shape[-1]
        flat = g_out.reshape(-1, d)
        xf = self.xhat.reshape(-1, d)
        self.dgamma = (flat * xf).sum(axis=0)
        self.dbeta = flat.sum(axis=0)
        gx = g_out * self.gamma
        # the mean and the variance both depend on every entry, so two correction terms
        return self.istd * (gx - gx.mean(axis=-1, keepdims=True)
                            - self.xhat * (gx * self.xhat).mean(axis=-1, keepdims=True))

    def params(self):
        return [("gamma", self.gamma, self.dgamma), ("beta", self.beta, self.dbeta)]


class Residual:
    """
    Wraps a list of layers f and computes  y = x + f(x).

    The backward pass is the clearest possible example of Question A2(a) from
    Assignment 2: x reaches the output along two paths, so the two gradients add.
    """
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        h = x
        for layer in self.layers:
            h = layer.forward(h)
        return x + h

    def backward(self, g_out):
        h = g_out
        for layer in reversed(self.layers):
            h = layer.backward(h)
        return g_out + h                  # the skip path contributes g_out directly

    def params(self):
        out = []
        for layer in self.layers:
            out += layer.params()
        return out


class MeanPool:
    """Averages over the token axis: (B, N, d) -> (B, d). Used as the classifier head."""
    def forward(self, x):
        self.shape = x.shape
        return x.mean(axis=1)

    def backward(self, g_out):
        B, N, d = self.shape
        return np.repeat(g_out[:, None, :], N, axis=1) / N

    def params(self):
        return []
