"""
a2utils.py  --  provided code for Assignment 2. You do not need to edit this file.

    load_mnist()      loads MNIST (downloading it only if it is missing), returns normalized arrays
    grad_check()      compares a layer's backward() against finite differences
    batches()         yields shuffled minibatches
"""
import gzip, os, struct, urllib.request
import numpy as np

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

If the message mentions CERTIFICATE_VERIFY_FAILED, your Python cannot check website
certificates. This is common with Python installed from python.org on a Mac. Fix it with
ONE of these, then rerun the cell:

  1. Run the certificate installer that came with Python, in a Terminal:
         /Applications/Python\\ 3.X/Install\\ Certificates.command
     (replace 3.X with your version, e.g. 3.10 -- or double-click it in Finder)
  2. Or:  pip install certifi   (this file uses it automatically if it is installed)

Otherwise, download the four files by hand in a web browser from
    {url}
        train-images-idx3-ubyte.gz   train-labels-idx1-ubyte.gz
        t10k-images-idx3-ubyte.gz    t10k-labels-idx1-ubyte.gz
and put them, still compressed, in this folder:
    {folder}
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


def load_mnist(folder="mnist_data"):
    """
    Returns (x_train, y_train, x_test, y_test).

    Images are float64 in [0, 1], each flattened to a row of 784 values:
    x_train has shape (60000, 784) and x_test (10000, 784).
    Labels are integers 0..9.

    The four MNIST files ship with the assignment in mnist_data/, so normally nothing is
    downloaded. If they are missing, they are fetched once and cached.
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
    xtr = xtr.reshape(len(xtr), -1).astype(np.float64) / 255.0
    xte = xte.reshape(len(xte), -1).astype(np.float64) / 255.0
    return xtr, ytr.astype(np.int64), xte, yte.astype(np.int64)


def batches(x, y, batch_size, rng):
    """Yield (x_batch, y_batch) over one shuffled pass of the data."""
    order = rng.permutation(len(x))
    for i in range(0, len(x), batch_size):
        idx = order[i:i + batch_size]
        yield x[idx], y[idx]


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
    """Test accuracy, computed in batches to keep memory use small."""
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
