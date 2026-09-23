# Assignment 2 — Backpropagation from Scratch

**CS5131 / CS6031 Computer Vision** · Due **Tuesday, October 6, 2026, 11:59 PM** · 65 points

## What's in this folder

| file | what it is |
|---|---|
| `a2_assignment.ipynb` | the assignment — this is the file you work in |
| `a2utils.py` | provided helper code: data loading and gradient checking. Do not edit it. |
| `mnist_data/` | the MNIST dataset, already included, so nothing needs downloading |
| `requirements.txt` | the Python packages you need |

## Setup

1. Keep everything in this folder together. The notebook looks for `a2utils.py` and
   `mnist_data/` next to itself.
2. Install the packages, once:

       pip install -r requirements.txt

3. Open `a2_assignment.ipynb` in Jupyter, or in VS Code with the Jupyter extension, from
   inside this folder.

## If something goes wrong

- **`ModuleNotFoundError: No module named 'a2utils'`** — the notebook is not in the same
  folder as `a2utils.py`. Move them back together.
- **You replaced or edited `a2utils.py` and nothing changed** — restart the kernel
  (*Kernel → Restart*) and run the notebook again from the top. Python does not re-read a
  module it has already imported.
- **A gradient check prints `FAIL`** — fix that layer before moving on. The check tells you
  which array is wrong (`input`, `W` or `b`), which narrows down where to look.

## Submission

Submit `a2_assignment.ipynb`, executed from top to bottom with all outputs visible
(*Kernel → Restart & Run All*), together with `a2utils.py`. Every gradient check should
print `PASS`.
