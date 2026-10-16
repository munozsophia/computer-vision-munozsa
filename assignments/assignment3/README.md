# Assignment 3 — Convolution and Attention, from Scratch

Computer Vision &nbsp;·&nbsp; Due **Friday 23 October 2026**, 11:59 PM &nbsp;·&nbsp; 54 points

## Setting up

This folder is self-contained. Nothing is downloaded; MNIST is already in `mnist_data/`.

    python3 -m venv ~/cvenv
    source ~/cvenv/bin/activate          # Windows: ~\cvenv\Scripts\activate
    pip install -r requirements.txt
    jupyter notebook a3_assignment.ipynb

NumPy and matplotlib are the only requirements. Keep `a3utils.py` and `mnist_data/` in the same
folder as the notebook.

## What is in here

| file | |
|---|---|
| `a3_assignment.ipynb` | the assignment — this is the only file you edit |
| `a3utils.py` | provided code; do not edit it, but do read it |
| `mnist_data/` | the dataset |
| `requirements.txt` | pinned versions |

## What you are given

`a3utils.py` contains the layers you wrote in Assignment 2 — `Linear`, `ReLU`,
`SoftmaxCrossEntropy`, `MLP` and `train` — so this assignment stands on its own. Use them rather
than your own, so that everyone's comparison numbers mean the same thing.

Two small changes from the Assignment 2 versions, both noted in the file: `Linear` now applies to
the **last axis** of an array with any number of leading axes (on 2-D input it is identical to
what you wrote), and `train` takes an optional `momentum` argument that nothing here needs.

`LayerNorm`, `Residual` and `MeanPool` are also in there. Nothing in the notebook calls them —
they are the pieces of the vision transformer that Part B asks about, put where you can read them
instead of taking the questions' word for what they do. `Residual.backward` is two lines and is
the clearest example you will see of the rule you proved in Assignment 2, Question A2(a).

## Rules

- **NumPy only.** No PyTorch, TensorFlow, JAX, autograd, or anything that computes gradients for
  you. The point is to write the backward passes yourself.
- Edit only the cells marked `# YOUR CODE HERE` and the *Your answer* cells.
- **Get each gradient check to print `PASS` before moving on.** A layer whose backward pass is
  wrong will make every later part fail in ways that are very hard to read. A relative error near
  `1e-10` means correct; `1e-2` or worse means a bug. The check tells you *which* gradient is
  wrong, which is usually enough to find it.
- Show your working on the theory questions. A correct final expression with no derivation earns
  little credit.

## How long it takes

Budget **6 to 9 hours**. Roughly:

| | hours |
|---|---|
| A1 `Conv2D` forward and backward | 2–4 |
| A2, A3 pooling and flatten | 1 |
| A4, A5 training and the comparison | 1–1.5 |
| B1–B3 theory | 1.5–2.5 |
| B4 `RowSoftmax` | 0.5 |

`Conv2D.backward` is the hard part, and it is meant to be. If it is running much over four hours,
the usual cause is debugging against MNIST instead of against the gradient check: the check runs
on a `(2, 3, 6, 6)` array in under a second and tells you *which* of `input`, `W` and `b` is
wrong, which is nearly always enough to find the error. Training tells you only that something
is wrong, ten minutes later.

A complete *run* is only about three minutes, almost all of it the CNN at roughly 40 seconds an
epoch. While you are debugging, train on a subset — `x_train[:5000]` — to get feedback in
seconds, and remember to switch back before your final run. A gradient check takes under a
second, so there is never a reason to train a network whose checks have not passed.

## Suggested schedule

Part A is most of the work; start it in week 1. B1 to B3 need no computer, so they are the part
to do on a train or over coffee.

## Submitting

Submit the notebook **executed from top to bottom** with all outputs visible
(*Kernel → Restart & Run All*), together with `a3utils.py`. Every gradient check should print
`PASS`.

If you use a coding assistant or language model, add a short note at the end of the notebook
saying what you used and for what. This is not penalised; failing to disclose it is.
