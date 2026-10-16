# CS5131 · Computer Vision

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

Coursework for **CS5131 Computer Vision** at the University of Cincinnati, built from first
principles in NumPy: frequency-domain image analysis, then neural networks — an MLP, a
convolutional network and attention — with every backward pass written by hand and no
autograd framework.

| | |
|---|---|
| **Instructor** | Dr. Vikram Ravindra |
| **Student** | Sophia Munoz · [munozsa@mail.uc.edu](mailto:munozsa@mail.uc.edu) |
| **Textbook** | *Foundations of Computer Vision* — Torralba, Isola & Freeman |
| **Last updated** | October 16, 2026 |

---

## Assignments

| # | Assignment | Topics | Files | Status |
|:-:|---|---|---|:-:|
| 1 | [**Fourier Analysis**](assignments/assignment1) | Transfer functions, predicting spectra, amplitude vs. phase, shifts, convolution in the spatial and frequency domains | [Notebook](assignments/assignment1/fourier_assignment.ipynb) · [HTML](assignments/assignment1/fourier_assignment.html) | ✅ |
| 2 | [**Backpropagation from Scratch**](assignments/assignment2) | Softmax & cross-entropy, numerical stability, linear / ReLU / softmax layers, gradient checking, training an MLP on MNIST | [Notebook](assignments/assignment2/a2_assignment.ipynb) · [HTML](assignments/assignment2/a2_assignment.html) | ✅ |
| 3 | [**Convolution and Attention from Scratch**](assignments/assignment3) | `Conv2D` forward & backward, max pooling, training a CNN on MNIST and comparing it to the MLP; attention as a weighted average, √m scaling, permutation equivariance & positional codes, row softmax | [Notebook](assignments/assignment3/a3_assignment.ipynb) | 🚧 |

> **Tip:** completed assignments have an exported `.html` copy, so you can read the results
> with all outputs visible without running anything.

---

## Repository structure

```text
computer-vision-munozsa/
├── README.md
└── assignments/
    ├── assignment1/          # Fourier Analysis
    │   ├── fourier_assignment.ipynb
    │   ├── fourier_assignment.html
    │   └── images/           # test images (boat, building, facade, stripes, texture)
    ├── assignment2/          # Backpropagation from Scratch
    │   ├── a2_assignment.ipynb
    │   ├── a2_assignment.html
    │   ├── a2utils.py        # data loading + gradient checking helpers
    │   ├── mnist_data/       # MNIST, bundled — no download needed
    │   ├── requirements.txt
    │   └── README.md
    └── assignment3/          # Convolution and Attention from Scratch
        ├── a3_assignment.ipynb
        ├── a3utils.py        # Assignment 2 layers + LayerNorm, Residual, MeanPool
        ├── mnist_data/
        ├── requirements.txt
        └── README.md
```

---

## Running the notebooks

```bash
git clone https://github.com/munozsophia/computer-vision-munozsa.git
cd computer-vision-munozsa

# create a virtual environment and install dependencies (NumPy, Matplotlib, Jupyter)
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r assignments/assignment2/requirements.txt

# launch Jupyter from inside an assignment folder so relative paths resolve
cd assignments/assignment3
jupyter notebook a3_assignment.ipynb
```

Each notebook expects its data and helper files to sit next to it, so open it from inside
its own folder. The [Assignment 2](assignments/assignment2/README.md) and
[Assignment 3](assignments/assignment3/README.md) READMEs cover setup details and
troubleshooting.

---

<sub>Repository: [github.com/munozsophia/computer-vision-munozsa](https://github.com/munozsophia/computer-vision-munozsa)</sub>