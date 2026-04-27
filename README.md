
## Overview

MRWFDBMC predicts potential circRNA–disease associations by leveraging a multimodal random walk fusion strategy and an SVD-based matrix completion framework.
Framework. 
## Repository structure
```text
MRWFDBMC/
|-- README.md
|-- code/
|   |-- main.py                # Main training and cross-validation evaluation script
|   |-- model.py               # MRWFDBMC model definition
|   |-- layer.py               # Encoders, decoders, and hypergraph neural network layers
|   |-- function.py            # Result extraction and graph utility functions
|   |-- NMF.py                 # SVD-based matrix completion module
|   |-- hypergraph_utils.py    # Hypergraph construction and incidence matrix utilities
|   |-- randomfusion.py        # Multimodal random-walk fusion module
|   `-- utils.py               # Evaluation metric functions
|-- circRNAdisease(533,89)/
|   |-- association.txt
|   |-- GKGIP_circRNA.txt
|   |-- GKGIP_disease.txt
|   |-- LKGIP_circRNA.txt
|   `-- LKGIP_disease.txt
`-- circRNAdisease(514,62)/
    |-- association.txt
    |-- GKGIP_circRNA.txt
    |-- GKGIP_disease.txt
    |-- LKGIP_circRNA.txt
    `-- LKGIP_disease.txt
```

## Data description

Each dataset folder contains:

- `association.txt`: binary association matrix
- `GKGIP_circRNA.txt`: Gaussian kernel similarity matrix for circRNA
- `GKGIP_disease.txt`: Gaussian kernel similarity matrix for disease
- `LKGIP_circRNA.txt`: Laplace kernel similarity matrix for circRNA
- `LKGIP_disease.txt`: Laplace kernel similarity matrix for disease

The folder name `circRNAdisease(m,n)` indicates the size of the association matrix, where `m` is the number of circRNAs and `n` is the number of diseases.

## Requirements

The code imports the following Python packages:

- Python 3.8+
- PyTorch
- NumPy
- SciPy
- scikit-learn
- matplotlib

You can install the common dependencies with:

```bash
pip install torch numpy scipy scikit-learn matplotlib
```

## Important note before running

The current `code/main.py` uses **hard-coded dataset paths**. In the public repository, the script currently points to:

```python
MD = np.loadtxt("circRNAdisease(533,89)/association.txt")
```

Before running the code, please edit the dataset section in `code/main.py` and switch to one of the available datasets.

For example, to use `circRNAdisease(514,62)`, uncomment or edit the corresponding lines:

```python
MD = np.loadtxt("circRNAdisease(514,62)/association.txt")
C1 = np.loadtxt("circRNAdisease(514,62)/GKGIP_circRNA.txt")
D1 = np.loadtxt("circRNAdisease(514,62)/GKGIP_disease.txt")
C2 = np.loadtxt("circRNAdisease(514,62)/LKGIP_circRNA.txt")
D2 = np.loadtxt("circRNAdisease(514,62)/LKGIP_disease.txt")
```

## How to run

After selecting an available dataset in `code/main.py`, run the following commands:

```bash
cd code
python main.py
```

If you want to disable GPU acceleration, run:

```bash
python main.py --no-cuda
```

## Output

During execution, the script prints training and evaluation statistics, including:

- fold-level AUC and AUPR
- precision, recall, F1-score, MCC, and accuracy
- average performance across cross-validation folds


## Citation

If you use this code in your research, please cite the corresponding MRWFDBMC paper.
