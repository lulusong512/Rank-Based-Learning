# rbl_pkg

**rbl_pkg** is a Python package implementing **Rank-Based Learning (RBL)** — a novel learning algorithm designed to classify binary outcomes (e.g., disease vs. control) by learning the optimal feature ranking profile from high-throughput omics data. RBL estimates a feature permutation that best distinguishes cases from controls based on ranking similarity, leveraging the **Metropolis–Hastings** stochastic search algorithm.

---

## Key Features

- Rank-based classification that is robust to batch effects and missing data  
- Custom similarity scoring between ranking profiles  
- Metropolis–Hastings optimization for searching permutations  
- Evaluation via AUC and confidence intervals  
- Parallel computing support with `joblib`

---
## Cite

If you use this code in any published work, please cite XX (placeholder for the forthcoming publication).

## Installation

### 1. Create a virtual environment (using conda)

This package supports Python versions 3.8 to 3.12. The following commands create an environment named `rbl_env` with Python 3.11 (you may change the environment name and Python version as needed):

```bash
conda create -n rbl_env python=3.11
conda activate rbl_env
```

### 2. Install package

```bash
pip install rbl_pkg
```



