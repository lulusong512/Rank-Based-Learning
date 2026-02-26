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

If you use this code in any published work, please cite @article{song2025rankbased,
  title={Rank-based learning: A novel high-throughput algorithm resilient to missing data and effective for datasets with small sample size},
  author={Song, Lulu and Rudsari, Hamid Khoshfekr and Fahrmann, Johannes F and Vykoukal, Jody and Hanash, Sam and Long, James P and Irajizad, Ehsan},
  journal={Briefings in Bioinformatics},
  volume={26},
  number={6},
  year={2025},
  doi={10.1093/bib/bbaf666}
}.

## Installation

### 1. Create a virtual environment (using conda)

This package supports Python versions 3.8 to 3.12. The following commands create an environment named `rbl_env` with Python 3.11 (you may change the environment name and Python version as needed):

```bash
conda create -n rbl_env python=3.11
conda activate rbl_env
```

### 2. Install package directly from GitHub

```bash
git clone https://github.com/lulusong512/Rank-Based-Learning.git
cd Rank-Based-Learning
pip install -e .
```



