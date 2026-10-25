# Adaptive-MGCNet: Strictly Leakage-Free Multimodal Adaptive Population Graph Convolutional Network for ASD Diagnosis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **Adaptive-MGCNet**, an end-to-end multimodal population graph convolutional network designed for Autism Spectrum Disorder (ASD) detection. This repository provides strictly leakage-free evaluation protocols across 8 balanced acquisition sites ($N=493$) from the ABIDE I dataset.

---

## 📌 Framework Overview

![Adaptive-MGCNet Architecture](figures/fig1.png)

Adaptive-MGCNet coordinates:
1. **Site Harmonization & Leakage-Free Feature Engineering**: Cross-scanner harmonization using Empirical Bayes ComBat and strictly train-partition-isolated RF-RFE (7,381 $\to$ 500 features).
2. **Adaptive Population Graph Learning**: Dynamic affinity estimation via dual-branch scaled dot-product attention with Top-K(K=5) neighborhood sparsification.
3. **Decoupled Late Fusion**: Two-layer sparse GCN graph convolution paired with deep phenotypic covariate projections.

---

## 📂 Repository Structure

```text
Adaptive-MGCNet/
├── figures/                # Architectural diagrams and explainability figures
├── models/                 # PyTorch model definitions
│   ├── __init__.py
│   ├── adaptive_graph.py   # Adaptive graph learning & Top-K sparsification
│   ├── gcn_module.py       # Two-layer sparse GCN module
│   └── adaptive_mgcnet.py  # End-to-end framework
├── preprocessing/          # Multi-site filtering & fold-isolated RFE
│   ├── __init__.py
│   ├── filter_large_sites.py
│   ├── rfe_selection.py
│   └── consensus_analysis.py
├── utils/                  # Evaluation metrics & reproducibility helpers
│   ├── __init__.py
│   ├── metrics.py
│   └── helpers.py
├── train_10fold_cv.py      # Main script for 10-fold cross-validation
├── requirements.txt        # Python package dependencies
└── README.md
```





## 🛠️ Installation

# Clone the repository
git clone [https://github.com/your-username/Adaptive-MGCNet.git](https://github.com/your-username/Adaptive-MGCNet.git)
cd Adaptive-MGCNet

# Install required dependencies
pip install -r requirements.txt





## 🚀 Quick Start

### 1. Data Preparation

Place the preprocessed feature files in the repository root directory:

- `X_fmri_7381_large_sites_combat.npy`: Harmonized tangent-space functional connectivity matrix.
- `pheno_large_sites.csv`: Phenotypic metadata for the 8 large-cohort sites.

### 2. Run 10-Fold Cross-Validation

To reproduce the primary 10-fold cross-validation benchmark:

python train_10fold_cv.py





## 📊 Benchmark Results

Performance across the 8-site balanced ABIDE cohort (N=493):

|               | ACC                     | PRE                     | SEN                      | SPE                     | AUC                     | F1                      |
| ------------- | ----------------------- | ----------------------- | ------------------------ | ----------------------- | ----------------------- | ----------------------- |
| SVM           | 68.58%  ± 9.72%         | 68.55%  ± 9.73%         | 65.10%  ± 15.08%         | 72.00%  ± 7.16%         | 75.41%  ± 8.98%         | 66.45%  ± 12.58%        |
| MLP           | 65.32%  ± 9.17%         | 64.79%  ± 9.76%         | 63.85%  ± 14.48%         | 66.80%  ± 9.47%         | 71.38%  ± 8.73%         | 63.91%  ± 11.50%        |
| PopulationGCN | 44.24%  ± 8.00%         | 43.98%  ± 7.64%         | 43.25%  ± 7.20%          | 45.20%  ± 13.39%        | 43.36%  ± 6.64%         | 42.29%  ± 8.83%         |
| DeepGCN       | 67.76%  ± 5.10%         | 68.22%  ± 5.27%         | 64.67%  ± 9.88%          | 70.80%  ± 5.95%         | 71.58%  ± 6.40%         | 66.09%  ± 6.92%         |
| DGM           | 65.11%  ± 7.23%         | 63.96%  ± 6.70%         | 65.53%  ± 18.38%         | 64.80%  ± 9.93%         | 70.28%  ± 8.79%         | 63.66%  ± 11.95%        |
| EVGCN         | 57.78%  ± 10.75%        | 57.75%  ± 12.64%        | 56.33%  ± 11.42%         | 59.20%  ± 13.00%        | 58.67%  ± 10.89%        | 56.79%  ± 11.14%        |
| DAGMNet       | 60.67%  ± 5.72%         | 61.44%  ± 6.38%         | 56.48%  ± 12.64%         | 64.80%  ± 10.85%        | 64.78%  ± 7.79%         | 58.03%  ± 7.65%         |
| AGMGC         | 61.04%  ± 3.82%         | 61.84%  ± 5.57%         | 58.37%  ± 13.72%         | 63.60%  ± 13.39%        | 65.61%  ± 4.40%         | 58.91%  ± 7.23%         |
| Ours          | **69.59%**  **± 6.63%** | **69.25%**  **± 6.87%** | **68.77%**  **± 10.62%** | **70.40%**  **± 6.97%** | **74.52%**  **± 9.34%** | **68.75%**  **± 8.04%** |



## 📜 Citation

If you find this code or our paper useful in your research, please consider citing:

@article{AdaptiveMGCNet2026,
  title={Adaptive-MGCNet: A Strictly Leakage-Free Multimodal Adaptive Population Graph Convolutional Network for Autism Spectrum Disorder Diagnosis},
  author={Your Name and Collaborators},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2026}
}