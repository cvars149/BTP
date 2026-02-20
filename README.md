# SENSE Framework — Privacy-Preserving Dimensionality Reduction

A research implementation of the **SENSE** (Secure Embedding via Non-anchor Substitution and Encoding) framework for anchor-based, privacy-preserving dimensionality reduction and visualization.

---

## Overview

This repository contains two complementary experiments that explore how high-dimensional data can be embedded into 2D while keeping the raw feature values of individual data points private. The core idea: a small set of **anchor points** is used as a shared reference frame. Non-anchor ("private") points are represented only through their distances or geometric relationships to anchors — never exposed directly.

---

## Methods

### Method 1 — `Method_1.py`: Anchor-Based Intermediate Projection on MNIST

Evaluates two dimensionality reduction strategies on MNIST (784-dimensional image data) by comparing their ability to preserve neighborhood structure against a ground truth t-SNE embedding.

**Pipeline**

```
X_NA (784D) ──[Method]──► Intermediate (d_l D) ──[t-SNE]──► 2D Embedding
X_NA (784D) ──────────────────────────────────────[t-SNE]──► Ground Truth 2D
                              │
                         F-Score comparison
```

**Two methods are compared across varying intermediate dimensions (`d_l`):**

| Method | Description |
|---|---|
| **Random Projection (RP)** | Projects anchors with a random (optionally orthogonalized) matrix; non-anchor positions are solved via trilateration using squared distances |
| **PCA** | Projects anchors using principal components; same trilateration step for non-anchor positions |

**Key Features**

- GPU-accelerated distance matrix computation via PyTorch `cdist`
- KMeans-based anchor selection for representative coverage
- Fixed anchor count (`m = 783`) with variable dataset sizes (10%, 20%, 30% anchors)
- F-Score evaluation at `k = 10` and `k = 15` neighbors
- Fast t-SNE via `openTSNE` with sklearn fallback

---

### Method 2 — `Method_2.py`: OD Proxy Features on German Credit Dataset

Implements the **Orthogonal Distance (OD)** proxy encoding using the L∞ metric on the German Credit dataset. Each non-anchor point is encoded as a vector of perpendicular foot distances to each anchor pair — a compact, privacy-preserving representation.

**Pipeline**

```
X_NA (20D) ──[OD Encoding (L∞)]──► Proxy Features (L dims) ──[PaCMAP]──► 2D Embedding
```

**OD Feature Computation**

For each non-anchor point **u** and each anchor pair **(A, B)**:

1. Compute the projection parameter: `k = (b² + c² - a²) / (2c²)` where `a, b, c` are L∞ distances
2. Compute foot-of-perpendicular: `D = A + k·(B − A)`
3. OD feature = L∞ norm of `D`

The set of all `L = C(m, 2)` OD values forms the proxy feature vector for each point.

**Evaluation Metrics**

| Metric | What it Measures |
|---|---|
| Distance Error (DE) | Frobenius norm difference between proxy and original distance matrices (lower = better) |
| F-Score (proxy, k=10) | Neighbor overlap between proxy and original distance matrices |
| F-Score (embedding, k=15) | Neighbor overlap between 2D embedding and original distances |
| Trustworthiness | Are new embedding neighbors actually nearby in original space? |
| Continuity | Are original neighbors still neighbors in the embedding? |
| Steadiness | Fraction of embedding clusters that are valid in original space |
| Cohesiveness | Fraction of original clusters preserved in the embedding |
| Pearson Correlation | Linear correlation between proxy and original distances |

---

## Installation

### Requirements

```bash
pip install numpy torch scikit-learn openTSNE matplotlib seaborn tqdm pacmap scipy pandas
```

> If `openTSNE` or `pacmap` are not available, the scripts will fall back to sklearn t-SNE or auto-install pacmap respectively.

### Hardware

- **Method 1** benefits significantly from a CUDA-capable GPU for distance matrix computation. CPU fallback is automatic.
- **Method 2** runs entirely on CPU and is feasible for the 1000-sample German Credit dataset.

---

## Usage

### Method 1

```bash
python Method_1.py
```

**Configuration** (top of file, `Config` class):

| Parameter | Default | Description |
|---|---|---|
| `NUM_ANCHORS` | `783` | Fixed number of anchor points |
| `ANCHOR_PERCENTAGES` | `[10, 20, 30]` | % of total data that are anchors |
| `D_L_VALUES` | 10 log-spaced values [30, 782] | Intermediate dimensions to test |
| `K_VALUES` | `[10, 15]` | Neighborhood sizes for F-Score |
| `METHODS` | `['random_projection', 'pca']` | Methods to compare |
| `RP_ORTHOGONALIZE` | `True` | Use QR orthogonalization for RP |
| `RP_SPARSE` | `False` | Use sparse Achlioptas projection |
| `RIDGE_LAMBDA` | `1e-6` | Tikhonov regularization for trilateration |
| `TSNE_PERPLEXITY` | `30` | t-SNE perplexity |
| `USE_FAST_TSNE` | `True` | Use `openTSNE` (faster) |

**Outputs:**
- `f_score_comparison_k10_2d.png` — F-Score vs `d_l` for k=10
- `f_score_comparison_k15_2d.png` — F-Score vs `d_l` for k=15
- Console summary with best/mean F-scores and timing

---

### Method 2

```bash
python Method_2.py
```

No configuration is required — the script runs end-to-end with the German Credit dataset and default settings.

**Outputs:**
- `german_credit_od_evaluation.png` — 2D scatter plot + metric bar chart
- `german_credit_od_results.txt` — Formatted metric table
- `german_credit_od_results.csv` — Results in CSV format

---

## Project Structure

```
.
├── Method_1.py                    # MNIST: RP vs PCA intermediate projection + t-SNE
├── Method_2.py                    # German Credit: OD proxy features + PaCMAP
└── README.md                      # This file
```

**Generated outputs (after running):**

```
├── f_score_comparison_k10_2d.png
├── f_score_comparison_k15_2d.png
├── german_credit_od_evaluation.png
├── german_credit_od_results.txt
└── german_credit_od_results.csv
```

---

## Technical Background

### Anchor-Based Trilateration

Both methods reconstruct non-anchor positions using the same trilateration approach:

Given anchor embeddings `A_low` (shape `[m, d_l]`) and distances `F` from non-anchors to anchors:

1. Translate to anchor `a_0` as origin: `A_sys = A_low[1:] - a_0`
2. Form the linear system using squared-distance identity:
   ```
   B[i, j] = 0.5 * (F[i,0]² - F[i,j+1]² + ||A_sys[j]||²)
   ```
3. Solve: `Y_rel = B @ pinv(A_sys)` with Ridge regularization
4. Translate back: `Y_final = Y_rel + a_0`

This means non-anchor points are never projected directly — their coordinates are *inferred* from distances alone, which is the core privacy guarantee.

### Privacy Model

The SENSE framework assumes:
- **Anchors** are a public, shared reference (released or known to all parties)
- **Non-anchor points** are private; only their distances to anchors are revealed
- The reconstruction quality is bounded by the number of anchors and the intermediate dimension

### OD Encoding (Method 2)

The Orthogonal Distance encoding replaces raw features with geometric relationships. Using L∞ instead of L2 for distance calculations changes the shape of neighborhoods (L∞ balls are hypercubes), which can be advantageous for high-dimensional tabular data where features have very different scales.

---

## Evaluation Notes

- **F-Score** is the primary metric in Method 1; it directly measures how well local structure is preserved between the reconstructed and ground truth embeddings.
- **Trustworthiness vs Continuity** in Method 2 measure complementary aspects: trustworthiness penalizes false neighbors (tears in embedding), while continuity penalizes missing neighbors (compressions).
- **Steadiness vs Cohesiveness** (Jeon et al., 2021) are cluster-level analogs: steadiness checks embedding clusters against original space; cohesiveness checks original clusters against the embedding.

---

## Citation / References

- Jeon, I., et al. (2021). *Measuring the quality of dimensionality reduction: An information-theoretic approach.* (Steadiness & Cohesiveness)
- van der Maaten, L., & Hinton, G. (2008). *Visualizing data using t-SNE.*
- Wang, Y., et al. (2021). *Understanding How Dimension Reduction Tools Work: An Empirical Approach to Deciphering t-SNE, UMAP, TriMAP, and PaCMAP.*
- Achlioptas, D. (2003). *Database-friendly random projections.*
