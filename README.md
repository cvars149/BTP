# SENSE Framework — Privacy-Preserving Dimensionality Reduction

A research implementation of the **SENSE** (Secure Embedding via Non-anchor Substitution and Encoding) framework for anchor-based, privacy-preserving dimensionality reduction and visualization.

---

## Overview

This repository contains two complementary experiments that explore how high-dimensional data can be embedded into 2D while keeping the raw feature values of individual data points private. The core idea: a small set of **anchor points** is used as a shared reference frame. Non-anchor ("private") points are represented only through their geometric relationships to anchors — their raw coordinates are never exposed or stored.

---

## Methods

### Method 1 — `Method_1.py`: Anchor-Based Intermediate Projection on MNIST

Evaluates two dimensionality reduction strategies on MNIST (784-dimensional image data) by comparing their ability to preserve neighbourhood structure against a ground truth t-SNE embedding.

**Pipeline**

```
X_NA (784D) ──[Method(d_l)]──► Intermediate (d_l D) ──[t-SNE]──► 2D
X_NA (784D) ──────────────────────────────────────────[t-SNE]──► Ground Truth 2D
                                      │
                                 F-Score comparison
```

**Two methods compared across varying intermediate dimensions `d_l`:**

| Method | Description |
|---|---|
| **Random Projection (RP)** | Projects anchors with a random (optionally orthogonalized) matrix; non-anchor positions solved via trilateration using squared L2 distances |
| **PCA** | Projects anchors using principal components; same trilateration step for non-anchor positions |

**Key features:**
- GPU-accelerated distance matrix computation via PyTorch `cdist`
- KMeans-based anchor selection for representative coverage
- Fixed anchor count (`m = 783`) with variable dataset sizes (10%, 20%, 30% anchors)
- F-Score evaluation at `k = 10` and `k = 15` neighbours
- Fast t-SNE via `openTSNE` with sklearn fallback

**Outputs:**
- `f_score_comparison_k10_2d.png`
- `f_score_comparison_k15_2d.png`

---

### Method 2 — `Method_2.py`: OD Proxy Features on German Credit Dataset

Implements the **Orthogonal Distance (OD)** proxy encoding using the **L2 (Euclidean) norm** on the German Credit dataset. Each non-anchor point is encoded as a vector of distances from the origin to the projection point on each anchor pair line — a compact, privacy-preserving representation.

**Pipeline**

```
X_NA (20D) ──[OD Encoding (L2)]──► Proxy Features (171D) ──[t-SNE]──► 2D
```

**OD Feature Computation**

For each non-anchor point **u** and each anchor pair **(A, B)**, all distances use the **L2 norm**:

1. Compute pairwise L2 distances: `c = ||B-A||₂`, `b = ||u-A||₂`, `a = ||u-B||₂`
2. Apply Law of Cosines to get the signed projection length: `AD = (b² + c² - a²) / (2c)`
3. Compute projection parameter: `k = AD / c`
4. Compute projection point on line AB in feature space: `D = A + k*(B - A)`
5. OD feature = **L2 distance from origin O to D**: `OD = ||D||₂`

> **Key distinction**: OD is `||D||₂` (distance from origin to the foot point), **not** `||u - D||₂` (perpendicular distance from u to the line). This encoding does not reveal u's offset from the line.

The proxy-space pairwise distances between proxy vectors use the **Chebyshev (`l∞`) metric**.

**Embedding:** t-SNE (`perplexity=30`, `init='pca'`, `n_iter=1000`, `random_state=42`)

**Evaluation Metrics:**

| Metric | What it Measures |
|---|---|
| Distance Error (DE) | Frobenius norm difference between proxy and original distance matrices |
| F-Score (proxy, k=10) | Neighbour overlap between proxy and original distance matrices |
| F-Score (embedding, k=15) | Neighbour overlap between 2D embedding and original distances |
| Trustworthiness | Are new embedding neighbours actually nearby in original space? |
| Continuity | Are original neighbours still neighbours in the embedding? |
| Steadiness | Fraction of embedding clusters valid in original space |
| Cohesiveness | Fraction of original clusters preserved in the embedding |
| Pearson Correlation | Linear correlation between proxy and original distances |

**Outputs:**
- `german_credit_od_tsne_evaluation.png`
- `german_credit_od_tsne_results.txt`
- `german_credit_od_tsne_results.csv`

---

## Installation

```bash
pip install numpy torch scikit-learn openTSNE matplotlib seaborn tqdm scipy pandas
```

> `openTSNE` is used in Method 1 for faster t-SNE with automatic sklearn fallback.

**Hardware:** Method 1 benefits significantly from a CUDA GPU for distance matrix computation. Method 2 runs fully on CPU.

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
| `D_L_VALUES` | 10 log-spaced in [30, 782] | Intermediate dimensions to test |
| `K_VALUES` | `[10, 15]` | Neighbourhood sizes for F-Score |
| `METHODS` | `['random_projection', 'pca']` | Methods to compare |
| `RP_ORTHOGONALIZE` | `True` | QR orthogonalization for RP |
| `RP_SPARSE` | `False` | Sparse Achlioptas projection |
| `RIDGE_LAMBDA` | `1e-6` | Tikhonov regularization for trilateration |
| `TSNE_PERPLEXITY` | `30` | t-SNE perplexity |
| `USE_FAST_TSNE` | `True` | Use `openTSNE` (faster) |

### Method 2

```bash
python Method_2.py
```

No configuration required — runs end-to-end on the German Credit dataset with fixed settings.

---

## Project Structure

```
.
├── Method_1.py          # MNIST: RP vs PCA intermediate projection + t-SNE
├── Method_2.py          # German Credit: OD proxy features (L2) + t-SNE
└── README.md            # This file
```

**Generated outputs (after running):**

```
├── f_score_comparison_k10_2d.png
├── f_score_comparison_k15_2d.png
├── german_credit_od_tsne_evaluation.png
├── german_credit_od_tsne_results.txt
└── german_credit_od_tsne_results.csv
```

---

## Technical Background

### Anchor-Based Trilateration (Method 1)

Non-anchor positions are reconstructed purely from distances to anchors — their raw coordinates are never projected directly:

1. Translate to anchor `a_0` as origin: `A_sys = A_low[1:] - a_0`
2. Form the linear system using squared-distance identity: `B[i,j] = 0.5 * (F[i,0]² - F[i,j+1]² + ||A_sys[j]||²)`
3. Solve with Ridge regularization: `Y_rel = B @ pinv(A_sys)`
4. Translate back: `Y_final = Y_rel + a_0`

### OD Encoding (Method 2)

The OD encoding replaces raw features with the L2 distance from the coordinate origin to the projection foot point on each anchor pair line. Using the Chebyshev metric in proxy space to compare encoded points captures worst-case deviation across all anchor pairs. The raw feature vector `u` is never stored — only the scalar `OD = ||D||₂` per anchor pair is retained.

### Privacy Model

- **Anchors** are a shared public reference (known to all parties)
- **Non-anchor points** are private; only their OD-encoded proxy values are released
- `u` cannot be recovered from `||D||₂` values alone because `D` depends only on the anchors and the projection parameter `k`, not directly on `u`'s perpendicular offset from the line

---

## Citation / References

- Venna, J. & Kaski, S. (2006). *Local multidimensional scaling.* Neural Networks.
- Jeon, J. et al. (2022). *Measuring the quality of dimensionality reduction: Steadiness and Cohesiveness.* IEEE TVCG.
- Wang, Y. et al. (2021). *Understanding how dimension reduction tools work: t-SNE, UMAP, TriMAP, and PaCMAP.* JMLR.
- van der Maaten, L. & Hinton, G. (2008). *Visualizing data using t-SNE.* JMLR.
- Achlioptas, D. (2003). *Database-friendly random projections.* JCSS.
