#!/usr/bin/env python3

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder
from sklearn.manifold import TSNE


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def calculate_distance_error(G_proxy, G_original):
    return np.linalg.norm(G_proxy - G_original, 'fro') / np.linalg.norm(G_original, 'fro')


def calculate_fscore_neighbors(G_pred, G_true, k):
    n = G_pred.shape[0]
    nn_true = np.argsort(G_true, axis=1)[:, 1:k+1]
    nn_pred = np.argsort(G_pred, axis=1)[:, 1:k+1]

    tp = fp = fn = 0
    for i in range(n):
        t, p = set(nn_true[i]), set(nn_pred[i])
        tp += len(t & p)
        fp += len(p - t)
        fn += len(t - p)

    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom > 0 else 0.0


def calculate_pearson_correlation(G_true, G_recon):
    idx = np.triu_indices(G_true.shape[0], k=1)
    corr, _ = pearsonr(G_true[idx], G_recon[idx])
    return corr


def calculate_trustworthiness(G_original, Y, k=10):
    D_y = cdist(Y, Y)
    N = G_original.shape[0]

    nn_orig = np.argsort(G_original, axis=1)
    nn_y    = np.argsort(D_y, axis=1)

    tw_sum = 0.0
    for i in range(N):
        U = set(nn_y[i, 1:k+1]) - set(nn_orig[i, 1:k+1])
        for j in U:
            tw_sum += np.where(nn_orig[i] == j)[0][0] - k

    norm = 2.0 / (N * k * (2 * N - 3 * k - 1))
    return 1.0 - norm * tw_sum


def calculate_continuity(G_original, Y, k=10):
    D_y = cdist(Y, Y)
    N = G_original.shape[0]

    nn_orig = np.argsort(G_original, axis=1)
    nn_y    = np.argsort(D_y, axis=1)

    cont_sum = 0.0
    for i in range(N):
        V = set(nn_orig[i, 1:k+1]) - set(nn_y[i, 1:k+1])
        for j in V:
            cont_sum += np.where(nn_y[i] == j)[0][0] - k

    norm = 2.0 / (N * k * (2 * N - 3 * k - 1))
    return 1.0 - norm * cont_sum


def calculate_steadiness_cohesiveness(G_original, Y, n_iter=1000, k=None):
    D_y = cdist(Y, Y)
    N = G_original.shape[0]

    if k is None:
        k = max(10, int(0.05 * N))

    np.random.seed(42)
    stead, cohes = [], []

    for _ in range(n_iter):
        i = np.random.randint(N)
        stead.append(
            len(set(np.argsort(D_y[i])[1:k+1]) &
                set(np.argsort(G_original[i])[1:k+1])) / k
        )

    for _ in range(n_iter):
        i = np.random.randint(N)
        cohes.append(
            len(set(np.argsort(G_original[i])[1:k+1]) &
                set(np.argsort(D_y[i])[1:k+1])) / k
        )

    return np.mean(stead), np.mean(cohes)


# =============================================================================
# MAIN
# =============================================================================

print("=" * 80)
print("SENSE – OD Proxy + t-SNE | German Credit Dataset")
print("=" * 80)

# -------------------------------------------------------------------------
# Load Dataset
# -------------------------------------------------------------------------
data = fetch_openml("credit-g", version=1, parser="auto")
X_raw = data.data
y_raw = data.target.values

cat_cols = X_raw.select_dtypes(include=["object"]).columns
num_cols = X_raw.select_dtypes(include=[np.number]).columns

X_enc = X_raw.copy()
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_enc[cat_cols] = encoder.fit_transform(X_raw[cat_cols])

X = X_enc.values.astype(float)
y = y_raw

N, d = X.shape
m = d - 1

# -------------------------------------------------------------------------
# Anchor Selection
# -------------------------------------------------------------------------
np.random.seed(42)
anchor_idx = np.random.choice(N, m, replace=False)
mask = np.ones(N, dtype=bool)
mask[anchor_idx] = False

X_A  = X[anchor_idx]
X_NA = X[mask]
y_NA = y[mask]

# -------------------------------------------------------------------------
# OD Proxy Construction
# -------------------------------------------------------------------------
P = list(combinations(range(m), 2))
L = len(P)
n = X_NA.shape[0]

X_proxy = np.zeros((n, L))

for p_idx in range(n):
    u = X_NA[p_idx]
    for k_idx, (i, j) in enumerate(P):
        A, B = X_A[i], X_A[j]
        c = np.linalg.norm(B - A)
        if c == 0:
            continue
        b = np.linalg.norm(u - A)
        a = np.linalg.norm(u - B)
        AD = (b**2 + c**2 - a**2) / (2 * c)
        D = A + (AD / c) * (B - A)
        X_proxy[p_idx, k_idx] = np.linalg.norm(D)

# -------------------------------------------------------------------------
# Distance Matrices
# -------------------------------------------------------------------------
G_orig  = cdist(X_NA, X_NA)
G_proxy = cdist(X_proxy, X_proxy, metric="chebyshev")

# -------------------------------------------------------------------------
# t-SNE Embedding
# -------------------------------------------------------------------------
tsne = TSNE(
    n_components=2,
    perplexity=30,
    init="pca",
    learning_rate="auto",
    n_iter=1000,
    random_state=42,
)
Y = tsne.fit_transform(X_proxy)
G_Y = cdist(Y, Y)

# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------
results = {
    "Distance Error": calculate_distance_error(G_proxy, G_orig),
    "F-score (proxy, k=10)": calculate_fscore_neighbors(G_proxy, G_orig, 10),
    "F-score (embed, k=15)": calculate_fscore_neighbors(G_orig, G_Y, 15),
    "Trustworthiness": calculate_trustworthiness(G_orig, Y, 10),
    "Continuity": calculate_continuity(G_orig, Y, 10),
    "Pearson": calculate_pearson_correlation(G_orig, G_proxy),
}

s, c = calculate_steadiness_cohesiveness(G_orig, Y)
results["Steadiness"] = s
results["Cohesiveness"] = c

# -------------------------------------------------------------------------
# Results Table
# -------------------------------------------------------------------------
df = pd.DataFrame(
    {"Metric": results.keys(), "Value": [f"{v:.6f}" for v in results.values()]}
)
print(df)

# -------------------------------------------------------------------------
# Visualization
# -------------------------------------------------------------------------
y_plot = np.array([1 if yy == "good" else 0 for yy in y_NA])

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.scatter(Y[:, 0], Y[:, 1], c=y_plot, cmap="viridis", s=30, alpha=0.6)
plt.title("OD Proxy → t-SNE Embedding")
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(df["Metric"], df["Value"].astype(float))
plt.xticks(rotation=45, ha="right")
plt.title("Evaluation Metrics")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("german_credit_od_tsne_evaluation.png", dpi=300)
plt.show()

print("\nSaved: german_credit_od_tsne_evaluation.png")
print("=" * 80)
print("DONE")
print("=" * 80)
