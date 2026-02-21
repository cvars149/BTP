
"""
Method 2: SENSE Framework - OD (Orthogonal Distance) Method on German Credit Dataset
=====================================================================================
Pipeline:
  1. Load German Credit dataset from OpenML
  2. Encode categorical features with OrdinalEncoder
  3. Select anchors randomly (m = d_h - 1)
  4. Compute OD proxy features using L2 norm for all anchor pairs
  5. Compute distance matrices (proxy: Chebyshev; original: Euclidean)
  6. Generate 2D embedding with t-SNE
  7. Evaluate with: Distance Error, F-Score, Trustworthiness, Continuity,
     Steadiness, Cohesiveness, Pearson Correlation

Key idea: For each non-anchor point u and each pair of anchors (A, B),
project u onto the line AB using the Law of Cosines (L2 distances).
The OD feature is the L2 distance from the origin O to the projection
point D on the line. The set of all such OD values forms the proxy space.
The raw coordinates of u are never stored or transmitted.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from itertools import combinations
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder
from sklearn.manifold import TSNE
from scipy.stats import pearsonr
import pandas as pd


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_distance_error(G_proxy, G_original):
    """
    Relative Frobenius norm error between proxy and original distance matrices.
    Lower is better; 0 means perfect distance preservation.
    """
    return np.linalg.norm(G_proxy - G_original, 'fro') / np.linalg.norm(G_original, 'fro')


def calculate_fscore_neighbors(G_pred, G_true, k):
    """
    F-score based on k-nearest neighbour overlap between two distance matrices.

    Parameters
    ----------
    G_pred : (n, n) candidate distance matrix
    G_true : (n, n) reference distance matrix
    k      : int, number of neighbours

    Returns
    -------
    float : F-score in [0, 1]
    """
    n = G_pred.shape[0]
    neighbors_true = np.argsort(G_true, axis=1)[:, 1:k+1]
    neighbors_pred = np.argsort(G_pred, axis=1)[:, 1:k+1]

    tp_total = fp_total = fn_total = 0

    for i in range(n):
        true_set = set(neighbors_true[i])
        pred_set = set(neighbors_pred[i])
        tp_total += len(true_set & pred_set)
        fp_total += len(pred_set - true_set)
        fn_total += len(true_set - pred_set)

    denom = 2 * tp_total + fp_total + fn_total
    return (2 * tp_total) / denom if denom > 0 else 0.0


def calculate_pearson_correlation(G_true, G_reconstructed):
    """
    Pearson correlation between upper-triangle entries of two distance matrices.
    Values near 1 indicate strong linear agreement.
    """
    rows, cols = np.triu_indices(G_true.shape[0], k=1)
    corr, _ = pearsonr(G_true[rows, cols], G_reconstructed[rows, cols])
    return corr


def calculate_trustworthiness(X_original, Y_embedding, k=10):
    """
    Trustworthiness: penalises false neighbours introduced by the embedding.
    High value means no spurious neighbours are added.

    Parameters
    ----------
    X_original  : (n, n) distance matrix or (n, d) raw features
    Y_embedding : (n, 2) 2-D embedding
    k           : neighbourhood size

    Returns
    -------
    float : trustworthiness in [0, 1]
    """
    D_original = X_original if (
        len(X_original.shape) == 2 and X_original.shape[0] == X_original.shape[1]
    ) else cdist(X_original, X_original, metric='euclidean')

    D_embedding = cdist(Y_embedding, Y_embedding, metric='euclidean')
    N = D_original.shape[0]

    nn_original  = np.argsort(D_original,  axis=1)[:, 1:k+1]
    nn_embedding = np.argsort(D_embedding, axis=1)[:, 1:k+1]

    trustworthiness_sum = 0.0
    for i in range(N):
        false_neighbors = set(nn_embedding[i]) - set(nn_original[i])
        for j in false_neighbors:
            rank_original = np.where(np.argsort(D_original[i]) == j)[0][0]
            trustworthiness_sum += (rank_original - k)

    normalization = (2.0 / (N * k * (2 * N - 3 * k - 1))) if k > 1 else 0.0
    return 1.0 - normalization * trustworthiness_sum


def calculate_continuity(X_original, Y_embedding, k=10):
    """
    Continuity: penalises original neighbours torn apart in the embedding.
    High value means original neighbourhoods are preserved.

    Parameters
    ----------
    X_original  : (n, n) distance matrix or (n, d) raw features
    Y_embedding : (n, 2) 2-D embedding
    k           : neighbourhood size

    Returns
    -------
    float : continuity in [0, 1]
    """
    D_original = X_original if (
        len(X_original.shape) == 2 and X_original.shape[0] == X_original.shape[1]
    ) else cdist(X_original, X_original, metric='euclidean')

    D_embedding = cdist(Y_embedding, Y_embedding, metric='euclidean')
    N = D_original.shape[0]

    nn_original  = np.argsort(D_original,  axis=1)[:, 1:k+1]
    nn_embedding = np.argsort(D_embedding, axis=1)[:, 1:k+1]

    continuity_sum = 0.0
    for i in range(N):
        missing_neighbors = set(nn_original[i]) - set(nn_embedding[i])
        for j in missing_neighbors:
            rank_embedding = np.where(np.argsort(D_embedding[i]) == j)[0][0]
            continuity_sum += (rank_embedding - k)

    normalization = (2.0 / (N * k * (2 * N - 3 * k - 1))) if k > 1 else 0.0
    return 1.0 - normalization * continuity_sum


def calculate_steadiness_cohesiveness_exact(X_original, Y_embedding, n_iterations=1000, k=None):
    """
    Steadiness and Cohesiveness via set-overlap (Jeon et al., 2021).

    Steadiness  : form cluster in embedding  → validate in original space.
    Cohesiveness: form cluster in original   → validate in embedding.

    Parameters
    ----------
    X_original   : (n, n) distance matrix or (n, d) raw features
    Y_embedding  : (n, 2) 2-D embedding
    n_iterations : number of random seed points sampled
    k            : cluster size (default: max(10, 5% of N))

    Returns
    -------
    (steadiness, cohesiveness) : both floats in [0, 1]
    """
    D_original = X_original if (
        len(X_original.shape) == 2 and X_original.shape[0] == X_original.shape[1]
    ) else cdist(X_original, X_original, metric='euclidean')

    D_embedding = cdist(Y_embedding, Y_embedding, metric='euclidean')
    N = D_original.shape[0]

    if k is None:
        k = max(10, int(N * 0.05))

    np.random.seed(42)
    steadiness_scores   = []
    cohesiveness_scores = []

    # Steadiness: cluster in embedding -> check in original
    for _ in range(n_iterations):
        center_idx  = np.random.randint(0, N)
        cluster_emb = np.argsort(D_embedding[center_idx])[1:k+1]
        true_orig   = np.argsort(D_original[center_idx])[1:k+1]
        steadiness_scores.append(len(np.intersect1d(cluster_emb, true_orig)) / k)

    # Cohesiveness: cluster in original -> check in embedding
    for _ in range(n_iterations):
        center_idx   = np.random.randint(0, N)
        cluster_orig = np.argsort(D_original[center_idx])[1:k+1]
        true_emb     = np.argsort(D_embedding[center_idx])[1:k+1]
        cohesiveness_scores.append(len(np.intersect1d(cluster_orig, true_emb)) / k)

    return np.mean(steadiness_scores), np.mean(cohesiveness_scores)


# =============================================================================
# MAIN
# =============================================================================

print("=" * 80)
print("SENSE Framework - German Credit Dataset (OD Method + t-SNE)")
print("=" * 80)

# ─── 1. Load & Encode Data ────────────────────────────────────────────────────
print("\nLoading German Credit Dataset from OpenML...")
german_credit = fetch_openml('credit-g', version=1, parser='auto')
X_full = german_credit.data
y_full = german_credit.target.values

categorical_cols = X_full.select_dtypes(include=['object', 'category']).columns
numerical_cols   = X_full.select_dtypes(include=[np.number]).columns
print(f"Categorical columns: {len(categorical_cols)}")
print(f"Numerical columns:   {len(numerical_cols)}")

X_encoded = X_full.copy()
if len(categorical_cols) > 0:
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_encoded[categorical_cols] = encoder.fit_transform(X_full[categorical_cols])

X_all = X_encoded.values.astype(float)

# ─── 2. Anchor Selection ──────────────────────────────────────────────────────
np.random.seed(42)
X = X_all
y = y_full
N, d_h = X.shape
m = d_h - 1   # number of anchors = d_h - 1

anchor_indices     = np.random.choice(N, size=m, replace=False)
non_anchor_mask    = np.ones(N, dtype=bool)
non_anchor_mask[anchor_indices] = False
non_anchor_indices = np.where(non_anchor_mask)[0]

X_A  = X[anchor_indices]
X_NA = X[non_anchor_indices]
y_NA = y[non_anchor_indices]

n = len(non_anchor_indices)

print(f"\nDataset Shape:               {X.shape}")
print(f"Dimension (d_h):             {d_h}")
print(f"Number of Anchors (m=d_h-1): {m}")
print(f"Number of Private Points:    {n}")

# ─── 3. OD Proxy Feature Computation (L2) ────────────────────────────────────
P = list(combinations(range(m), 2))
L = len(P)
print(f"Anchor Pairs (L = C(m,2)):   {L}")
print(f"Proxy Space Dimension:       {L}")

print("\nComputing OD Features (L2 norm)...")
X_proxy_OD = np.zeros((n, L))

for point_idx in range(n):
    u = X_NA[point_idx]
    for pair_idx, (i, j) in enumerate(P):
        A = X_A[i]
        B = X_A[j]

        # All three distances computed with L2 norm
        c = np.linalg.norm(B - A, ord=2)
        b = np.linalg.norm(u - A, ord=2)
        a = np.linalg.norm(u - B, ord=2)

        if c != 0:
            AD    = (b**2 + c**2 - a**2) / (2 * c)   # signed projection length
            k_val = AD / c                             # normalised parameter along AB
        else:
            k_val = 0

        D  = A + k_val * (B - A)       # projection point D on line aᵢaⱼ
        OD = np.linalg.norm(D, ord=2)  # distance from origin O to D

        X_proxy_OD[point_idx, pair_idx] = OD

# ─── 4. Distance Matrices ─────────────────────────────────────────────────────
print("Computing distance matrices...")
G_original_L2 = cdist(X_NA,       X_NA,       metric='euclidean')  # original space
G_proxy_OD    = cdist(X_proxy_OD, X_proxy_OD, metric='chebyshev')  # proxy space

# ─── 5. t-SNE Embedding ───────────────────────────────────────────────────────
print("Generating 2D embedding using t-SNE...")
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    init='pca',
    n_iter=1000,
    random_state=42,
    verbose=1
)
Y_OD   = tsne.fit_transform(X_proxy_OD)
G_Y_OD = cdist(Y_OD, Y_OD, metric='euclidean')

# ─── 6. Evaluation Metrics ───────────────────────────────────────────────────
print("\nCalculating evaluation metrics...")
results = {
    'DE_proxy':        calculate_distance_error(G_proxy_OD, G_original_L2),
    'FS_proxy_10':     calculate_fscore_neighbors(G_proxy_OD, G_original_L2, k=10),
    'FS_embedding_15': calculate_fscore_neighbors(G_original_L2, G_Y_OD, k=15),
    'Trustworthiness': calculate_trustworthiness(G_original_L2, Y_OD, k=10),
    'Continuity':      calculate_continuity(G_original_L2, Y_OD, k=10),
    'Pearson_corr':    calculate_pearson_correlation(G_original_L2, G_proxy_OD),
}

steadiness, cohesiveness = calculate_steadiness_cohesiveness_exact(G_original_L2, Y_OD)
results['Steadiness']   = steadiness
results['Cohesiveness'] = cohesiveness

# ─── 7. Print Results Table ───────────────────────────────────────────────────
print("\n" + "=" * 80)
print("EVALUATION METRICS")
print("=" * 80)

metrics_data = {
    'Metric': [
        'Distance Error',
        'F-score (proxy, k=10)',
        'F-score (embedding, k=15)',
        'Trustworthiness',
        'Continuity',
        'Steadiness',
        'Cohesiveness',
        'Pearson Correlation',
    ],
    'Value': [
        f"{results['DE_proxy']:.6f}",
        f"{results['FS_proxy_10']:.6f}",
        f"{results['FS_embedding_15']:.6f}",
        f"{results['Trustworthiness']:.6f}",
        f"{results['Continuity']:.6f}",
        f"{results['Steadiness']:.6f}",
        f"{results['Cohesiveness']:.6f}",
        f"{results['Pearson_corr']:.6f}",
    ]
}

df = pd.DataFrame(metrics_data)
print()
print("+" + "-"*30 + "+" + "-"*12 + "+")
print("| {:<28} | {:<10} |".format("Metric", "Value"))
print("+" + "="*30 + "+" + "="*12 + "+")
for _, row in df.iterrows():
    print("| {:<28} | {:<10} |".format(row['Metric'], row['Value']))
print("+" + "-"*30 + "+" + "-"*12 + "+")

# ─── 8. Visualisation ────────────────────────────────────────────────────────
y_NA_numeric = np.array([1 if label == 'good' else 0 for label in y_NA])

fig = plt.figure(figsize=(16, 5))

ax1 = plt.subplot(1, 2, 1)
scatter = ax1.scatter(Y_OD[:, 0], Y_OD[:, 1], c=y_NA_numeric,
                      cmap='viridis', alpha=0.6, s=30)
ax1.set_title('OD Method - 2D Embedding (t-SNE)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Dimension 1')
ax1.set_ylabel('Dimension 2')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax1, label='Class (0=bad, 1=good)')

ax2 = plt.subplot(1, 2, 2)
metrics_list = ['DE', 'F-10', 'F-15', 'Trust', 'Cont', 'Stead', 'Cohes', 'Pearson']
values = [
    results['DE_proxy'],
    results['FS_proxy_10'],
    results['FS_embedding_15'],
    results['Trustworthiness'],
    results['Continuity'],
    results['Steadiness'],
    results['Cohesiveness'],
    results['Pearson_corr'],
]
colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71', '#f39c12', '#1abc9c', '#e67e22', '#34495e']
ax2.bar(metrics_list, values, color=colors, alpha=0.7, width=0.7)
ax2.set_title('All Evaluation Metrics', fontsize=12, fontweight='bold')
ax2.set_ylabel('Score')
ax2.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
ax2.grid(True, alpha=0.3, axis='y')
ax2.tick_params(axis='x', rotation=45)

for i, v in enumerate(values):
    ax2.text(i, v + max(values) * 0.02, f'{v:.4f}',
             ha='center', va='bottom', fontweight='bold', fontsize=8)

plt.tight_layout()
plt.savefig('german_credit_od_tsne_evaluation.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved to: german_credit_od_tsne_evaluation.png")

# ─── 9. Save Results ─────────────────────────────────────────────────────────
with open('german_credit_od_tsne_results.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("GERMAN CREDIT - OD METHOD (t-SNE) EVALUATION\n")
    f.write("=" * 80 + "\n\n")
    f.write("+" + "-"*30 + "+" + "-"*12 + "+\n")
    f.write("| {:<28} | {:<10} |\n".format("Metric", "Value"))
    f.write("+" + "="*30 + "+" + "="*12 + "+\n")
    for _, row in df.iterrows():
        f.write("| {:<28} | {:<10} |\n".format(row['Metric'], row['Value']))
    f.write("+" + "-"*30 + "+" + "-"*12 + "+\n")

df.to_csv('german_credit_od_tsne_results.csv', index=False)

print("\nResults saved to:")
print("  - german_credit_od_tsne_results.txt")
print("  - german_credit_od_tsne_results.csv")
print("  - german_credit_od_tsne_evaluation.png")
print("\n" + "=" * 80)
print("Done!")
print("=" * 80)
