"""
Method 2: SENSE Framework - OD (Orthogonal Distance) Method on German Credit Dataset
=====================================================================================
Pipeline:
  1. Load German Credit dataset from OpenML
  2. Encode categorical features with OrdinalEncoder
  3. Select anchors randomly (m = d_h - 1)
  4. Compute OD proxy features using L_inf metric for all anchor pairs
  5. Compute distance matrices (proxy vs. original)
  6. Generate 2D embedding with PaCMAP
  7. Evaluate with: Distance Error, F-Score, Trustworthiness, Continuity,
     Steadiness, Cohesiveness, Pearson Correlation

Key idea: For each non-anchor point u and each pair of anchors (A, B),
compute the foot-of-perpendicular distance (OD) as a privacy-preserving
proxy feature. The set of all pairwise OD values forms the proxy space.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from itertools import combinations
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import pearsonr
import pandas as pd

try:
    import pacmap
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pacmap", "--break-system-packages"])
    import pacmap


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_distance_error(G_proxy, G_original):
    """
    Relative Frobenius norm distance error between proxy and original distance matrices.

    Lower is better. A value of 0 means perfect distance preservation.
    """
    return np.linalg.norm(G_proxy - G_original, 'fro') / np.linalg.norm(G_original, 'fro')


def calculate_fscore_neighbors(G_pred, G_true, k):
    """
    F-score based on k-nearest neighbor overlap between two distance matrices.

    Parameters
    ----------
    G_pred : (n, n) predicted distance matrix
    G_true : (n, n) true distance matrix
    k : int, number of neighbors

    Returns
    -------
    float : F-score in [0, 1]
    """
    n = G_pred.shape[0]
    neighbors_true = np.argsort(G_true, axis=1)[:, 1:k + 1]
    neighbors_pred = np.argsort(G_pred, axis=1)[:, 1:k + 1]

    tp_total, fp_total, fn_total = 0, 0, 0

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

    Measures linear relationship between true and reconstructed distances.
    """
    rows, cols = np.triu_indices(G_true.shape[0], k=1)
    vec_true = G_true[rows, cols]
    vec_reconstructed = G_reconstructed[rows, cols]
    corr, _ = pearsonr(vec_true, vec_reconstructed)
    return corr


def calculate_trustworthiness(X_original, Y_embedding, k=10):
    """
    Trustworthiness: measures if new neighbors in the embedding were also
    nearby in the original space. High means no false neighbors introduced.

    Parameters
    ----------
    X_original : distance matrix (n, n) or raw features (n, d)
    Y_embedding : (n, 2) embedding
    k : int, neighborhood size

    Returns
    -------
    float : trustworthiness in [0, 1]
    """
    if len(X_original.shape) == 2 and X_original.shape[0] == X_original.shape[1]:
        D_original = X_original
    else:
        D_original = cdist(X_original, X_original, metric='euclidean')

    D_embedding = cdist(Y_embedding, Y_embedding, metric='euclidean')
    N = D_original.shape[0]
    nn_original = np.argsort(D_original, axis=1)[:, 1:k + 1]
    nn_embedding = np.argsort(D_embedding, axis=1)[:, 1:k + 1]

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
    Continuity: measures if original neighbors are preserved in the embedding.
    High means original neighborhoods are not torn apart.

    Parameters
    ----------
    X_original : distance matrix (n, n) or raw features (n, d)
    Y_embedding : (n, 2) embedding
    k : int, neighborhood size

    Returns
    -------
    float : continuity in [0, 1]
    """
    if len(X_original.shape) == 2 and X_original.shape[0] == X_original.shape[1]:
        D_original = X_original
    else:
        D_original = cdist(X_original, X_original, metric='euclidean')

    D_embedding = cdist(Y_embedding, Y_embedding, metric='euclidean')
    N = D_original.shape[0]
    nn_original = np.argsort(D_original, axis=1)[:, 1:k + 1]
    nn_embedding = np.argsort(D_embedding, axis=1)[:, 1:k + 1]

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

    Steadiness: form cluster in embedding → check how valid it is in original space.
    Cohesiveness: form cluster in original space → check how well it's preserved in embedding.

    Parameters
    ----------
    X_original : distance matrix (n, n) or raw features (n, d)
    Y_embedding : (n, 2) embedding
    n_iterations : int, number of random seed points to sample
    k : int or None, cluster size (defaults to max(10, 5% of N))

    Returns
    -------
    (steadiness, cohesiveness) : both floats in [0, 1]
    """
    if len(X_original.shape) == 2 and X_original.shape[0] == X_original.shape[1]:
        D_original = X_original
    else:
        D_original = cdist(X_original, X_original, metric='euclidean')

    D_embedding = cdist(Y_embedding, Y_embedding, metric='euclidean')
    N = D_original.shape[0]

    if k is None:
        k = max(10, int(N * 0.05))

    np.random.seed(42)
    steadiness_scores = []
    cohesiveness_scores = []

    # Steadiness: cluster in embedding → validate in original
    for _ in range(n_iterations):
        center_idx = np.random.randint(0, N)
        cluster_emb = np.argsort(D_embedding[center_idx])[1:k + 1]
        true_orig = np.argsort(D_original[center_idx])[1:k + 1]
        intersect = np.intersect1d(cluster_emb, true_orig)
        steadiness_scores.append(len(intersect) / k)

    # Cohesiveness: cluster in original → check preservation in embedding
    for _ in range(n_iterations):
        center_idx = np.random.randint(0, N)
        cluster_orig = np.argsort(D_original[center_idx])[1:k + 1]
        true_emb = np.argsort(D_embedding[center_idx])[1:k + 1]
        intersect = np.intersect1d(cluster_orig, true_emb)
        cohesiveness_scores.append(len(intersect) / k)

    return np.mean(steadiness_scores), np.mean(cohesiveness_scores)


# =============================================================================
# OD PROXY FEATURE COMPUTATION
# =============================================================================

def compute_od_proxy_features(X_A, X_NA, anchor_pairs):
    """
    Compute Orthogonal Distance (OD) proxy features using the L_inf metric.

    For each non-anchor point u and each anchor pair (A, B):
      1. Project u onto the line AB: D = A + k*(B-A), k = (b^2 + c^2 - a^2) / (2c^2)
      2. OD = L_inf norm of the foot-of-perpendicular D

    This encoding is privacy-preserving because the original coordinates
    of u cannot be recovered from OD values alone.

    Parameters
    ----------
    X_A : (m, d) anchor points
    X_NA : (n, d) non-anchor points
    anchor_pairs : list of (i, j) tuples

    Returns
    -------
    X_proxy : (n, L) proxy feature matrix where L = len(anchor_pairs)
    """
    n = X_NA.shape[0]
    L = len(anchor_pairs)
    X_proxy = np.zeros((n, L))

    print(f"Computing OD Features with L_inf metric for {n} points × {L} anchor pairs...")

    for point_idx in range(n):
        u = X_NA[point_idx]

        for pair_idx, (i, j) in enumerate(anchor_pairs):
            A = X_A[i]
            B = X_A[j]

            c = np.linalg.norm(B - A, ord=np.inf)
            b = np.linalg.norm(u - A, ord=np.inf)
            a = np.linalg.norm(u - B, ord=np.inf)

            if c != 0:
                AD = (b ** 2 + c ** 2 - a ** 2) / (2 * c)
                k = AD / c
            else:
                k = 0

            D = A + k * (B - A)
            OD = np.linalg.norm(D, ord=np.inf)
            X_proxy[point_idx, pair_idx] = OD

    return X_proxy


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_results(Y_embedding, y_labels, results):
    """Generate visualization: 2D scatter + metrics bar chart."""
    y_numeric = np.array([1 if label == 'good' else 0 for label in y_labels])

    fig = plt.figure(figsize=(16, 5))

    ax1 = plt.subplot(1, 2, 1)
    scatter = ax1.scatter(Y_embedding[:, 0], Y_embedding[:, 1], c=y_numeric,
                          cmap='viridis', alpha=0.6, s=30)
    ax1.set_title('OD with L_inf - 2D Embedding (PaCMAP)', fontsize=12, fontweight='bold')
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
        results['Pearson_corr']
    ]
    colors = ['#e74c3c', '#3498db', '#9b59b6', '#2ecc71', '#f39c12', '#1abc9c', '#e67e22', '#34495e']
    ax2.bar(metrics_list, values, color=colors, alpha=0.7, width=0.7)
    ax2.set_title('All Evaluation Metrics', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)

    max_val = max(values) if max(values) > 0 else 1
    for i, v in enumerate(values):
        ax2.text(i, v + max_val * 0.02, f'{v:.4f}',
                 ha='center', va='bottom', fontweight='bold', fontsize=8)

    plt.tight_layout()
    plt.savefig('german_credit_od_evaluation.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to: german_credit_od_evaluation.png")
    plt.show()


def save_results(results):
    """Save evaluation results to text and CSV files."""
    metrics_data = {
        'Metric': [
            'Distance Error',
            'F-score (proxy, k=10)',
            'F-score (embedding, k=15)',
            'Trustworthiness',
            'Continuity',
            'Steadiness',
            'Cohesiveness',
            'Pearson Correlation'
        ],
        'Value': [
            f"{results['DE_proxy']:.6f}",
            f"{results['FS_proxy_10']:.6f}",
            f"{results['FS_embedding_15']:.6f}",
            f"{results['Trustworthiness']:.6f}",
            f"{results['Continuity']:.6f}",
            f"{results['Steadiness']:.6f}",
            f"{results['Cohesiveness']:.6f}",
            f"{results['Pearson_corr']:.6f}"
        ]
    }
    df = pd.DataFrame(metrics_data)

    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)
    print()
    print("+" + "-" * 30 + "+" + "-" * 12 + "+")
    print("| {:<28} | {:<10} |".format("Metric", "Value"))
    print("+" + "=" * 30 + "+" + "=" * 12 + "+")
    for _, row in df.iterrows():
        print("| {:<28} | {:<10} |".format(row['Metric'], row['Value']))
    print("+" + "-" * 30 + "+" + "-" * 12 + "+")

    with open('german_credit_od_results.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GERMAN CREDIT - OD METHOD EVALUATION\n")
        f.write("=" * 80 + "\n\n")
        f.write("+" + "-" * 30 + "+" + "-" * 12 + "+\n")
        f.write("| {:<28} | {:<10} |\n".format("Metric", "Value"))
        f.write("+" + "=" * 30 + "+" + "=" * 12 + "+\n")
        for _, row in df.iterrows():
            f.write("| {:<28} | {:<10} |\n".format(row['Metric'], row['Value']))
        f.write("+" + "-" * 30 + "+" + "-" * 12 + "+\n")

    df.to_csv('german_credit_od_results.csv', index=False)

    print("\nResults saved to:")
    print("  - german_credit_od_results.txt")
    print("  - german_credit_od_results.csv")
    print("  - german_credit_od_evaluation.png")

    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SENSE Framework - German Credit Dataset (OD Method)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Load & Encode Data
    # -------------------------------------------------------------------------
    print("\nLoading German Credit Dataset from OpenML...")
    german_credit = fetch_openml('credit-g', version=1, parser='auto')

    X_full = german_credit.data
    y_full = german_credit.target.values

    categorical_cols = X_full.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X_full.select_dtypes(include=[np.number]).columns

    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Numerical columns: {len(numerical_cols)}")

    X_encoded = X_full.copy()
    if len(categorical_cols) > 0:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_encoded[categorical_cols] = encoder.fit_transform(X_full[categorical_cols])

    X_all = X_encoded.values.astype(float)
    X = X_all
    y = y_full
    N, d_h = X.shape

    # -------------------------------------------------------------------------
    # 2. Select Anchors
    # -------------------------------------------------------------------------
    m = d_h - 1     # number of anchors
    n = N - m       # number of private (non-anchor) points

    np.random.seed(42)
    anchor_indices = np.random.choice(N, size=m, replace=False)
    non_anchor_mask = np.ones(N, dtype=bool)
    non_anchor_mask[anchor_indices] = False
    non_anchor_indices = np.where(non_anchor_mask)[0]

    X_A = X[anchor_indices]
    X_NA = X[non_anchor_indices]
    y_NA = y[non_anchor_indices]

    print(f"\nDataset Shape: {X.shape}")
    print(f"Dimension (d_h): {d_h}")
    print(f"Number of Anchors (m = d_h - 1): {m}")
    print(f"Number of Private Points (n): {n}")

    # -------------------------------------------------------------------------
    # 3. Compute OD Proxy Features
    # -------------------------------------------------------------------------
    anchor_pairs = list(combinations(range(m), 2))
    L = len(anchor_pairs)
    print(f"Number of Anchor Pairs: {L}")
    print(f"Proxy Space Dimensions: {L}")

    X_proxy_OD_Linf = compute_od_proxy_features(X_A, X_NA, anchor_pairs)

    # -------------------------------------------------------------------------
    # 4. Compute Distance Matrices
    # -------------------------------------------------------------------------
    print("\nComputing distance matrices...")
    G_original_L2 = cdist(X_NA, X_NA, metric='euclidean')
    G_proxy_OD_Linf = cdist(X_proxy_OD_Linf, X_proxy_OD_Linf, metric='chebyshev')

    # -------------------------------------------------------------------------
    # 5. Generate 2D Embedding with PaCMAP
    # -------------------------------------------------------------------------
    print("Generating 2D embedding using PaCMAP...")
    embedding_OD = pacmap.PaCMAP(n_components=2, n_neighbors=10, random_state=42)
    Y_OD_Linf = embedding_OD.fit_transform(X_proxy_OD_Linf)
    G_Y_OD = cdist(Y_OD_Linf, Y_OD_Linf, metric='euclidean')

    # -------------------------------------------------------------------------
    # 6. Evaluate
    # -------------------------------------------------------------------------
    print("\nCalculating evaluation metrics...")

    results = {
        'DE_proxy':         calculate_distance_error(G_proxy_OD_Linf, G_original_L2),
        'FS_proxy_10':      calculate_fscore_neighbors(G_proxy_OD_Linf, G_original_L2, k=10),
        'FS_embedding_15':  calculate_fscore_neighbors(G_original_L2, G_Y_OD, k=15),
        'Trustworthiness':  calculate_trustworthiness(G_original_L2, Y_OD_Linf, k=10),
        'Continuity':       calculate_continuity(G_original_L2, Y_OD_Linf, k=10),
        'Pearson_corr':     calculate_pearson_correlation(G_original_L2, G_proxy_OD_Linf)
    }

    steadiness, cohesiveness = calculate_steadiness_cohesiveness_exact(G_original_L2, Y_OD_Linf)
    results['Steadiness'] = steadiness
    results['Cohesiveness'] = cohesiveness

    # -------------------------------------------------------------------------
    # 7. Save & Visualize
    # -------------------------------------------------------------------------
    save_results(results)
    plot_results(Y_OD_Linf, y_NA, results)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
