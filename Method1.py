"""
Method 1: SENSE Framework - Anchor-Based Dimensionality Reduction on MNIST
===========================================================================
Pipeline:
  1. Load MNIST data
  2. Select anchors (KMeans-based)
  3. Compute distance matrices (GPU-accelerated)
  4. Apply Random Projection or PCA to non-anchor points
  5. Refine to 2D with t-SNE
  6. Evaluate using F-Score against ground truth t-SNE embedding

Methods compared: Random Projection vs PCA
Dataset: MNIST (784-dimensional)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from openTSNE import TSNE as FastTSNE
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    NUM_ANCHORS = 783
    NORMALIZE_METHOD = 'standard'
    ANCHOR_PERCENTAGES = [10, 20, 30]
    ANCHOR_METHOD = 'kmeans'

    @staticmethod
    def calculate_total_data_size(anchor_pct):
        return int(783 * 100 / anchor_pct)

    D_L_VALUES = np.logspace(np.log10(30), np.log10(782), 10).astype(int)
    K_VALUES = [10, 15]
    METHODS = ['random_projection', 'pca']

    RP_SEED = 42
    RP_ORTHOGONALIZE = True
    RP_SPARSE = False
    RIDGE_LAMBDA = 1e-6

    TSNE_PERPLEXITY = 30
    TSNE_N_ITER = 1000
    TSNE_RANDOM_STATE = 42
    USE_FAST_TSNE = True

    FIGURE_SIZE = (15, 10)
    DPI = 100


config = Config()

print("\n" + "=" * 80)
print("CONFIGURATION")
print("=" * 80)
print(f"Fixed number of anchors: {config.NUM_ANCHORS}")
print("\nData sizes for each anchor percentage:")
for pct in config.ANCHOR_PERCENTAGES:
    total_size = config.calculate_total_data_size(pct)
    non_anchor = total_size - config.NUM_ANCHORS
    print(f"  {pct}% anchors: Total={total_size}, Anchors={config.NUM_ANCHORS}, Non-Anchors={non_anchor}")
print("=" * 80 + "\n")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_preprocess_mnist(n_samples, normalize='standard'):
    """Load and preprocess MNIST dataset."""
    print(f"Loading {n_samples} MNIST samples...")
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    X = mnist.data.values[:n_samples] if hasattr(mnist.data, 'values') else mnist.data[:n_samples]
    y = mnist.target.values[:n_samples] if hasattr(mnist.target, 'values') else mnist.target[:n_samples]

    X = X.astype(np.float32)

    if normalize == 'standard':
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif normalize == 'minmax':
        X = X / 255.0

    print(f"Data shape: {X.shape}, Labels shape: {y.shape}")
    return X, y


# =============================================================================
# ANCHOR SELECTION
# =============================================================================

def select_anchors(X, m, method='kmeans', seed=42):
    """
    Select m anchor points from the dataset.

    Parameters
    ----------
    X : np.ndarray of shape (N, d)
    m : int, number of anchors
    method : str, 'kmeans' or 'random'
    seed : int, random seed

    Returns
    -------
    X_A : anchor points
    X_NA : non-anchor points
    anchor_indices : indices of anchor points
    non_anchor_indices : indices of non-anchor points
    """
    print(f"Selecting {m} anchors using {method} method...")
    n = len(X)

    if method == 'kmeans':
        kmeans = MiniBatchKMeans(n_clusters=m, random_state=seed, batch_size=1024, max_iter=100)
        kmeans.fit(X)
        anchor_indices = []
        for center in kmeans.cluster_centers_:
            distances = np.sum((X - center) ** 2, axis=1)
            anchor_indices.append(np.argmin(distances))
        anchor_indices = np.array(anchor_indices)
    else:
        np.random.seed(seed)
        anchor_indices = np.random.choice(n, m, replace=False)

    non_anchor_indices = np.setdiff1d(np.arange(n), anchor_indices)

    X_A = X[anchor_indices]
    X_NA = X[non_anchor_indices]

    print(f"Anchors: {len(anchor_indices)}, Non-Anchors: {len(non_anchor_indices)}")

    return X_A, X_NA, anchor_indices, non_anchor_indices


# =============================================================================
# DISTANCE MATRIX COMPUTATION (GPU-ACCELERATED)
# =============================================================================

def compute_distance_matrices_gpu(X_A, X_NA, batch_size=1024):
    """
    Compute pairwise distance matrices using GPU acceleration.

    Returns
    -------
    E : (m, m) anchor-to-anchor distance matrix
    F : (n, m) non-anchor-to-anchor distance matrix
    """
    print("Computing distance matrices on GPU...")

    X_A_torch = torch.from_numpy(X_A).to(device)
    X_NA_torch = torch.from_numpy(X_NA).to(device)

    E = torch.cdist(X_A_torch, X_A_torch, p=2).cpu().numpy()

    F_list = []
    for i in range(0, len(X_NA), batch_size):
        batch = X_NA_torch[i:i + batch_size]
        F_batch = torch.cdist(batch, X_A_torch, p=2).cpu().numpy()
        F_list.append(F_batch)

    F = np.vstack(F_list)

    print(f"Distance matrices computed - E: {E.shape}, F: {F.shape}")
    return E, F


# =============================================================================
# t-SNE EMBEDDING
# =============================================================================

def apply_tsne_to_embedding(embedding, config, desc="t-SNE"):
    """Apply t-SNE to reduce embedding to 2D."""
    print(f"Applying {desc}...")

    if config.USE_FAST_TSNE:
        try:
            tsne = FastTSNE(
                n_components=2,
                perplexity=config.TSNE_PERPLEXITY,
                n_iter=config.TSNE_N_ITER,
                random_state=config.TSNE_RANDOM_STATE,
                n_jobs=-1,
                verbose=False
            )
            embedding_2d = tsne.fit(embedding)
        except Exception:
            print("FastTSNE failed, using sklearn t-SNE...")
            tsne = TSNE(
                n_components=2,
                perplexity=config.TSNE_PERPLEXITY,
                n_iter=config.TSNE_N_ITER,
                random_state=config.TSNE_RANDOM_STATE,
                verbose=0
            )
            embedding_2d = tsne.fit_transform(embedding)
    else:
        tsne = TSNE(
            n_components=2,
            perplexity=config.TSNE_PERPLEXITY,
            n_iter=config.TSNE_N_ITER,
            random_state=config.TSNE_RANDOM_STATE,
            verbose=0
        )
        embedding_2d = tsne.fit_transform(embedding)

    print(f"2D embedding shape: {embedding_2d.shape}")
    return embedding_2d


# =============================================================================
# EVALUATION METRIC: F-SCORE
# =============================================================================

def calculate_f_score(embedding1, embedding2, k):
    """
    Calculate F-score between two embeddings based on k-nearest neighbor overlap.

    Parameters
    ----------
    embedding1 : np.ndarray, ground truth 2D embedding
    embedding2 : np.ndarray, predicted 2D embedding
    k : int, number of neighbors

    Returns
    -------
    float : F-score in [0, 1]
    """
    nn1 = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(embedding1)
    indices1 = nn1.kneighbors(embedding1, return_distance=False)[:, 1:]

    nn2 = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1).fit(embedding2)
    indices2 = nn2.kneighbors(embedding2, return_distance=False)[:, 1:]

    tp, fp, fn = 0, 0, 0
    for i in range(len(embedding1)):
        neighbors1 = set(indices1[i])
        neighbors2 = set(indices2[i])
        tp += len(neighbors1.intersection(neighbors2))
        fp += len(neighbors2.difference(neighbors1))
        fn += len(neighbors1.difference(neighbors2))

    f_score = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    return f_score


# =============================================================================
# DIMENSIONALITY REDUCTION METHODS
# =============================================================================

def random_projection_method(X_A, F, d_l, seed=42, orthogonalize=True, sparse=False, ridge_lambda=1e-6):
    """
    Random Projection-based coordinate reconstruction.

    Projects anchors into a d_l-dimensional space using a random matrix,
    then solves for non-anchor coordinates via trilateration using distances F.

    Parameters
    ----------
    X_A : (m, d_h) anchor points in high-dim space
    F : (n, m) distance matrix from non-anchors to anchors
    d_l : int, target intermediate dimension
    seed : int, random seed
    orthogonalize : bool, use QR orthogonalization
    sparse : bool, use sparse random projection
    ridge_lambda : float, regularization for linear system

    Returns
    -------
    Y_final : (n, d_l) non-anchor embeddings
    """
    d_h = X_A.shape[1]
    np.random.seed(seed)

    if sparse:
        choices = [np.sqrt(3), 0, -np.sqrt(3)]
        probs = [1 / 6, 2 / 3, 1 / 6]
        R = np.random.choice(choices, size=(d_h, d_l), p=probs).astype(np.float32)
    else:
        R = np.random.randn(d_h, d_l).astype(np.float32) / np.sqrt(d_l)

    if orthogonalize:
        Q, _ = np.linalg.qr(R)
        R = Q

    # Project anchors into low-dim space
    A_low = X_A @ R

    # Trilateration: solve for non-anchor positions
    a_0 = A_low[0:1]
    A_sys = A_low[1:] - a_0

    ATA = A_sys.T @ A_sys
    ATA_reg = ATA + ridge_lambda * np.eye(ATA.shape[0])
    A_pinv = np.linalg.solve(ATA_reg, A_sys.T)

    r_sq = F ** 2
    A_sys_norm_sq = np.sum(A_sys ** 2, axis=1)
    B_matrix = 0.5 * (r_sq[:, 0:1] - r_sq[:, 1:] + A_sys_norm_sq)

    Y_rel = B_matrix @ A_pinv.T
    Y_final = Y_rel + a_0

    return Y_final


def pca_method(X_A, F, d_l, ridge_lambda=1e-6):
    """
    PCA-based coordinate reconstruction.

    Projects anchors using PCA, then reconstructs non-anchor positions
    via trilateration using the anchor-to-non-anchor distances.

    Parameters
    ----------
    X_A : (m, d_h) anchor points
    F : (n, m) distance matrix from non-anchors to anchors
    d_l : int, number of PCA components
    ridge_lambda : float, regularization

    Returns
    -------
    Y_final : (n, d_l) non-anchor embeddings
    """
    pca = PCA(n_components=d_l)
    A_low = pca.fit_transform(X_A).astype(np.float32)

    a_0 = A_low[0:1]
    A_sys = A_low[1:] - a_0

    ATA = A_sys.T @ A_sys
    ATA_reg = ATA + ridge_lambda * np.eye(ATA.shape[0])
    A_pinv = np.linalg.solve(ATA_reg, A_sys.T)

    r_sq = F ** 2
    A_sys_norm_sq = np.sum(A_sys ** 2, axis=1)
    B_matrix = 0.5 * (r_sq[:, 0:1] - r_sq[:, 1:] + A_sys_norm_sq)

    Y_rel = B_matrix @ A_pinv.T
    Y_final = Y_rel + a_0

    return Y_final


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================

def run_experiment(X, y, d_l, method_name, ground_truth_2d, config):
    """
    Run a single experiment for a given method and intermediate dimension d_l.

    Returns
    -------
    Y_final_2d : (n, 2) final 2D embedding of non-anchor points
    total_time : float, total wall-clock time in seconds
    """
    X_A, X_NA, anchor_idx, non_anchor_idx = select_anchors(
        X, config.NUM_ANCHORS, method=config.ANCHOR_METHOD, seed=config.RP_SEED
    )

    E, F = compute_distance_matrices_gpu(X_A, X_NA, batch_size=1024)
    del X_NA

    start_time = time.time()

    if method_name == 'random_projection':
        Y_intermediate = random_projection_method(
            X_A, F, d_l,
            seed=config.RP_SEED,
            orthogonalize=config.RP_ORTHOGONALIZE,
            sparse=config.RP_SPARSE,
            ridge_lambda=config.RIDGE_LAMBDA
        )
    elif method_name == 'pca':
        Y_intermediate = pca_method(X_A, F, d_l, ridge_lambda=config.RIDGE_LAMBDA)

    method_time = time.time() - start_time

    tsne_start = time.time()
    Y_final_2d = apply_tsne_to_embedding(
        Y_intermediate, config,
        desc=f"t-SNE on {method_name} embedding"
    )
    tsne_time = time.time() - tsne_start
    total_time = method_time + tsne_time

    print(f"Method time: {method_time:.2f}s, t-SNE time: {tsne_time:.2f}s, Total: {total_time:.2f}s")
    return Y_final_2d, total_time


def run_full_experiments(config):
    """Run all experiments across anchor percentages, methods, and d_l values."""
    results = {}

    for anchor_pct in config.ANCHOR_PERCENTAGES:
        total_size = config.calculate_total_data_size(anchor_pct)
        non_anchor_count = total_size - config.NUM_ANCHORS

        print(f"\n{'=' * 80}")
        print(f"RUNNING EXPERIMENTS FOR {anchor_pct}% ANCHORS")
        print(f"Total Data: {total_size}, Anchors: {config.NUM_ANCHORS}, Non-Anchors: {non_anchor_count}")
        print(f"{'=' * 80}\n")

        X, y = load_and_preprocess_mnist(n_samples=total_size, normalize=config.NORMALIZE_METHOD)

        X_A, X_NA_true, anchor_idx, non_anchor_idx = select_anchors(
            X, config.NUM_ANCHORS, method=config.ANCHOR_METHOD, seed=config.RP_SEED
        )

        print("\n" + "=" * 60)
        print("CREATING GROUND TRUTH 2D EMBEDDING (t-SNE on original data)")
        print("=" * 60)
        ground_truth_2d = apply_tsne_to_embedding(X_NA_true, config, desc="Ground Truth t-SNE")
        print("=" * 60 + "\n")

        results[anchor_pct] = {'ground_truth_2d': ground_truth_2d}

        for method_name in config.METHODS:
            print(f"\n{'─' * 60}")
            print(f"Method: {method_name.upper()}")
            print(f"{'─' * 60}")
            results[anchor_pct][method_name] = {
                'f_scores_k10': [],
                'f_scores_k15': [],
                'times': []
            }

            for d_l in tqdm(config.D_L_VALUES, desc=f"d_l values for {method_name}"):
                Y_final_2d, total_time = run_experiment(X, y, d_l, method_name, ground_truth_2d, config)

                f_score_k10 = calculate_f_score(ground_truth_2d, Y_final_2d, k=10)
                f_score_k15 = calculate_f_score(ground_truth_2d, Y_final_2d, k=15)

                results[anchor_pct][method_name]['f_scores_k10'].append(f_score_k10)
                results[anchor_pct][method_name]['f_scores_k15'].append(f_score_k15)
                results[anchor_pct][method_name]['times'].append(total_time)

                print(f"  d_l={d_l:3d}: F(k=10)={f_score_k10:.4f}, "
                      f"F(k=15)={f_score_k15:.4f}, Time={total_time:.2f}s")

    return results


# =============================================================================
# PLOTTING & SUMMARY
# =============================================================================

def plot_results(results, config):
    """Plot F-score vs intermediate dimension for each anchor percentage."""
    n_anchor_pcts = len(config.ANCHOR_PERCENTAGES)

    for k_val in config.K_VALUES:
        fig, axes = plt.subplots(1, n_anchor_pcts, figsize=(6 * n_anchor_pcts, 5))
        if n_anchor_pcts == 1:
            axes = [axes]

        for idx, anchor_pct in enumerate(config.ANCHOR_PERCENTAGES):
            ax = axes[idx]
            total_size = config.calculate_total_data_size(anchor_pct)

            for method_name in config.METHODS:
                f_scores = results[anchor_pct][method_name][f'f_scores_k{k_val}']
                ax.plot(config.D_L_VALUES, f_scores, marker='o',
                        label=method_name.upper(), linewidth=2, markersize=6)

            ax.set_xlabel('Intermediate Dimension (d_l)', fontsize=12)
            ax.set_ylabel(f'F-Score (k={k_val})', fontsize=12)
            ax.set_title(f'{anchor_pct}% Anchors (N={total_size}, m={config.NUM_ANCHORS}) - k={k_val}',
                         fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log')
            ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(f'f_score_comparison_k{k_val}_2d.png', dpi=config.DPI, bbox_inches='tight')
        plt.show()

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS (F-Scores on 2D t-SNE Embeddings)")
    print("=" * 80)
    for anchor_pct in config.ANCHOR_PERCENTAGES:
        total_size = config.calculate_total_data_size(anchor_pct)
        print(f"\n{anchor_pct}% Anchors (Total N={total_size}):")
        for method_name in config.METHODS:
            avg_time = np.mean(results[anchor_pct][method_name]['times'])
            best_f10 = np.max(results[anchor_pct][method_name]['f_scores_k10'])
            best_f15 = np.max(results[anchor_pct][method_name]['f_scores_k15'])
            mean_f10 = np.mean(results[anchor_pct][method_name]['f_scores_k10'])
            mean_f15 = np.mean(results[anchor_pct][method_name]['f_scores_k15'])
            print(f"  {method_name.upper():20s} - Avg Time: {avg_time:6.2f}s")
            print(f"                        Best F(k=10): {best_f10:.4f}, Mean F(k=10): {mean_f10:.4f}")
            print(f"                        Best F(k=15): {best_f15:.4f}, Mean F(k=15): {mean_f15:.4f}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SECURE DIMENSIONALITY REDUCTION EXPERIMENTS")
    print("Phase 4: Manifold Refinement to 2D with t-SNE")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Fixed anchors: {config.NUM_ANCHORS}")
    print(f"  Anchor percentages: {config.ANCHOR_PERCENTAGES}")
    for pct in config.ANCHOR_PERCENTAGES:
        total = config.calculate_total_data_size(pct)
        print(f"    {pct}% → Total data: {total}")
    print(f"  Methods: {config.METHODS}")
    print(f"  d_l range: {config.D_L_VALUES[0]} to {config.D_L_VALUES[-1]} ({len(config.D_L_VALUES)} points)")
    print(f"  k values: {config.K_VALUES}")
    print(f"  t-SNE parameters: perplexity={config.TSNE_PERPLEXITY}, n_iter={config.TSNE_N_ITER}")
    print(f"  Device: {device}")
    print("\nPipeline:")
    print("  1. Ground Truth: X_NA (784D) → t-SNE → 2D")
    print("  2. Methods: X_NA → Method(d_l) → t-SNE → 2D")
    print("  3. F-Score: Compare two 2D embeddings")
    print("=" * 80)

    results = run_full_experiments(config)
    plot_results(results, config)

    print("\n" + "=" * 80)
    print("EXPERIMENTS COMPLETED!")
    print("=" * 80)
