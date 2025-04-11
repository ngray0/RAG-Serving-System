
import numpy as np
import cupy as cp
import time # For timing within the function
def pairwise_l2_squared_numpy(X_np, C_np):
    """
    Computes pairwise **squared** L2 distances using NumPy.
    X_np: (N, D) data points OR (Q, D) query points (NumPy array)
    C_np: (K, D) centroids OR (N, D) database points (NumPy array)
    Returns: (N|Q, K|N) NumPy array of SQUARED distances.
    """
    # Ensure inputs are NumPy arrays and float32
    if not isinstance(X_np, np.ndarray): X_np = np.asarray(X_np, dtype=np.float32)
    elif X_np.dtype != np.float32: X_np = X_np.astype(np.float32)
    if not isinstance(C_np, np.ndarray): C_np = np.asarray(C_np, dtype=np.float32)
    elif C_np.dtype != np.float32: C_np = C_np.astype(np.float32)

    # ||x - c||^2 = ||x||^2 - 2<x, c> + ||c||^2
    # Use einsum for potentially better memory efficiency/speed for sum of squares
    X_norm_sq = np.einsum('ij,ij->i', X_np, X_np)[:, np.newaxis] # Shape (N|Q, 1)
    C_norm_sq = np.einsum('ij,ij->i', C_np, C_np)[np.newaxis, :] # Shape (1, K|N)

    # Dot product using matrix multiplication
    dot_products = np.dot(X_np, C_np.T) # Shape (N|Q, K|N)

    # Broadcasting: (N|Q, 1) - 2*(N|Q, K|N) + (1, K|N) -> (N|Q, K|N)
    dist_sq = X_norm_sq - 2 * dot_products + C_norm_sq
    # Clamp to avoid small numerical negatives
    return np.maximum(0.0, dist_sq)


def our_kmeans_cpu(N_A, D, A_np, K, max_iters=100, tol=1e-4):
    """
    Performs K-means clustering entirely using NumPy on the CPU.

    Args:
        N_A (int): Number of data points.
        D (int): Dimensionality.
        A_np (np.ndarray): Data points (N_A, D) on CPU (as NumPy ndarray).
        K (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance for centroid movement convergence check.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - centroids_np (np.ndarray): Final centroids (K, D).
            - assignments_np (np.ndarray): Final cluster assignment (N_A,).
    """
    # --- Input Validation ---
    if not isinstance(A_np, np.ndarray):
        raise TypeError("Input data 'A_np' must be a NumPy ndarray.")
    actual_N, actual_D = A_np.shape
    if actual_N != N_A or actual_D != D:
        print(f"Warning: N_A/D ({N_A}/{D}) mismatch with A_np shape {A_np.shape}. Using shape from A_np.")
        N_A, D = actual_N, actual_D
    if not (K > 0):
        raise ValueError("K must be positive.")
    if K > N_A:
         print(f"Warning: K ({K}) > N_A ({N_A}). Setting K = N_A.")
         K = N_A
    if A_np.dtype != np.float32:
        print(f"Warning: Input data dtype is {A_np.dtype}. Converting to float32.")
        A_np = A_np.astype(np.float32)

    print(f"Running K-Means (Pure NumPy CPU): N={N_A}, D={D}, K={K}")
    start_time_total = time.time()

    # Handle K=0 edge case
    if K == 0:
         print("Warning: K=0 requested for KMeans. Returning empty centroids.")
         return np.empty((0, D), dtype=np.float32), np.empty((N_A,), dtype=np.int32)

    # --- Initialization ---
    # Randomly choose K unique points from A_np as initial centroids
    initial_indices = np.random.choice(N_A, K, replace=False)
    centroids_np = A_np[initial_indices].copy()
    assignments_np = np.empty(N_A, dtype=np.int32)
    old_centroids_np = np.empty_like(centroids_np)

    # --- Iteration Loop ---
    for i in range(max_iters):
        # Store old centroids for convergence check
        old_centroids_np = centroids_np.copy()

        # --- 1. Assignment Step (NumPy) ---
        # Calculate distances from all points to all centroids
        all_dist_sq_np = pairwise_l2_squared_numpy(A_np, centroids_np) # Shape (N_A, K)
        # Assign each point to the closest centroid
        assignments_np = np.argmin(all_dist_sq_np, axis=1).astype(np.int32) # Shape (N_A,)

        # --- 2. Update Step (NumPy - Optimized) ---
        new_sums_np = np.zeros((K, D), dtype=np.float32)
        # Use np.add.at for efficient scatter-add like summing
        np.add.at(new_sums_np, assignments_np, A_np)

        # Count points in each cluster efficiently
        cluster_counts_np = np.bincount(assignments_np, minlength=K).astype(np.float32)

        # Avoid division by zero for empty clusters
        empty_cluster_mask = (cluster_counts_np == 0)
        # Set count to 1 for empty clusters to avoid division error (their sum is 0 anyway)
        final_counts_safe_np = np.maximum(cluster_counts_np, 1.0)

        # Calculate new centroids
        centroids_np = new_sums_np / final_counts_safe_np[:, np.newaxis] # Broadcast division

        # Handle empty clusters: keep their old position
        if np.any(empty_cluster_mask):
            centroids_np[empty_cluster_mask] = old_centroids_np[empty_cluster_mask]

        # Safety check for NaN/inf (shouldn't happen with maximum(1) but good practice)
        if not np.all(np.isfinite(centroids_np)):
            print(f"Warning: Non-finite values found in centroids at iteration {i+1}. Replacing with old.")
            centroids_np = np.nan_to_num(centroids_np, nan=old_centroids_np[np.isnan(centroids_np)])


        # --- Check Convergence ---
        centroid_diff_np = np.linalg.norm(centroids_np - old_centroids_np)
        # Optional: Print iteration stats
        # print(f"  Iter {i+1}/{max_iters} | Centroid Diff: {centroid_diff_np:.4f}")

        if centroid_diff_np < tol:
            print(f"Converged after {i+1} iterations.")
            break

    if i == max_iters - 1:
        print(f"Reached max iterations ({max_iters}).")

    total_time = time.time() - start_time_total
    print(f"Total CPU K-Means time: {total_time:.4f}s")

    # Return NumPy arrays
    return centroids_np, assignments_np.astype(np.int64) # Return int64 assignments if needed downstream

# --- Example Usage ---
if __name__ == "__main__":
    # --- Parameters ---
    N_data = 10000
    Dim = 1024
    N_queries = 20000 # Example
    num_clusters_for_kmeans = 500
    K1_probe = 200
    K2_final = 150
    USE_CPU_KMEANS = True # <<<--- Set to True to use the CPU version

    # --- Check CuPy ---
    # ... (cupy check) ...

    # --- Generate Base Data (Still on GPU initially is fine) ---
    print("="*40); print("Generating Base Data (A_cp on GPU)..."); print("="*40)
    try:
        A_data_cp = cp.random.randn(N_data, Dim, dtype=cp.float32)
        print(f"Database shape (GPU): {A_data_cp.shape}")
    except Exception as e: print(f"Error generating A_data_cp: {e}"); exit()

    # --- Build Index ONCE ---
    print("\n" + "="*40)
    print(f"Building Centroids via {'CPU' if USE_CPU_KMEANS else 'GPU'} KMeans (Run Once)...")
    print("="*40)
    initial_centroids_cp = None # This will store the final centroids on GPU
    actual_k_clusters = 0
    build_time_actual = 0
    try:
        build_start_actual = time.time()
        if USE_CPU_KMEANS:
            print("Transferring data to CPU for KMeans...")
            A_data_np = cp.asnumpy(A_data_cp) # GPU -> CPU transfer
            print("Running CPU KMeans...")
            centroids_np, _ = our_kmeans_cpu(N_data, Dim, A_data_np, num_clusters_for_kmeans, max_iters=50)
            # Clear NumPy array if memory is tight
            del A_data_np
            print("Transferring computed centroids back to GPU...")
            initial_centroids_cp = cp.asarray(centroids_np, dtype=cp.float32) # CPU -> GPU transfer
        # Validate centroids computed either way
        if initial_centroids_cp.dtype != cp.float32: initial_centroids_cp = initial_centroids_cp.astype(cp.float32)
        actual_k_clusters = initial_centroids_cp.shape[0]
        build_time_actual = time.time() - build_start_actual
        if actual_k_clusters == 0: raise ValueError("KMeans returned 0 centroids.")
        print(f"{'CPU' if USE_CPU_KMEANS else 'GPU'} KMeans completed. Found {actual_k_clusters} centroids.")
        print(f"Actual Build Time: {build_time_actual:.4f}s")

    except Exception as e: print(f"Error during initial KMeans build: {e}"); import traceback; traceback.print_exc(); exit()

    # --- Generate NEW Queries (on GPU) ---
    # ... (Generate X_queries_cp_new as before) ...

    # --- Run ANN Search (using centroids on GPU) ---
    # The ANN search part remains unchanged as it expects centroids on the GPU ('initial_centroids_cp')
    print("\n" + "="*40); print(f"Testing our_ann_user_pseudocode_impl_l2 with {N_queries} NEW queries..."); print("="*40)
    # ... (Call our_ann_user_pseudocode_impl_l2 using initial_centroids_cp) ...
    # ... (Print results) ...

    # --- Optional: Calculate Recall ---
    # The recall calculation also uses initial_centroids_cp (on GPU)
    # ... (Recall calculation as before) ...