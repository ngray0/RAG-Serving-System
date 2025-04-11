import numpy as np
import time
import traceback # For error printing

# ============================================================================
# CPU Helper Function (Pairwise Squared L2 Distance)
# ============================================================================

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
    X_norm_sq = np.einsum('ij,ij->i', X_np, X_np)[:, np.newaxis] # Shape (N|Q, 1)
    C_norm_sq = np.einsum('ij,ij->i', C_np, C_np)[np.newaxis, :] # Shape (1, K|N)
    dot_products = np.dot(X_np, C_np.T) # Shape (N|Q, K|N)
    dist_sq = X_norm_sq - 2 * dot_products + C_norm_sq
    return np.maximum(0.0, dist_sq) # Clamp negatives

# ============================================================================
# CPU KMeans Implementation
# ============================================================================

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
    if not isinstance(A_np, np.ndarray): raise TypeError("A_np must be NumPy ndarray.")
    actual_N, actual_D = A_np.shape
    if actual_N != N_A or actual_D != D: N_A, D = actual_N, actual_D
    if not (K > 0): raise ValueError("K must be positive.")
    if K > N_A: K = N_A
    if A_np.dtype != np.float32: A_np = A_np.astype(np.float32)

    # print(f"Running K-Means (Pure NumPy CPU): N={N_A}, D={D}, K={K}")
    start_time_total = time.time()
    if K == 0: return np.empty((0, D), dtype=np.float32), np.empty((N_A,), dtype=np.int32)

    # --- Initialization ---
    initial_indices = np.random.choice(N_A, K, replace=False)
    centroids_np = A_np[initial_indices].copy()
    assignments_np = np.empty(N_A, dtype=np.int32)
    old_centroids_np = np.empty_like(centroids_np)

    # --- Iteration Loop ---
    for i in range(max_iters):
        old_centroids_np = centroids_np.copy()
        # --- Assignment Step ---
        all_dist_sq_np = pairwise_l2_squared_numpy(A_np, centroids_np)
        assignments_np = np.argmin(all_dist_sq_np, axis=1).astype(np.int32)
        # --- Update Step (Optimized NumPy) ---
        new_sums_np = np.zeros((K, D), dtype=np.float32)
        np.add.at(new_sums_np, assignments_np, A_np) # Efficient sum
        cluster_counts_np = np.bincount(assignments_np, minlength=K).astype(np.float32)
        # Avoid division by zero
        empty_cluster_mask = (cluster_counts_np == 0)
        final_counts_safe_np = np.maximum(cluster_counts_np, 1.0)
        centroids_np = new_sums_np / final_counts_safe_np[:, np.newaxis]
        # Handle empty clusters
        if np.any(empty_cluster_mask):
            centroids_np[empty_cluster_mask] = old_centroids_np[empty_cluster_mask]
        # Handle potential NaN/inf
        if not np.all(np.isfinite(centroids_np)):
            # print(f"Warning: Non-finite values found in centroids at iteration {i+1}.")
            centroids_np = np.nan_to_num(centroids_np, nan=old_centroids_np[np.isnan(centroids_np)])
        # --- Check Convergence ---
        centroid_diff_np = np.linalg.norm(centroids_np - old_centroids_np)
        if centroid_diff_np < tol:
            # print(f"Converged after {i+1} iterations.")
            break
    # if i == max_iters - 1: print(f"Reached max iterations ({max_iters}).")
    # print(f"Total CPU K-Means time: {time.time() - start_time_total:.4f}s")
    return centroids_np, assignments_np.astype(np.int64)

# ============================================================================
# CPU ANN Function (User Pseudocode, L2 Version)
# ============================================================================
def our_ann_user_pseudocode_impl_l2_cpu(N_A, D, A_np, X_np, k_clusters, K1, K2,
                                        max_kmeans_iters=100, centroids_np=None):
    """
    Implements the user's specific 4-step pseudocode using L2 DISTANCE on the CPU (NumPy).
    Uses SQUARED L2 internally for optimal neighbor finding speed.
    Can optionally accept pre-computed centroids.
    Note: Finds nearest K2 CLUSTER CENTERS based on L2 distance.

    Args:
        N_A (int): Number of database points (for KMeans if centroids_np is None).
        D (int): Dimensionality.
        A_np (np.ndarray): Database vectors (N_A, D) on CPU (for KMeans if centroids_np is None).
        X_np (np.ndarray): Query vectors (Q, D) on CPU.
        k_clusters (int): Target number of clusters for KMeans (if centroids_np is None).
        K1 (int): Number of nearest cluster centers to initially identify (Step 2).
        K2 (int): Number of nearest cluster centers to finally return from the K1 set (Step 3/4).
        max_kmeans_iters (int): Max iterations for K-Means.
        centroids_np (np.ndarray, optional): Pre-computed centroids (k, D). If provided, KMeans is skipped.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
            - topk2_centroid_indices_np (np.ndarray): Indices of the final K2 nearest *centroids* (Q, K2).
            - topk2_centroid_distances_sq_np (np.ndarray): **SQUARED** L2 distances to these K2 *centroids* (Q, K2).
            - centroids_used_np (np.ndarray): The actual centroids used.
            - build_time (float): Time taken for KMeans (0 if centroids_np was provided).
            - search_time (float): Time taken for the search logic.
    """
    # --- Input Validation ---
    Q = X_np.shape[0]
    if not isinstance(X_np, np.ndarray): raise TypeError("X_np must be NumPy ndarray.")
    if X_np.dtype != np.float32: X_np = X_np.astype(np.float32)
    if X_np.shape[1] != D: raise ValueError(f"Query dimension mismatch: X D={X_np.shape[1]}, expected D={D}")

    build_time = 0.0
    search_time = 0.0
    actual_k_clusters = 0
    centroids_used_np = None
    final_topk2_centroid_indices_np = None
    final_topk2_centroid_distances_sq_np = None

    # --- Step 1: Use KMeans or provided centroids ---
    if centroids_np is None:
        # print("No precomputed centroids provided. Running CPU Euclidean KMeans...")
        build_start_time = time.time()
        if not isinstance(A_np, np.ndarray): raise TypeError("A_np needed for KMeans.")
        #... (validate A_np)
        if A_np.dtype != np.float32: A_np = A_np.astype(np.float32)
        if A_np.shape[1] != D: raise ValueError("A_np dimension mismatch.")

        centroids_used_np, _ = our_kmeans_cpu(N_A, D, A_np, k_clusters, max_iters=max_kmeans_iters)
        build_time = time.time() - build_start_time
        # print(f"Build time (CPU KMeans): {build_time:.4f}s")
    else:
        # print("Using precomputed centroids.")
        #... (validate provided centroids_np)
        if not isinstance(centroids_np, np.ndarray): raise TypeError("Provided centroids invalid type.")
        if centroids_np.ndim != 2 or centroids_np.shape[1] != D: raise ValueError(f"Provided centroids invalid shape/dim {centroids_np.shape}.")
        if centroids_np.dtype != np.float32: centroids_np = centroids_np.astype(np.float32)
        centroids_used_np = centroids_np

    actual_k_clusters = centroids_used_np.shape[0]
    if actual_k_clusters == 0: # Error handling
        print("Error: No centroids available. Cannot proceed.")
        empty_indices = np.full((Q, K2 if K2 > 0 else 1), -1, dtype=np.int64)
        empty_dists = np.full((Q, K2 if K2 > 0 else 1), np.inf, dtype=np.float32)
        return empty_indices, empty_dists, np.empty((0,D), dtype=np.float32), build_time, 0.0

    # print(f"Using {actual_k_clusters} centroids for search.")
    K1 = min(K1, actual_k_clusters); K2 = min(K2, K1)
    if K1 <= 0 or K2 <= 0: # Error handling
        print(f"Error: K1 ({K1}) or K2 ({K2}) is non-positive. Cannot proceed.")
        empty_indices = np.full((Q, K2 if K2 > 0 else 1), -1, dtype=np.int64)
        empty_dists = np.full((Q, K2 if K2 > 0 else 1), np.inf, dtype=np.float32)
        return empty_indices, empty_dists, centroids_used_np, build_time, 0.0
    # print(f"Adjusted params: K1={K1}, K2={K2}")

    # --- Search Phase (Using SQUARED L2 Distance) ---
    search_start_time = time.time()

    # Calculate all query-centroid SQUARED L2 distances
    all_query_centroid_dists_sq = pairwise_l2_squared_numpy(X_np, centroids_used_np)

    # Step 2: Find K1 nearest centroids (based on squared L2 distance)
    k1_partition = min(K1, actual_k_clusters) # np.argpartition takes k, not k-1
    if K1 >= actual_k_clusters: topk1_centroid_indices = np.argsort(all_query_centroid_dists_sq, axis=1)[:, :K1]
    else: topk1_centroid_indices = np.argpartition(all_query_centroid_dists_sq, kth=k1_partition-1, axis=1)[:, :K1] # kth is 0-based
    topk1_centroid_dists_sq = np.take_along_axis(all_query_centroid_dists_sq, topk1_centroid_indices, axis=1)

    # Step 3: Find K2 nearest among K1 (based on squared L2 distance)
    k2_partition = min(K2, K1)
    if K2 >= K1: relative_indices_k2 = np.argsort(topk1_centroid_dists_sq, axis=1)[:, :K2]
    else: relative_indices_k2 = np.argpartition(topk1_centroid_dists_sq, kth=k2_partition-1, axis=1)[:, :K2]
    topk2_subset_dists_sq = np.take_along_axis(topk1_centroid_dists_sq, relative_indices_k2, axis=1)
    topk2_centroid_indices_np = np.take_along_axis(topk1_centroid_indices, relative_indices_k2, axis=1)

    # Step 4: Sort final K2 results (based on squared L2 distance)
    sort_order_k2 = np.argsort(topk2_subset_dists_sq, axis=1)
    final_topk2_centroid_indices_np = np.take_along_axis(topk2_centroid_indices_np, sort_order_k2, axis=1)
    final_topk2_centroid_distances_sq_np = np.take_along_axis(topk2_subset_dists_sq, sort_order_k2, axis=1) # SQUARED Distances

    search_time = time.time() - search_start_time
    # print(f"Search time (User Pseudocode, L2 CPU): {search_time:.4f}s")

    return final_topk2_centroid_indices_np.astype(np.int64), \
           final_topk2_centroid_distances_sq_np, \
           centroids_used_np, \
           build_time, \
           search_time

# ============================================================================
# Main Execution Block (CPU ONLY)
# ============================================================================
# Assume necessary imports (numpy as np, time, traceback) and function definitions
# (pairwise_l2_squared_numpy, our_kmeans_cpu, our_ann_user_pseudocode_impl_l2_cpu)
# are present above this block.

if __name__ == "__main__":
    # --- Fixed Parameters for all dimension runs ---
    N_data = 1000000 # Using 1 Million points as requested
    N_queries = 1000  # Reduced queries for faster testing per dimension
    num_clusters_for_kmeans = 100
    K1_probe = 30
    K2_final = 10
    kmeans_max_iters = 50 # Max iterations for KMeans

    # --- Dimensions to Test ---
    dimensions_to_test = [2, 4, 64, 256, 1024]

    print(f"--- CPU ONLY EXECUTION LOOPING THROUGH DIMENSIONS ---")
    print(f"Fixed Params: N={N_data}, Q={N_queries}, k_clusters={num_clusters_for_kmeans}, K1={K1_probe}, K2={K2_final}")
    print(f"Testing Dimensions: {dimensions_to_test}")

    # Loop through each dimension
    for Dim in dimensions_to_test:
        print("\n" + "#"*70)
        print(f"# Starting Test for Dimension D = {Dim}")
        print("#"*70)

        # --- Generate Base Data (on CPU) for the current dimension ---
        print("\n" + "="*40); print(f"Generating Base Data (D={Dim})..."); print("="*40)
        try:
            # Use NumPy directly
            A_data_np = np.random.randn(N_data, Dim).astype(np.float32)
            print(f"Database shape (CPU): {A_data_np.shape}")
            # Check memory usage roughly
            mem_gb = A_data_np.nbytes / (1024**3)
            print(f"Approx. memory for A_data_np: {mem_gb:.2f} GB")
            if mem_gb > 8: # Arbitrary threshold
                 print("Warning: Dataset size is large, KMeans might be very slow / require significant RAM.")

        except MemoryError:
             print(f"Error: Failed to allocate memory for A_data_np (D={Dim}). Skipping this dimension.")
             continue # Skip to the next dimension
        except Exception as e:
            print(f"Error generating A_data_np (D={Dim}): {e}");
            continue # Skip to the next dimension

        # --- Build Index ONCE (CPU KMeans) for the current dimension ---
        print("\n" + "="*40); print(f"Building Centroids via CPU KMeans (D={Dim})..."); print("="*40)
        initial_centroids_np = None
        actual_k_clusters = 0
        build_time_actual = 0
        try:
            build_start_actual = time.time()
            # Call CPU KMeans
            initial_centroids_np, _ = our_kmeans_cpu(N_data, Dim, A_data_np, num_clusters_for_kmeans, max_iters=kmeans_max_iters)
            actual_k_clusters = initial_centroids_np.shape[0]
            build_time_actual = time.time() - build_start_actual
            if actual_k_clusters == 0: raise ValueError("KMeans returned 0 centroids.")
            print(f"CPU KMeans completed. Found {actual_k_clusters} centroids.")
            print(f"Actual Build Time (D={Dim}): {build_time_actual:.4f}s")
        except Exception as e:
            print(f"Error during initial KMeans build (D={Dim}): {e}");
            traceback.print_exc();
            continue # Skip to the next dimension

        # --- Generate NEW Queries (on CPU) for the current dimension ---
        print("\n" + "="*40); print(f"Generating {N_queries} NEW Test Queries (D={Dim})..."); print("="*40)
        try:
            # Use NumPy directly
            X_queries_np_new = np.random.randn(N_queries, Dim).astype(np.float32)
            print(f"New query shape (CPU): {X_queries_np_new.shape}")
        except Exception as e:
            print(f"Error generating new queries (D={Dim}): {e}");
            continue # Skip to the next dimension

        # --- Run ANN Search (CPU Function) for the current dimension ---
        print("\n" + "="*40); print(f"Testing CPU ANN Search (D={Dim})..."); print("="*40)
        k1_run = min(K1_probe, actual_k_clusters)
        k2_run = min(K2_final, k1_run)
        print(f"(Using {actual_k_clusters} centroids, K1={k1_run}, K2={k2_run}, L2 Distance, CPU Search)")
        print("="*40)
        ann_indices_centroids_np = None
        ann_dists_sq_centroids_np = None
        centroids_used_np = None
        search_t = 0
        try:
            # Call the CPU L2 version of the function
            # Pass A_data_np in case the implementation needs N/D from it if centroids aren't passed
            # (though our current one recalculates N/D from A_np if building index inside)
            ann_indices_centroids_np, ann_dists_sq_centroids_np, centroids_used_np, build_t_ignored, search_t = our_ann_user_pseudocode_impl_l2_cpu(
                N_A=N_data, D=Dim, A_np=A_data_np,
                X_np=X_queries_np_new, # Use CPU queries for current Dim
                k_clusters=actual_k_clusters, # Not used if centroids passed
                K1=k1_run,
                K2=k2_run,
                centroids_np=initial_centroids_np # Pass CPU centroids for current Dim
            )
            print("CPU User Pseudocode ANN results shape (Centroid Indices):", ann_indices_centroids_np.shape)
            print("CPU User Pseudocode ANN results shape (**SQUARED** L2 Distances to Centroids):", ann_dists_sq_centroids_np.shape)
            print(f"CPU User Pseudocode ANN Search Time (D={Dim}): {search_t:.4f}s")
            if search_t > 0: print(f"-> Throughput: {N_queries / search_t:.2f} queries/sec (CPU)")

        except Exception as e:
            print(f"Error during ANN execution (D={Dim}): {e}");
            traceback.print_exc();
            # Don't continue to recall if ANN failed
            ann_indices_centroids_np = None # Ensure recall doesn't run

        # --- Calculate Recall (CPU) for the current dimension ---
        if ann_indices_centroids_np is not None and centroids_used_np is not None and centroids_used_np.shape[0] > 0:
            print("\n" + "="*40); print(f"Calculating Recall (D={Dim}, CPU L2)..."); print("="*40)
            K_recall = k2_run
            try:
                # Ground truth calculation on CPU
                # print("Calculating ground truth (CPU Brute-force nearest centroids using SQUARED L2)...")
                start_gt = time.time()
                all_query_centroid_dists_sq_gt = pairwise_l2_squared_numpy(X_queries_np_new, centroids_used_np)

                actual_num_centroids = centroids_used_np.shape[0]
                k_recall_adjusted = min(K_recall, actual_num_centroids)

                if k_recall_adjusted > 0:
                    true_knn_centroid_indices_np = np.argsort(all_query_centroid_dists_sq_gt, axis=1)[:, :k_recall_adjusted]
                else:
                    true_knn_centroid_indices_np = np.empty((N_queries, 0), dtype=np.int64)
                # print(f"Ground truth calculation time: {time.time() - start_gt:.4f}s")

                # Compare results
                total_intersect_centroids = 0
                ann_indices_np = ann_indices_centroids_np[:, :k_recall_adjusted]
                true_indices_np = true_knn_centroid_indices_np
                for i in range(N_queries):
                    approx_centroid_ids = set(idx for idx in ann_indices_np[i] if idx >= 0)
                    true_centroid_ids = set(true_indices_np[i])
                    total_intersect_centroids += len(approx_centroid_ids.intersection(true_centroid_ids))

                if N_queries > 0 and k_recall_adjusted > 0:
                    avg_recall_centroids = total_intersect_centroids / (N_queries * k_recall_adjusted)
                    print(f"\nAverage Recall @ {k_recall_adjusted} (vs CPU brute-force CENTROIDS w/ L2, D={Dim}): {avg_recall_centroids:.4f} ({avg_recall_centroids:.2%})")
                    # Commentary
                    epsilon = 1e-9
                    if abs(avg_recall_centroids - 1.0) < epsilon: print("Result: 100% recall indicates K1 was large enough...")
                    else: print(f"Result: Recall ({avg_recall_centroids:.4f}) < 100%...")
                else: print("\nCannot calculate recall (N_queries=0 or K2=0).")

            except Exception as e:
                print(f"Error during recall calculation (D={Dim}): {e}");
                traceback.print_exc()
        else:
             print("\nSkipping recall calculation as ANN results or centroids are unavailable.")

        print(f"\n--- Finished Test for Dimension D = {Dim} ---")
        # Optional: clean up large array to free memory before next iteration
        del A_data_np, X_queries_np_new, initial_centroids_np
        if 'centroids_used_np' in locals(): del centroids_used_np
        if 'ann_indices_centroids_np' in locals(): del ann_indices_centroids_np
        if 'ann_dists_sq_centroids_np' in locals(): del ann_dists_sq_centroids_np


    print("\n--- ALL CPU DIMENSION TESTS FINISHED ---")