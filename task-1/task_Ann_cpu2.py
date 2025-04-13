import numpy as np
import time
import traceback # For error printing


# ============================================================================
# CPU Helper Function (Pairwise Squared L2 Distance - Required by KMeans)
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

    if X_np.ndim == 1: X_np = X_np[np.newaxis, :] # Ensure X is 2D
    if C_np.ndim == 1: C_np = C_np[np.newaxis, :] # Ensure C is 2D

    if X_np.shape[0] == 0 or C_np.shape[0] == 0:
        return np.empty((X_np.shape[0], C_np.shape[0]), dtype=np.float32)
    if X_np.shape[1] != C_np.shape[1]:
        raise ValueError(f"Dimension mismatch: X_np={X_np.shape[1]}, C_np={C_np.shape[1]}")

    # ||x - c||^2 = ||x||^2 - 2<x, c> + ||c||^2
    try:
        X_norm_sq = np.einsum('ij,ij->i', X_np, X_np)[:, np.newaxis]
        C_norm_sq = np.einsum('ij,ij->i', C_np, C_np)[np.newaxis, :]
        dot_products = np.dot(X_np, C_np.T)
        dist_sq = X_norm_sq + C_norm_sq - 2 * dot_products
        return np.maximum(0.0, dist_sq) # Clamp negatives
    except MemoryError as e:
        print(f"MemoryError in pairwise_l2_squared_numpy: Shapes X={X_np.shape}, C={C_np.shape}")
        raise e
    except Exception as e:
        print(f"Error in pairwise_l2_squared_numpy: {e}")
        raise e

# ============================================================================
# CPU KMeans Implementation
# ============================================================================

def our_kmeans_cpu(N_A, D, A_np, K, max_iters=100, tol=1e-4, verbose=False):
    """
    Performs K-means clustering entirely using NumPy on the CPU.

    Args:
        N_A (int): Number of data points (inferred).
        D (int): Dimensionality (inferred).
        A_np (np.ndarray): Data points (N_A, D) on CPU (NumPy ndarray).
        K (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance for centroid movement convergence check.
        verbose (bool): If True, prints iteration details.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - centroids_np (np.ndarray): Final centroids (K, D), float32.
            - assignments_np (np.ndarray): Final cluster assignment (N_A,), int64.
    """
    # --- Input Validation ---
    if not isinstance(A_np, np.ndarray): A_np = np.asarray(A_np, dtype=np.float32)
    elif A_np.dtype != np.float32: A_np = A_np.astype(np.float32)

    actual_N, actual_D = A_np.shape
    if actual_N == 0: raise ValueError("Input data A_np cannot be empty.")
    N_A = actual_N
    D = actual_D

    if not (K > 0): raise ValueError("K must be positive.")
    if K > N_A:
        print(f"Warning: CPU KMeans requested K ({K}) > N_A ({N_A}). Using K={N_A}.")
        K = N_A
    if K == 0: # K might be 0 after adjustment if N_A was 0 (already checked)
         return np.empty((0, D), dtype=np.float32), np.empty((N_A,), dtype=np.int64)

    print(f"Running K-Means (Pure NumPy CPU): N={N_A}, D={D}, K={K}")
    start_time_total = time.time()

    # --- Initialization ---
    try:
        # Ensure unique indices are chosen if K=N_A
        initial_indices = np.random.choice(N_A, K, replace=False)
    except ValueError as e:
         print(f"Error during KMeans initialization choice: N_A={N_A}, K={K}. Error: {e}")
         raise e # Likely N_A < K, though checks should prevent this
    centroids_np = A_np[initial_indices].copy() # Shape (K, D)
    assignments_np = np.empty(N_A, dtype=np.int32) # Use int32 internally

    # --- Iteration Loop ---
    for i in range(max_iters):
        old_centroids_np = centroids_np.copy()

        # --- Assignment Step ---
        try:
            all_dist_sq_np = pairwise_l2_squared_numpy(A_np, centroids_np) # Shape (N_A, K)
        except MemoryError as e:
            print(f"MemoryError during KMeans assignment step (Iter {i+1}): Calc ({N_A}, {K}) dist matrix.")
            raise e
        except Exception as e:
            print(f"Error during KMeans assignment step (Iter {i+1}): {e}")
            raise e

        new_assignments_np = np.argmin(all_dist_sq_np, axis=1).astype(np.int32) # Shape (N_A,)

        # --- Update Step (Optimized NumPy) ---
        # Faster way to sum points for each cluster than looping
        new_sums_np = np.zeros((K, D), dtype=np.float32)
        np.add.at(new_sums_np, new_assignments_np, A_np) # Efficient in-place sum based on index

        # Count points in each cluster
        cluster_counts_np = np.bincount(new_assignments_np, minlength=K).astype(np.float32)

        # Avoid division by zero for empty clusters
        empty_cluster_mask = (cluster_counts_np == 0)
        final_counts_safe_np = np.maximum(cluster_counts_np, 1.0) # Replace 0 with 1 for division

        # Calculate new centroids
        new_centroids_np = new_sums_np / final_counts_safe_np[:, np.newaxis] # Broadcast division

        # Handle empty clusters: keep old centroid position
        num_empty = np.sum(empty_cluster_mask)
        if num_empty > 0:
            # print(f"  Iter {i+1}: Found {num_empty} empty clusters. Re-using old centroids.")
            new_centroids_np[empty_cluster_mask] = old_centroids_np[empty_cluster_mask]
            if num_empty == K:
                 print(f"Warning: Iter {i+1}, ALL clusters are empty. Stopping iteration.")
                 centroids_np = old_centroids_np # Restore previous state
                 assignments_np = new_assignments_np.astype(np.int64) # Use current assignments
                 break # Stop if all clusters became empty


        # Handle potential NaN/inf resulting from 0/0 or inf/inf (less likely with clamping)
        if not np.all(np.isfinite(new_centroids_np)):
            # print(f"Warning: Non-finite values found in centroids at iteration {i+1}. Replacing.")
            nan_mask = np.isnan(new_centroids_np)
            new_centroids_np[nan_mask] = old_centroids_np[nan_mask] # Replace NaN with old value

        # Update assignments and centroids for next iteration / final return
        assignments_np = new_assignments_np
        centroids_np = new_centroids_np

        # --- Check Convergence ---
        centroid_diff_np = np.linalg.norm(centroids_np - old_centroids_np)

        if verbose and ((i+1) % 10 == 0 or centroid_diff_np < tol or i == max_iters -1):
             print(f"  Iter {i+1}/{max_iters} | Centroid Diff: {centroid_diff_np:.4f}")

        if centroid_diff_np < tol:
            print(f"Converged after {i+1} iterations (centroid movement < {tol}).")
            break

    if i == max_iters - 1: print(f"Reached max iterations ({max_iters}).")

    # Final check for non-empty clusters
    final_counts = np.bincount(assignments_np, minlength=K)
    actual_k = np.sum(final_counts > 0)
    print(f"K-Means finished. Found {actual_k} non-empty clusters out of {K} requested.")

    total_time = time.time() - start_time_total
    print(f"Total CPU K-Means time: {total_time:.4f}s")

    # Return centroids and int64 assignments
    return centroids_np.astype(np.float32), assignments_np.astype(np.int64)


def ann_ivf_like_cpu(
    N_A, D, A_np, X_np, K_final,
    num_clusters, num_clusters_to_probe,
    max_kmeans_iters=100,
    precomputed_centroids=None,
    precomputed_assignments=None,
    verbose_kmeans=False
    ):
    """
    Performs ANN search based on user's pseudocode using NumPy on the CPU.
    Finds K_final nearest DATA POINTS (IVF-like approach).

    Args:
        N_A (int): Number of database points (inferred).
        D (int): Dimension (inferred).
        A_np (np.ndarray): Database vectors (N_A, D), float32, CPU.
        X_np (np.ndarray): Query vectors (Q, D), float32, CPU.
        K_final (int): Final number of nearest *data point* neighbors.
        num_clusters (int): Number of clusters for K-Means.
        num_clusters_to_probe (int): Number of nearest clusters to probe (K1).
        max_kmeans_iters (int): Max iterations for K-Means.
        precomputed_centroids (np.ndarray, optional): (num_clusters, D) centroids.
        precomputed_assignments (np.ndarray, optional): (N_A,) assignments (int64).
        verbose_kmeans (bool): Verbosity for KMeans function (not used in CPU version here).

    Returns:
        tuple[np.ndarray, np.ndarray, float, float]:
            - all_indices_np (np.ndarray): Indices (in A) of K_final nearest neighbors (Q, K_final). Int64.
            - all_distances_sq_np (np.ndarray): **Squared** L2 distances (Q, K_final). Float32.
            - build_time (float): Time for K-Means + Inverted Index.
            - search_time (float): Time for searching all queries.
    """
    print("--- Starting NumPy CPU ANN ---")
    # --- Input Validation & Data Prep ---
    if not isinstance(A_np, np.ndarray): A_np = np.asarray(A_np, dtype=np.float32)
    elif A_np.dtype != np.float32: A_np = A_np.astype(np.float32)
    if not isinstance(X_np, np.ndarray): X_np = np.asarray(X_np, dtype=np.float32)
    elif X_np.dtype != np.float32: X_np = X_np.astype(np.float32)

    if A_np.ndim != 2: raise ValueError(f"Database A_np must be 2D, got shape {A_np.shape}")
    if X_np.ndim != 2: raise ValueError(f"Queries X_np must be 2D, got shape {X_np.shape}")

    actual_N_A, actual_D = A_np.shape
    Q, query_D = X_np.shape

    N_A = actual_N_A
    D = actual_D
    if query_D != D: raise ValueError(f"Dimension mismatch: A D={D}, X D={query_D}")

    if N_A == 0: raise ValueError("Database A_np cannot be empty.")
    if Q == 0: print("Warning: Query set X_np is empty."); # Will return empty

    if not (K_final > 0): raise ValueError("K_final must be positive")
    if not (num_clusters > 0): raise ValueError("num_clusters must be positive")
    if not (num_clusters_to_probe > 0): raise ValueError("num_clusters_to_probe must be positive")

    print(f"Running ANN (NumPy CPU IVF-like): Q={Q}, N={N_A}, D={D}, K_final={K_final}")
    print(f"Params: num_clusters={num_clusters}, nprobe={num_clusters_to_probe}")

    build_time_total = 0.0
    build_start_time = time.time()

    # --- Step 1: K-Means Clustering & Index Setup ---
    assignments_np = None
    centroids_np = None
    kmeans_run_time = 0.0
    if precomputed_centroids is not None and precomputed_assignments is not None:
        print("Using precomputed centroids and assignments.")
        centroids_np = np.asarray(precomputed_centroids, dtype=np.float32)
        assignments_np = np.asarray(precomputed_assignments, dtype=np.int64)
        # Basic validation
        if centroids_np.ndim != 2 or centroids_np.shape[1] != D: raise ValueError("Invalid precomputed centroids shape/dim.")
        if assignments_np.ndim != 1 or assignments_np.shape[0] != N_A: raise ValueError("Invalid precomputed assignments shape.")
        actual_num_clusters = centroids_np.shape[0]
        if actual_num_clusters < num_clusters: print(f"Warning: Provided {actual_num_clusters} centroids < requested {num_clusters}.")
        elif actual_num_clusters > num_clusters:
            print(f"Warning: Using first {num_clusters} of {actual_num_clusters} provided centroids.")
            centroids_np = centroids_np[:num_clusters]
            if assignments_np.max() >= num_clusters: raise ValueError("Assignments index out of bounds after truncating centroids.")
        num_clusters = centroids_np.shape[0]

    elif precomputed_centroids is not None or precomputed_assignments is not None:
        raise ValueError("Provide both or neither of precomputed centroids/assignments.")
    else:
        print("Running KMeans (NumPy CPU)...")
        kmeans_start = time.time()
        centroids_np, assignments_np = our_kmeans_cpu(N_A, D, A_np, num_clusters, max_iters=max_kmeans_iters)
        kmeans_run_time = time.time() - kmeans_start
        actual_num_clusters = centroids_np.shape[0]
        if actual_num_clusters < num_clusters: print(f"Note: KMeans used K={actual_num_clusters}.")
        num_clusters = actual_num_clusters

    # Handle case where KMeans might fail or return no clusters
    if num_clusters == 0 or centroids_np is None or assignments_np is None:
        print("Error: No clusters found or provided. Cannot proceed.")
        empty_indices = np.full((Q, K_final), -1, dtype=np.int64)
        empty_dists = np.full((Q, K_final), np.inf, dtype=np.float32)
        build_time_total = time.time() - build_start_time
        return empty_indices, empty_dists, build_time_total, 0.0

    # Adjust num_clusters_to_probe
    num_clusters_to_probe = min(num_clusters_to_probe, num_clusters)
    if num_clusters_to_probe <= 0:
        print("Error: num_clusters_to_probe is 0. Cannot proceed.")
        empty_indices = np.full((Q, K_final), -1, dtype=np.int64)
        empty_dists = np.full((Q, K_final), np.inf, dtype=np.float32)
        build_time_total = time.time() - build_start_time
        return empty_indices, empty_dists, build_time_total, 0.0

    print(f"Building Inverted Index (NumPy) for {num_clusters} clusters...")
    # --- Build Inverted Index (NumPy) ---
    invidx_start_time = time.time()
    original_indices_np = np.arange(N_A, dtype=np.int64)

    # Sort original indices based on cluster assignments
    sort_permutation_np = np.argsort(assignments_np)
    inv_idx_values_np = original_indices_np[sort_permutation_np] # Original indices sorted by cluster
    sorted_assignments_np = assignments_np[sort_permutation_np] # Cluster IDs sorted

    # Find unique cluster IDs present and their counts/first occurrences
    unique_clusters_np, cluster_starts_np, cluster_counts_np = np.unique(
        sorted_assignments_np, return_index=True, return_counts=True
    )

    # Create full lookup tables
    full_inv_idx_starts_np = np.full((num_clusters,), -1, dtype=np.int64)
    full_inv_idx_counts_np = np.zeros((num_clusters,), dtype=np.int64)

    # Populate the tables
    valid_unique_mask_np = unique_clusters_np < num_clusters
    valid_unique_clusters_np = unique_clusters_np[valid_unique_mask_np]
    if valid_unique_clusters_np.size > 0:
        full_inv_idx_starts_np[valid_unique_clusters_np] = cluster_starts_np[valid_unique_mask_np]
        full_inv_idx_counts_np[valid_unique_clusters_np] = cluster_counts_np[valid_unique_mask_np]
    else:
         print("Warning: No valid unique clusters found for inverted index.")


    invidx_end_time = time.time()
    build_time_total = kmeans_run_time + (invidx_end_time - invidx_start_time)
    print(f"Index build time (Total): {build_time_total:.4f}s (KMeans: {kmeans_run_time:.4f}s, InvIdx: {invidx_end_time - invidx_start_time:.4f}s)")

    # --- Search Phase ---
    search_start_time = time.time()
    all_indices_np = np.full((Q, K_final), -1, dtype=np.int64)
    all_distances_sq_np = np.full((Q, K_final), np.inf, dtype=np.float32)

    # Handle empty query case
    if Q == 0:
        print("Empty query set, returning empty results.")
        return all_indices_np, all_distances_sq_np, build_time_total, 0.0

    # --- Step 2: Find nearest `num_clusters_to_probe` cluster centers ---
    print("Calculating query-centroid distances (CPU)...")
    try:
        all_query_centroid_dists_sq_np = pairwise_l2_squared_numpy(X_np, centroids_np) # Shape (Q, num_clusters)
    except MemoryError as e:
        print(f"MemoryError calculating query-centroid distances: Q={Q}, K={num_clusters}")
        raise e
    except Exception as e:
        print(f"Error calculating query-centroid distances: {e}")
        raise e

    print("Finding nearest clusters (CPU)...")
    # Find indices of the nearest clusters
    try:
        probe_partition_idx = min(num_clusters_to_probe, num_clusters) -1 # 0-based index
        if probe_partition_idx < 0: probe_partition_idx = 0

        if num_clusters_to_probe >= num_clusters:
            all_nearest_cluster_indices_np = np.argsort(all_query_centroid_dists_sq_np, axis=1)[:, :num_clusters_to_probe]
        else:
             if num_clusters > 0:
                 all_nearest_cluster_indices_np = np.argpartition(all_query_centroid_dists_sq_np, kth=probe_partition_idx, axis=1)[:, :num_clusters_to_probe]
             else: # Should not happen
                 all_nearest_cluster_indices_np = np.empty((Q,0), dtype=np.int64)

    except Exception as e:
        print(f"Error finding nearest clusters (argpartition/argsort): {e}")
        raise e

    # --- Step 3 & 4: Gather Candidates & Find Top K_final data points ---
    print(f"Searching {Q} queries (CPU)...")
    # Iterate through queries (CPU loop)
    for q_idx in range(Q):
        query_np = X_np[q_idx:q_idx+1] # Keep 2D: (1, D)
        probed_cluster_indices_np = all_nearest_cluster_indices_np[q_idx] # Shape (num_clusters_to_probe,)

        # --- Gather candidate original indices (from A) ---
        selected_starts_np = full_inv_idx_starts_np[probed_cluster_indices_np]
        selected_counts_np = full_inv_idx_counts_np[probed_cluster_indices_np]

        # Filter out invalid probes
        valid_probe_mask_np = selected_starts_np >= 0
        if not np.any(valid_probe_mask_np): continue # Skip if no valid clusters

        valid_starts_np = selected_starts_np[valid_probe_mask_np]
        valid_counts_np = selected_counts_np[valid_probe_mask_np]

        # Use list comprehension and np.concatenate (common NumPy pattern)
        try:
            candidate_indices_list = [
                inv_idx_values_np[start : start + count]
                for start, count in zip(valid_starts_np, valid_counts_np) if count > 0
            ]
            if not candidate_indices_list: continue # Skip if empty
            candidate_original_indices_np = np.concatenate(candidate_indices_list)
        except ValueError as e: # Handle potential empty list error
            if "need at least one array to concatenate" in str(e):
                 continue # Skip if list becomes empty
            else:
                 print(f"Error concatenating indices for query {q_idx}: {e}")
                 raise e # Re-raise other errors
        except IndexError as e:
            print(f"IndexError gathering candidates for query {q_idx}: {e}")
            continue


        # Remove duplicates
        unique_candidate_original_indices_np = np.unique(candidate_original_indices_np)
        num_unique_candidates = unique_candidate_original_indices_np.size
        if num_unique_candidates == 0: continue

        # --- Fetch candidate vectors ---
        try:
            # Add validity check
            max_idx = np.max(unique_candidate_original_indices_np)
            if max_idx >= N_A:
                 print(f"ERROR: Invalid candidate index {max_idx} >= N_A ({N_A}) for query {q_idx}. Filtering.")
                 valid_cand_mask = unique_candidate_original_indices_np < N_A
                 unique_candidate_original_indices_np = unique_candidate_original_indices_np[valid_cand_mask]
                 num_unique_candidates = unique_candidate_original_indices_np.size
                 if num_unique_candidates == 0: continue

            candidate_vectors_np = A_np[unique_candidate_original_indices_np] # Shape (num_unique, D)
        except MemoryError as e:
            print(f"MemoryError fetching candidates (Query {q_idx}, {num_unique_candidates} candidates): {e}")
            continue
        except IndexError as e:
            print(f"IndexError fetching candidates (Query {q_idx}): {e}")
            continue
        except Exception as e:
             print(f"Error fetching candidates (Query {q_idx}): {e}")
             continue

        # --- Calculate exact distances to candidates ---
        try:
            query_candidate_dists_sq_np = pairwise_l2_squared_numpy(query_np, candidate_vectors_np) # Shape (1, num_unique)
        except MemoryError as e:
            print(f"MemoryError calculating query-candidate dists (Query {q_idx}, {num_unique_candidates} candidates): {e}")
            continue
        except Exception as e:
             print(f"Error calculating query-candidate dists (Query {q_idx}): {e}")
             continue

        # --- Find top K_final among candidates ---
        actual_k_final = min(K_final, num_unique_candidates)
        if actual_k_final > 0:
            try:
                # Use argpartition + argsort
                k_partition_final_idx = min(actual_k_final, num_unique_candidates) - 1
                if k_partition_final_idx < 0: k_partition_final_idx = 0

                if actual_k_final >= num_unique_candidates:
                    topk_relative_indices_np = np.argsort(query_candidate_dists_sq_np[0])[:actual_k_final]
                else:
                    if num_unique_candidates > 0:
                         topk_indices_unstructured = np.argpartition(query_candidate_dists_sq_np[0], kth=k_partition_final_idx)[:actual_k_final]
                         topk_distances_sq_unstructured = query_candidate_dists_sq_np[0, topk_indices_unstructured]
                         sorted_order_in_k = np.argsort(topk_distances_sq_unstructured)
                         topk_relative_indices_np = topk_indices_unstructured[sorted_order_in_k]
                    else: # Should not happen if actual_k_final > 0
                         topk_relative_indices_np = np.empty((0,), dtype=np.int64)


                # Get sorted distances
                final_topk_distances_sq_np = query_candidate_dists_sq_np[0, topk_relative_indices_np]
                # Map relative indices back to original indices from A_np
                final_topk_original_indices_np = unique_candidate_original_indices_np[topk_relative_indices_np]

                # Store results
                all_indices_np[q_idx, :actual_k_final] = final_topk_original_indices_np
                all_distances_sq_np[q_idx, :actual_k_final] = final_topk_distances_sq_np

            except Exception as e:
                print(f"Error during top-K selection for query {q_idx}: {e}")
                # Leave results as -1/inf

    # --- Final Timing ---
    search_time = time.time() - search_start_time
    print(f"ANN search time: {search_time:.4f} seconds")
    if search_time > 0 and Q > 0: print(f"-> Throughput: {Q / search_time:.2f} queries/sec")

    return all_indices_np, all_distances_sq_np, build_time_total, search_time


# ============================================================================
# CPU Brute-Force k-NN (NumPy version - WITH QUERY BATCHING)
# ============================================================================
def numpy_knn_bruteforce(N_A, D, A_np, X_np, K, batch_size_q=1024): # Add batch_size_q parameter
    """
    Finds the K nearest neighbors using brute-force NumPy distance calculation.
    Uses query batching to handle potentially large Q * N distance matrices in RAM.
    Returns original indices (int64) and SQUARED L2 distances (float32).
    Handles K > N_A by padding results.

    Args:
        N_A (int): Number of database points (inferred).
        D (int): Dimension (inferred).
        A_np (np.ndarray): Database vectors (N_A, D) NumPy array.
        X_np (np.ndarray): Query vectors (Q, D) NumPy array.
        K (int): Number of neighbors to find.
        batch_size_q (int): Number of queries to process in each batch.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - all_topk_indices_np (np.ndarray): Indices (Q, K), int64.
            - all_topk_distances_sq_np (np.ndarray): Squared distances (Q, K), float32.
    """
    # --- Input Validation ---
    if not isinstance(A_np, np.ndarray): A_np = np.asarray(A_np, dtype=np.float32)
    elif A_np.dtype != np.float32: A_np = A_np.astype(np.float32)
    if not isinstance(X_np, np.ndarray): X_np = np.asarray(X_np, dtype=np.float32)
    elif X_np.dtype != np.float32: X_np = X_np.astype(np.float32)

    if A_np.ndim != 2: raise ValueError(f"Database A_np must be 2D (N, D), got shape {A_np.shape}")
    if X_np.ndim != 2: raise ValueError(f"Queries X_np must be 2D (Q, D), got shape {X_np.shape}")

    actual_N_A, actual_D = A_np.shape
    Q, query_D = X_np.shape

    if query_D != actual_D: raise ValueError(f"Dimension mismatch: A_np D={actual_D}, X_np D={query_D}")

    # Use actual N_A
    N_A = actual_N_A

    # Handle empty database or query set
    if N_A == 0:
         print("Warning: CPU Brute force called with empty database A_np.")
         return np.full((Q, K), -1, dtype=np.int64), np.full((Q, K), np.inf, dtype=np.float32)
    if Q == 0:
         print("Warning: CPU Brute force called with empty query set X_np.")
         return np.empty((0, K), dtype=np.int64), np.empty((0, K), dtype=np.float32)

    if not K > 0: raise ValueError("K must be positive")

    # Adjust K if it's larger than the number of data points
    effective_K = min(K, N_A)
    if effective_K != K:
         print(f"Note: CPU Brute force K={K} requested > N_A={N_A}. Using K={effective_K}.")

    if effective_K == 0: # Handle K=0 or effective_K=0 case
         return np.empty((Q, 0), dtype=np.int64), np.empty((Q, 0), dtype=np.float32)

    print(f"Running k-NN Brute Force (NumPy CPU, Batched): Q={Q}, N={N_A}, D={actual_D}, K={effective_K}, BatchSize={batch_size_q}")
    start_time = time.time()

    # Pre-allocate result arrays
    all_topk_indices_np = np.full((Q, effective_K), -1, dtype=np.int64)
    all_topk_distances_sq_np = np.full((Q, effective_K), np.inf, dtype=np.float32)

    # --- Batch Processing Loop ---
    for q_start in range(0, Q, batch_size_q):
        q_end = min(q_start + batch_size_q, Q)
        batch_q_indices = slice(q_start, q_end) # Slice for indexing results
        X_batch_np = X_np[batch_q_indices]      # Current batch of queries
        current_batch_size = X_batch_np.shape[0]
        if current_batch_size == 0: continue

        # print(f"  Processing query batch {q_start}-{q_end-1}...") # Optional progress

        # Calculate SQUARED L2 distances for the current batch
        try:
            batch_distances_sq = pairwise_l2_squared_numpy(X_batch_np, A_np) # Shape (current_batch_size, N_A)
        except MemoryError as e:
            batch_mem_gb = current_batch_size * N_A * 4 / (1024**3)
            print(f"MemoryError during CPU Brute Force batch distance calculation:")
            print(f"  Batch Q={current_batch_size}, N={N_A}. Estimated matrix memory: {batch_mem_gb:.2f} GB")
            print(f"  Try reducing batch_size_q (current={batch_size_q}) or ensure sufficient RAM.")
            raise e # Re-raise
        except Exception as e:
            print(f"Error during CPU Brute Force batch distance calculation: {e}")
            raise e

        # Find top K for the current batch using NumPy partition/sort
        # Need kth = K-1 for argpartition (0-based)
        k_partition_idx = min(effective_K, N_A) - 1 # N_A because np allows K=N_A; -1 for 0-based index
        if k_partition_idx < 0: k_partition_idx = 0 # Handle N_A=0 or K=1 case

        batch_topk_indices = None
        batch_topk_distances_sq = None

        try:
            if effective_K >= N_A: # If K includes all points, just sort all
                batch_topk_indices = np.argsort(batch_distances_sq, axis=1)[:, :effective_K]
            else:
                 # Ensure N_A > 0 before calling argpartition
                 if N_A > 0:
                      # Partition to find indices of K smallest (unsorted within K)
                      topk_indices_unstructured = np.argpartition(batch_distances_sq, kth=k_partition_idx, axis=1)[:, :effective_K]
                      # Get the distances for these K indices
                      topk_distances_sq_unstructured = np.take_along_axis(batch_distances_sq, topk_indices_unstructured, axis=1)
                      # Sort within the K elements based on distance
                      sorted_order_in_k = np.argsort(topk_distances_sq_unstructured, axis=1)
                      # Apply sort order to get final indices and distances
                      batch_topk_indices = np.take_along_axis(topk_indices_unstructured, sorted_order_in_k, axis=1)
                 else: # Should not happen if N_A=0 check passed
                      batch_topk_indices = np.empty((current_batch_size, 0), dtype=np.int64)

            # Retrieve the final sorted distances AFTER getting the sorted indices
            if batch_topk_indices is not None and batch_topk_indices.size > 0:
                 batch_topk_distances_sq = np.take_along_axis(batch_distances_sq, batch_topk_indices, axis=1)
            elif effective_K == 0:
                 batch_topk_distances_sq = np.empty((current_batch_size, 0), dtype=np.float32)
            else: # Handle unexpected empty indices
                 print(f"Warning: batch_topk_indices empty/None in batch {q_start}-{q_end-1}")
                 batch_topk_distances_sq = np.full((current_batch_size, effective_K), np.inf, dtype=np.float32)
                 if batch_topk_indices is None: batch_topk_indices = np.full((current_batch_size, effective_K), -1, dtype=np.int64)

            # Store batch results in the main arrays
            all_topk_indices_np[batch_q_indices] = batch_topk_indices
            all_topk_distances_sq_np[batch_q_indices] = batch_topk_distances_sq

            # Optional: Clear intermediate batch results (less critical on CPU usually)
            del batch_distances_sq, batch_topk_indices, batch_topk_distances_sq
            if 'topk_indices_unstructured' in locals(): del topk_indices_unstructured
            if 'topk_distances_sq_unstructured' in locals(): del topk_distances_sq_unstructured
            if 'sorted_order_in_k' in locals(): del sorted_order_in_k

        except Exception as e:
            print(f"Error during CPU Brute Force batch top-K ({q_start}-{q_end-1}): {e}")
            raise e # Raise error to stop potentially incorrect recall calculation
    # --- End Batch Loop ---

    end_time = time.time()
    print(f"k-NN Brute Force (NumPy CPU, Batched) total computation time: {end_time - start_time:.4f} seconds")

    # Pad results if original K > effective_K (i.e., K > N_A)
    if K > effective_K:
        pad_width = K - effective_K
        # Ensure results are not None before padding
        if all_topk_indices_np is None: all_topk_indices_np = np.full((Q, effective_K), -1, dtype=np.int64)
        if all_topk_distances_sq_np is None: all_topk_distances_sq_np = np.full((Q, effective_K), np.inf, dtype=np.float32)

        indices_pad = np.full((Q, pad_width), -1, dtype=np.int64)
        dists_pad = np.full((Q, pad_width), np.inf, dtype=np.float32)
        all_topk_indices_np = np.hstack((all_topk_indices_np, indices_pad))
        all_topk_distances_sq_np = np.hstack((all_topk_distances_sq_np, dists_pad))

    # Ensure final return types are correct
    if all_topk_indices_np is None: all_topk_indices_np = np.full((Q, K), -1, dtype=np.int64)
    if all_topk_distances_sq_np is None: all_topk_distances_sq_np = np.full((Q, K), np.inf, dtype=np.float32)

    return all_topk_indices_np.astype(np.int64), all_topk_distances_sq_np.astype(np.float32)


# ============================================================================
# Example Usage (Optional - How you would call it)
# ============================================================================
# ============================================================================
# Main Execution Block (CPU ONLY - IVF-like ANN + Recall vs True k-NN)
# ============================================================================
if __name__ == "__main__":
    # --- Fixed Parameters ---
    # Reduced N/Q for potentially faster CPU runs across multiple dimensions
    # Adjust these based on your CPU/RAM capabilities
    N_data = 1000000
    N_queries = 1
    K_final_neighbors = 10 # Final K for output

    # ANN Parameters
    num_clusters_kmeans = 1000   # K for KMeans (Step 1)
    num_clusters_probe = 400     # K1 (nprobe) for cluster probing (Step 2)
    kmeans_max_iters = 50       # Max iterations for KMeans

    # Recall threshold
    RECALL_THRESHOLD = 0.70

    # Dimensions to test
    dimensions_to_test = [2, 4, 64, 256, 1024]

    print("\n" + "="*60)
    print("--- NumPy CPU ANN Dimension Test ---")
    print("="*60)
    print(f"Fixed Params: N={N_data}, Q={N_queries}, K_final={K_final_neighbors}")
    print(f"ANN Params: num_clusters={num_clusters_kmeans}, nprobe={num_clusters_probe}")
    print(f"Testing Dimensions: {dimensions_to_test}")
    # --- NO WARMUP PHASE as requested ---

    # --- Main Loop Over Dimensions ---
    print("\n" + "="*60)
    print("--- Starting Dimension Tests ---")
    print("="*60)

    for Dim in dimensions_to_test:
        print("\n" + "#"*70)
        print(f"# Testing Dimension D = {Dim}")
        print("#"*70)

        # --- Per-Dimension Variables ---
        A_data_np = None
        X_queries_np = None
        ann_indices_np = None
        ann_dists_sq_np = None
        true_knn_indices_np = None
        build_t = 0
        search_t = 0

        try:
            # --- Generate Data for Current Dimension ---
            print(f"\n[D={Dim}] Generating Test Data (NumPy CPU)...")
            try:
                A_data_np = np.random.randn(N_data, Dim).astype(np.float32)
                X_queries_np = np.random.randn(N_queries, Dim).astype(np.float32)
                print(f"[D={Dim}] Data generated.")
                mem_gb_a = A_data_np.nbytes / (1024**3)
                mem_gb_x = X_queries_np.nbytes / (1024**3)
                print(f"[D={Dim}] Approx memory: A={mem_gb_a:.3f} GB, X={mem_gb_x:.3f} GB")
            except MemoryError as e:
                print(f"\n[D={Dim}] ERROR: MemoryError during data generation: {e}")
                print("Skipping this dimension.")
                continue # Skip to next dimension
            except Exception as e:
                print(f"\n[D={Dim}] ERROR generating data: {e}")
                continue # Skip to next dimension

            # --- Run ANN for Current Dimension ---
            print(f"\n[D={Dim}] Testing ANN (NumPy CPU IVF-like)...")
            try:
                # Call the CPU IVF-like function
                ann_indices_np, ann_dists_sq_np, build_t, search_t = ann_ivf_like_cpu(
                    N_A=N_data, D=Dim, A_np=A_data_np, X_np=X_queries_np, # Use current Dim data
                    K_final=K_final_neighbors,
                    num_clusters=num_clusters_kmeans,
                    num_clusters_to_probe=num_clusters_probe,
                    max_kmeans_iters=kmeans_max_iters
                )
                print(f"\n[D={Dim}] ANN Results:")
                if ann_indices_np is not None: print(f"  Indices shape: {ann_indices_np.shape}")
                if ann_dists_sq_np is not None: print(f"  Sq Distances shape: {ann_dists_sq_np.shape}")
                print(f"  Build Time: {build_t:.4f}s")
                print(f"  Search Time: {search_t:.4f}s")
                if search_t > 0: print(f"  -> Throughput: {N_queries / search_t:.2f} queries/sec")

            except MemoryError as e:
                print(f"\n[D={Dim}] ERROR: MemoryError during CPU ANN execution: {e}")
                ann_indices_np = None # Prevent recall
            except Exception as e:
                print(f"\n[D={Dim}] ERROR during CPU ANN execution: {e}")
                traceback.print_exc(); ann_indices_np = None # Prevent recall

            # --- Run Brute-Force KNN for Current Dimension ---
            if ann_indices_np is not None: # Only run if ANN succeeded
                print(f"\n[D={Dim}] Calculating Ground Truth (NumPy CPU k-NN)...")
                try:
                    # Call the batched CPU brute-force function
                    true_knn_indices_np, true_knn_dists_sq_np = numpy_knn_bruteforce(
                        N_A=N_data, D=Dim, A_np=A_data_np, X_np=X_queries_np, K=K_final_neighbors, batch_size_q=1024 # Adjust batch size if needed
                    )
                    print(f"\n[D={Dim}] Ground Truth Results:")
                    if true_knn_indices_np is not None: print(f"  Indices shape: {true_knn_indices_np.shape}")
                except MemoryError as e:
                    print(f"\n[D={Dim}] ERROR: MemoryError during CPU Brute Force k-NN: {e}")
                    true_knn_indices_np = None # Prevent recall
                except Exception as e:
                    print(f"\n[D={Dim}] ERROR during CPU Brute Force k-NN execution: {e}")
                    traceback.print_exc(); true_knn_indices_np = None # Prevent recall
                finally:
                     if 'true_knn_dists_sq_np' in locals(): del true_knn_dists_sq_np

            # --- Calculate Recall for Current Dimension ---
            if ann_indices_np is not None and true_knn_indices_np is not None:
                print(f"\n[D={Dim}] Calculating Recall@{K_final_neighbors}...")
                try:
                    total_intersect = 0
                    expected_neighbors_per_query = min(K_final_neighbors, N_data)

                    if N_queries > 0 and expected_neighbors_per_query > 0:
                        for i in range(N_queries):
                            ann_set = set(idx for idx in ann_indices_np[i] if idx >= 0)
                            true_set = set(idx for idx in true_knn_indices_np[i] if idx >= 0)
                            total_intersect += len(ann_set.intersection(true_set))

                        denominator = N_queries * expected_neighbors_per_query
                        avg_recall = total_intersect / denominator if denominator > 0 else 1.0

                        print(f"\n[D={Dim}] Average Recall @ {K_final_neighbors}: {avg_recall:.4f} ({avg_recall:.2%})")
                        if avg_recall >= RECALL_THRESHOLD: print(f"[D={Dim}] Recall meets threshold ({RECALL_THRESHOLD:.2%}). CORRECT.")
                        else: print(f"[D={Dim}] Recall BELOW threshold ({RECALL_THRESHOLD:.2%}). INCORRECT.")
                    else: print(f"\n[D={Dim}] Cannot calculate recall.")
                except Exception as e: print(f"\n[D={Dim}] ERROR during Recall calculation: {e}"); traceback.print_exc()
            elif ann_indices_np is None: print(f"\n[D={Dim}] Skipping Recall: ANN failed.")
            elif true_knn_indices_np is None: print(f"\n[D={Dim}] Skipping Recall: Brute Force failed.")

        finally:
            # --- Cleanup for Current Dimension ---
            print(f"\n[D={Dim}] Cleaning up NumPy arrays...")
            del A_data_np
            del X_queries_np
            if 'ann_indices_np' in locals(): del ann_indices_np
            if 'ann_dists_sq_np' in locals(): del ann_dists_sq_np
            if 'true_knn_indices_np' in locals(): del true_knn_indices_np

        print(f"\n--- Finished Test for Dimension D = {Dim} ---")

    print("\n" + "="*60)
    print("--- ALL CPU DIMENSION TESTS FINISHED ---")
    print("="*60)
