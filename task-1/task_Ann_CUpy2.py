import triton
import triton.language as tl
import math
# import heapq # Not needed for the IVF-like implementation, but keep if HNSW part is used elsewhere
import random
import time
import cupy as cp
import cupyx # Required for cupyx.scatter_add
import torch # Keep for device check or if HNSW part is used elsewhere

pytorch_device = None
if torch.cuda.is_available():
    pytorch_device = torch.device("cuda:0")
    print(f"PyTorch using device: {pytorch_device}")
else:
    pytorch_device = torch.device("cpu")
    print("PyTorch falling back to CPU.")

# Check and set CuPy device
cupy_device_ok = False
try:
    # Explicitly get device 0 and use it
    dev = cp.cuda.Device(0)
    dev.use()
    print(f"CuPy using GPU: {dev} ({cp.cuda.runtime.getDeviceProperties(0)['name']})")
    cupy_device_ok = True
except cp.cuda.runtime.CUDARuntimeError as e:
    print(f"CuPy CUDA Error: {e}")
    print("Cannot run CuPy operations without CUDA.")
    # Depending on the use case, you might exit here or let subsequent CuPy calls fail.
    # exit()


# --- Helper Functions (Less relevant for CuPy part, but keep for context) ---

# --- Distance Functions & Kernels (Triton kernel - Keep if HNSW is used elsewhere) ---
DEFAULT_BLOCK_D = 128


# ============================================================================
# CuPy Distance Functions (Used by K-Means and IVF-like ANN)
# ============================================================================

def pairwise_l2_squared_cupy(X_cp, C_cp):
    """
    Computes pairwise **squared** L2 distances using CuPy.
    X_cp: (N, D) data points OR (Q, D) query points
    C_cp: (K, D) centroids OR (N, D) database points
    Returns: (N|Q, K|N) tensor of SQUARED distances.
    """
    # Ensure inputs are CuPy arrays and float32
    if not isinstance(X_cp, cp.ndarray): X_cp = cp.asarray(X_cp, dtype=cp.float32)
    elif X_cp.dtype != cp.float32: X_cp = X_cp.astype(cp.float32)
    if not isinstance(C_cp, cp.ndarray): C_cp = cp.asarray(C_cp, dtype=cp.float32)
    elif C_cp.dtype != cp.float32: C_cp = C_cp.astype(cp.float32)

    if X_cp.ndim == 1: X_cp = X_cp[cp.newaxis, :] # Ensure X is 2D
    if C_cp.ndim == 1: C_cp = C_cp[cp.newaxis, :] # Ensure C is 2D

    # Handle empty inputs gracefully
    if X_cp.shape[0] == 0 or C_cp.shape[0] == 0:
        return cp.empty((X_cp.shape[0], C_cp.shape[0]), dtype=cp.float32)

    # Check dimension compatibility AFTER ensuring 2D and non-empty
    if X_cp.shape[1] != C_cp.shape[1]:
        raise ValueError(f"Dimension mismatch: X_cp has D={X_cp.shape[1]}, C_cp has D={C_cp.shape[1]}")

    # ||x - c||^2 = ||x||^2 - 2<x, c> + ||c||^2 (optimized calculation)
    try:
        X_norm_sq = cp.einsum('ij,ij->i', X_cp, X_cp)[:, cp.newaxis] # Shape (N|Q, 1)
        C_norm_sq = cp.einsum('ij,ij->i', C_cp, C_cp)[cp.newaxis, :] # Shape (1, K|N)
        # Use gemm for dot product (generally fastest) via matmul
        dot_products = cp.matmul(X_cp, C_cp.T) # Shape (N|Q, K|N)

        # Broadcasting: (N|Q, 1) - 2*(N|Q, K|N) + (1, K|N) -> (N|Q, K|N)
        dist_sq = X_norm_sq - 2 * dot_products + C_norm_sq
        return cp.maximum(0.0, dist_sq) # Clamp numerical negatives

    except cp.cuda.memory.OutOfMemoryError as e:
        print(f"OOM Error in pairwise_l2_squared_cupy: Shapes X={X_cp.shape}, C={C_cp.shape}")
        # Estimate memory needed for intermediate products if helpful
        dot_prod_mem = X_cp.shape[0] * C_cp.shape[0] * 4 / (1024**3) # GB
        print(f"Estimated memory for dot product matrix: {dot_prod_mem:.2f} GB")
        raise e # Re-raise the exception
    except Exception as e:
        print(f"Error in pairwise_l2_squared_cupy: {e}")
        raise e


def pairwise_l2_distance_cupy(X_cp, C_cp):
    """
    Computes pairwise standard L2 Euclidean distances (non-squared) using CuPy.
    Uses the optimized squared calculation internally.
    Returns: (N|Q, K|N) tensor of L2 distances.
    """
    dist_sq = pairwise_l2_squared_cupy(X_cp, C_cp)
    # Handle potential NaNs from sqrt(negative) if maximum(0.0,...) wasn't perfect
    dist = cp.sqrt(cp.maximum(0.0, dist_sq)) # Ensure input to sqrt is non-negative
    return dist

# ============================================================================
# K-Means Clustering (Pure CuPy Implementation - Required Building Block)
# ============================================================================

def our_kmeans(N_A, D, A_cp, K, max_iters=100, tol=1e-4):
    """
    Performs K-means clustering entirely using CuPy.
    Returns centroids_cp (K_actual, D) and assignments_cp (N_A,) dtype=int32.
    Handles potential empty clusters by keeping the previous centroid position.
    """
    # --- Input Validation ---
    if not cupy_device_ok: raise RuntimeError("CuPy device not available.")
    if not isinstance(A_cp, cp.ndarray):
        raise TypeError("Input data 'A_cp' must be a CuPy ndarray.")
    if A_cp.shape[0] == 0:
         raise ValueError("Input data 'A_cp' cannot be empty.")
    if A_cp.shape[0] != N_A or A_cp.shape[1] != D:
        print(f"Warning: N_A/D ({N_A}/{D}) mismatch with A_cp shape {A_cp.shape}. Using shape from A_cp.")
        N_A, D = A_cp.shape # Correct N_A and D based on actual data
        if N_A == 0: raise ValueError("Input data 'A_cp' is empty after shape correction.")

    if not (K > 0):
        raise ValueError("K must be positive.")
    # Allow K > N_A, but cap it and issue a warning
    if K > N_A:
        print(f"Warning: Requested K ({K}) > N_A ({N_A}). Using K={N_A}.")
        K = N_A

    if A_cp.dtype != cp.float32:
        print(f"Warning: Input data dtype is {A_cp.dtype}. Converting to float32.")
        A_cp = A_cp.astype(cp.float32)

    print(f"Running K-Means (Pure CuPy): N={N_A}, D={D}, K={K}")
    start_time_total = time.time()

    # --- Initialization (using CuPy random choice) ---
    # Ensure we select K unique indices from N_A points
    try:
        initial_indices = cp.random.choice(N_A, K, replace=False)
    except ValueError as e:
        # This can happen if N_A < K after potential adjustments, though logic above should prevent it.
        print(f"Error during KMeans initialization: {e}. N_A={N_A}, K={K}")
        raise e
    centroids_cp = A_cp[initial_indices].copy() # Shape (K, D)

    assignments_cp = cp.empty(N_A, dtype=cp.int32) # Use int32 for assignments
    # old_centroids_cp = cp.zeros_like(centroids_cp) # Store previous iteration's centroids for convergence check

    for i in range(max_iters):
        iter_start_time = time.time()
        old_centroids_cp = centroids_cp.copy() # Store centroids *before* update

        # --- 1. Assignment Step (CuPy) ---
        # pairwise_l2_squared_cupy is generally faster for finding minimums
        try:
             all_dist_sq_cp = pairwise_l2_squared_cupy(A_cp, centroids_cp) # Shape (N_A, K)
        except cp.cuda.memory.OutOfMemoryError as e:
             print(f"OOM Error during KMeans assignment step (Iter {i+1}): Calculating ({N_A}, {K}) distance matrix.")
             mem_gb = N_A * K * 4 / (1024**3)
             print(f"Estimated memory for distance matrix: {mem_gb:.2f} GB")
             raise e # Re-raise
        except Exception as e:
             print(f"Error during KMeans assignment step (Iter {i+1}): {e}")
             raise e

        new_assignments_cp = cp.argmin(all_dist_sq_cp, axis=1).astype(cp.int32) # Shape (N_A,)
        cp.cuda.Stream.null.synchronize()
        assign_time = time.time() - iter_start_time

        # Optional: Check if assignments changed (can converge faster sometimes)
        # if i > 0 and cp.all(new_assignments_cp == assignments_cp):
        #    print(f"Converged after {i} iterations (assignments stable).")
        #    break
        assignments_cp = new_assignments_cp # Update assignments

        # --- 2. Update Step (CuPy using cupyx.scatter_add) ---
        update_start_time = time.time()
        new_sums_cp = cp.zeros((K, D), dtype=cp.float32)
        cluster_counts_cp = cp.zeros(K, dtype=cp.float32) # Use float for scatter_add

        try:
            # Use cupyx.scatter_add (ensure cupyx is imported)
            cupyx.scatter_add(new_sums_cp, assignments_cp, A_cp) # Add vectors
            cupyx.scatter_add(cluster_counts_cp, assignments_cp, 1.0) # Increment count
        except Exception as e:
            print(f"Error during KMeans scatter_add update (Iter {i+1}): {e}")
            raise e

        # Avoid division by zero for empty clusters
        # Keep old centroid position if cluster becomes empty
        empty_cluster_mask = (cluster_counts_cp < 1e-9) # Use small threshold for float comparison
        non_empty_cluster_mask = ~empty_cluster_mask

        # Create new centroids array to store updates
        new_centroids_cp = cp.zeros_like(centroids_cp)

        # Update only non-empty clusters
        # Make counts safe for division where necessary
        safe_counts = cluster_counts_cp[non_empty_cluster_mask]
        if safe_counts.size > 0: # Check if there are any non-empty clusters
             new_centroids_cp[non_empty_cluster_mask] = new_sums_cp[non_empty_cluster_mask] / safe_counts[:, None] # Broadcast division

        # Keep old positions for empty clusters (handle case where all might be empty)
        num_empty = cp.sum(empty_cluster_mask)
        if num_empty > 0:
            new_centroids_cp[empty_cluster_mask] = old_centroids_cp[empty_cluster_mask]
            # print(f"Warning: Iter {i+1}, found {num_empty} empty clusters. Re-using old centroids.")
            if num_empty == K:
                 print(f"Warning: Iter {i+1}, ALL clusters are empty. Stopping iteration.")
                 centroids_cp = old_centroids_cp # Restore previous state
                 break # Stop if all clusters became empty


        cp.cuda.Stream.null.synchronize()
        update_time = time.time() - update_start_time

        # --- Check Convergence (based on centroid movement) ---
        # Compare new_centroids_cp with old_centroids_cp saved at the start of the loop
        centroid_diff_cp = cp.linalg.norm(new_centroids_cp - old_centroids_cp)
        centroids_cp = new_centroids_cp # Update centroids for next iteration calculation / final return

        # print(f"  Iter {i+1}/{max_iters} | Centroid Diff: {centroid_diff_cp:.4f} | Assign Time: {assign_time:.4f}s | Update Time: {update_time:.4f}s")

        if centroid_diff_cp < tol:
            print(f"Converged after {i+1} iterations (centroid movement < {tol}).")
            break
        # Prevent infinite loop if centroids somehow oscillate around tol
        # if cp.allclose(centroids_cp, old_centroids_cp, atol=tol/10):
        #     print(f"Converged after {i+1} iterations (centroids stable within tolerance).")
        #     break


    if i == max_iters - 1:
        print(f"Reached max iterations ({max_iters}).")

    total_time = time.time() - start_time_total
    print(f"Total K-Means time: {total_time:.4f}s")

    # Final pass to identify which centroids actually have points assigned
    # This helps determine the actual number of non-empty clusters.
    final_counts = cp.bincount(assignments_cp, minlength=K)
    non_empty_final_mask = (final_counts > 0)
    actual_k = cp.sum(non_empty_final_mask)
    print(f"K-Means finished. Found {actual_k} non-empty clusters out of {K} requested.")

    # Return all K centroids (some might correspond to empty clusters if not handled downstream)
    # and the final assignment array.
    return centroids_cp, assignments_cp.astype(cp.int32) # Ensure assignments are int32

# ============================================================================
# ANN Implementation based on User's New Pseudocode (IVF-like)
# ============================================================================

def ann_user_pseudocode_ivf_like(
    N_A, D, A_cp, X_cp, K_final,
    num_clusters, num_clusters_to_probe, # K_final = K, num_clusters=K from pseudo, num_clusters_to_probe=K1 from pseudo
    max_kmeans_iters=100,
    precomputed_centroids=None,
    precomputed_assignments=None
    ):
    """
    Performs ANN search based on the user's updated 4-step pseudocode,
    interpreted as an IVF-like approach finding nearest DATA POINTS.

    Pseudocode Mapping:
    1. Use KMeans -> Handled by `our_kmeans` or precomputed inputs.
    2. Find nearest K1 cluster centers -> `num_clusters_to_probe` nearest centroids found.
    3. Use KNN ... from K1 cluster centers -> Interpreted as "Identify K1 clusters to search".
    4. Merge vectors [from K1 clusters] and find top K -> Search data points in probed clusters.

    Args:
        N_A (int): Number of database points (can be inferred from A_cp).
        D (int): Dimensionality (can be inferred).
        A_cp (cp.ndarray): Database vectors (N_A, D) on GPU. Should be float32.
        X_cp (cp.ndarray): Query vectors (Q, D) on GPU. Should be float32.
        K_final (int): Final number of nearest *data point* neighbors to return.
        num_clusters (int): Number of clusters for K-Means (Step 1 K).
        num_clusters_to_probe (int): Number of nearest clusters to probe for candidates (Step 2 K1).
        max_kmeans_iters (int): Max iterations for K-Means if not precomputed.
        precomputed_centroids (cp.ndarray, optional): (num_clusters, D) centroids. If provided, skips KMeans.
        precomputed_assignments (cp.ndarray, optional): (N_A,) assignments. Needed if centroids provided.

    Returns:
        tuple[cp.ndarray, cp.ndarray, float, float]:
            - all_indices_cp (cp.ndarray): Original indices (in A_cp) of the K_final nearest neighbors (Q, K_final). Int64.
            - all_distances_sq_cp (cp.ndarray): **Squared** L2 distances of the K_final nearest neighbors (Q, K_final). Float32.
            - build_time (float): Time taken to build the index (KMeans + Inverted Index). 0 if precomputed.
            - search_time (float): Time taken for searching all queries.
    """
    if not cupy_device_ok: raise RuntimeError("CuPy device not available.")

    # --- Input Validation & Data Prep ---
    if not isinstance(A_cp, cp.ndarray): A_cp = cp.asarray(A_cp, dtype=cp.float32)
    elif A_cp.dtype != cp.float32: A_cp = A_cp.astype(cp.float32)
    if not isinstance(X_cp, cp.ndarray): X_cp = cp.asarray(X_cp, dtype=cp.float32)
    elif X_cp.dtype != cp.float32: X_cp = X_cp.astype(cp.float32)

    if A_cp.ndim != 2: raise ValueError(f"Database A_cp must be 2D (N, D), got shape {A_cp.shape}")
    if X_cp.ndim != 2: raise ValueError(f"Queries X_cp must be 2D (Q, D), got shape {X_cp.shape}")

    actual_N_A, actual_D = A_cp.shape
    Q, query_D = X_cp.shape

    if actual_N_A == 0: raise ValueError("Database A_cp cannot be empty.")
    if Q == 0: print("Warning: Query set X_cp is empty. Returning empty results."); # Continue to return empty arrays

    # Use actual dimensions from data
    N_A = actual_N_A
    D = actual_D
    if query_D != D: raise ValueError(f"Dimension mismatch: A_cp has D={D}, X_cp has D={query_D}")

    if not (K_final > 0): raise ValueError("K_final must be positive")
    if not (num_clusters > 0): raise ValueError("num_clusters must be positive")
    if not (num_clusters_to_probe > 0): raise ValueError("num_clusters_to_probe must be positive")

    # K_final can be larger than N_A, will be capped later per query if needed.

    print(f"Running ANN (User Pseudocode - IVF-like): Q={Q}, N={N_A}, D={D}, K_final={K_final}")
    print(f"Params: num_clusters={num_clusters}, num_clusters_to_probe={num_clusters_to_probe}")

    build_time_total = 0.0
    build_start_time = time.time()

    # --- Step 1: K-Means Clustering & Index Setup ---
    assignments_cp = None
    centroids_cp = None
    if precomputed_centroids is not None and precomputed_assignments is not None:
        print("Using precomputed centroids and assignments.")
        centroids_cp = cp.asarray(precomputed_centroids, dtype=cp.float32)
        assignments_cp = cp.asarray(precomputed_assignments, dtype=cp.int32)
        # Basic validation of precomputed data
        if centroids_cp.ndim != 2 or centroids_cp.shape[1] != D:
             raise ValueError(f"Precomputed centroids shape error: expected (*, {D}), got {centroids_cp.shape}")
        if assignments_cp.ndim != 1 or assignments_cp.shape[0] != N_A:
             raise ValueError(f"Precomputed assignments shape error: expected ({N_A},), got {assignments_cp.shape}")
        actual_num_clusters = centroids_cp.shape[0]
        if actual_num_clusters < num_clusters:
             print(f"Warning: Provided {actual_num_clusters} centroids, less than requested num_clusters {num_clusters}.")
        elif actual_num_clusters > num_clusters:
             print(f"Warning: Provided {actual_num_clusters} centroids, more than requested num_clusters {num_clusters}. Using first {num_clusters}.")
             centroids_cp = centroids_cp[:num_clusters]
             # Warning: Assignments might now refer to clusters beyond the truncated centroids. Check max assignment index.
             if assignments_cp.max() >= num_clusters:
                 print("ERROR: Precomputed assignments refer to cluster IDs >= truncated num_clusters. Inconsistent input.")
                 # Handle error appropriately - maybe re-run KMeans or raise error
                 raise ValueError("Inconsistent precomputed assignments and truncated centroids.")
        # Update num_clusters to reflect the actual number being used
        num_clusters = centroids_cp.shape[0]

    elif precomputed_centroids is not None or precomputed_assignments is not None:
        raise ValueError("Must provide both or neither of precomputed_centroids and precomputed_assignments.")
    else:
        print("Running KMeans...")
        # Cap requested clusters if > N_A (already handled inside our_kmeans)
        centroids_cp, assignments_cp = our_kmeans(N_A, D, A_cp, num_clusters, max_iters=max_kmeans_iters)
        actual_num_clusters = centroids_cp.shape[0] # our_kmeans returns centroids array, shape[0] is actual K
        if actual_num_clusters < num_clusters:
            print(f"Note: KMeans used K={actual_num_clusters}.")
        num_clusters = actual_num_clusters # Adjust parameter to match reality

    # Ensure consistency after KMeans or using precomputed
    if num_clusters == 0 or centroids_cp is None or assignments_cp is None:
        print("Error: No clusters found or provided after Step 1. Cannot proceed.")
        empty_indices = cp.full((Q, K_final), -1, dtype=cp.int64)
        empty_dists = cp.full((Q, K_final), cp.inf, dtype=cp.float32)
        return empty_indices, empty_dists, time.time() - build_start_time, 0.0

    # Adjust num_clusters_to_probe based on the actual number of clusters
    num_clusters_to_probe = min(num_clusters_to_probe, num_clusters)
    if num_clusters_to_probe <= 0:
        print("Error: num_clusters_to_probe became 0 after adjustment. Cannot proceed.")
        # Handle returning empty results as above
        empty_indices = cp.full((Q, K_final), -1, dtype=cp.int64)
        empty_dists = cp.full((Q, K_final), cp.inf, dtype=cp.float32)
        return empty_indices, empty_dists, time.time() - build_start_time, 0.0

    print(f"Building Inverted Index for {num_clusters} clusters...")
    # --- Build Inverted Index (GPU Optimized) ---
    build_invidx_start_time = time.time()

    try:
        original_indices = cp.arange(N_A, dtype=cp.int64) # Keep original indices as int64

        # Sort original indices based on cluster assignments
        sort_permutation = cp.argsort(assignments_cp)
        inv_idx_values_cp = original_indices[sort_permutation] # Contains original indices sorted by cluster
        sorted_assignments = assignments_cp[sort_permutation] # Cluster IDs corresponding to inv_idx_values_cp

        # Find start positions and counts for each cluster ID that is actually present
        unique_clusters, inv_idx_starts_cp, inv_idx_counts_cp = cp.unique(
            sorted_assignments, return_index=True, return_counts=True
        )

        # Create mappings for potentially empty clusters (map cluster_id -> index in unique_clusters)
        # Use actual_num_clusters for the size of these mapping arrays
        cluster_id_to_unique_idx = cp.full(num_clusters, -1, dtype=cp.int32)
        present_unique_clusters = unique_clusters[unique_clusters < num_clusters] # Ensure unique cluster IDs are valid
        cluster_id_to_unique_idx[present_unique_clusters] = cp.arange(len(present_unique_clusters), dtype=cp.int32)

        # Precompute starts and counts mapped to the full 0..num_clusters-1 range
        full_inv_idx_starts = cp.full(num_clusters, -1, dtype=cp.int32) # Default: invalid start index
        full_inv_idx_counts = cp.zeros(num_clusters, dtype=cp.int32)  # Default: zero count

        present_mask = (cluster_id_to_unique_idx != -1) # Mask for clusters that have points
        indices_in_unique = cluster_id_to_unique_idx[present_mask] # Indices into unique_clusters arrays

        if indices_in_unique.size > 0:
            present_cluster_ids = cp.where(present_mask)[0] # Get the actual cluster IDs (0..num_clusters-1) that are present
            # Filter starts and counts based on valid unique_clusters before assignment
            valid_starts = inv_idx_starts_cp[unique_clusters < num_clusters]
            valid_counts = inv_idx_counts_cp[unique_clusters < num_clusters]
            full_inv_idx_starts[present_cluster_ids] = valid_starts[indices_in_unique]
            full_inv_idx_counts[present_cluster_ids] = valid_counts[indices_in_unique]

    except Exception as e:
        print(f"Error during inverted index creation: {e}")
        raise e

    cp.cuda.Stream.null.synchronize() # Sync after index build
    build_time_total = time.time() - build_start_time
    print(f"Index build time (Total): {build_time_total:.4f}s")

    # --- Search Phase ---
    search_start_time = time.time()
    all_indices_cp = cp.full((Q, K_final), -1, dtype=cp.int64)
    all_distances_sq_cp = cp.full((Q, K_final), cp.inf, dtype=cp.float32)

    # Handle empty query case
    if Q == 0:
         print("Empty query set, returning empty results.")
         return all_indices_cp, all_distances_sq_cp, build_time_total, 0.0

    # --- Step 2: Find nearest `num_clusters_to_probe` cluster centers ---
    try:
        # Calculate all query-centroid distances at once (vectorized)
        all_query_centroid_dists_sq = pairwise_l2_squared_cupy(X_cp, centroids_cp) # Shape (Q, num_clusters)
    except cp.cuda.memory.OutOfMemoryError as e:
        print(f"OOM Error calculating query-centroid distances: Q={Q}, K={num_clusters}")
        mem_gb = Q * num_clusters * 4 / (1024**3)
        print(f"Estimated memory for distance matrix: {mem_gb:.2f} GB")
        raise e
    except Exception as e:
        print(f"Error calculating query-centroid distances: {e}")
        raise e


    # Find indices of the nearest clusters for all queries at once (vectorized)
    # Need k < N for argpartition
    probe_partition = min(num_clusters_to_probe, num_clusters - 1) if num_clusters > 0 else 0
    if probe_partition < 0: probe_partition = 0 # Ensure non-negative

    try:
        if num_clusters_to_probe >= num_clusters: # If probing all, just use argsort
             all_nearest_cluster_indices = cp.argsort(all_query_centroid_dists_sq, axis=1)[:, :num_clusters_to_probe]
        else:
             # Ensure argpartition is called only if num_clusters > 0
             if num_clusters > 0:
                 all_nearest_cluster_indices = cp.argpartition(all_query_centroid_dists_sq, probe_partition, axis=1)[:, :num_clusters_to_probe]
             else: # Should not happen based on earlier checks, but safety
                 all_nearest_cluster_indices = cp.empty((Q,0), dtype=cp.int32)

        # Shape (Q, num_clusters_to_probe) - Indices are centroid indices (0 to num_clusters-1)
        all_nearest_cluster_indices = all_nearest_cluster_indices.astype(cp.int32) # Ensure int32 for indexing helper arrays
    except Exception as e:
        print(f"Error finding nearest clusters (argpartition/argsort): {e}")
        raise e

    # --- Step 3 & 4: Gather Candidates & Find Top K_final data points ---
    # Iterate through queries (CPU loop, but GPU work inside)
    for q_idx in range(Q):
        query_cp = X_cp[q_idx:q_idx+1] # Keep it 2D: (1, D) for pairwise_l2_squared_cupy
        # Indices of the clusters to probe for *this* query
        nearest_cluster_indices = all_nearest_cluster_indices[q_idx] # Shape (num_clusters_to_probe,)

        # --- Gather candidate points from selected clusters (GPU Optimized) ---
        # Use advanced indexing to get starts and counts for probed clusters
        selected_starts = full_inv_idx_starts[nearest_cluster_indices]
        selected_counts = full_inv_idx_counts[nearest_cluster_indices]

        candidate_indices_list_gpu = []
        # This CPU loop iterates num_clusters_to_probe times (relatively small)
        for i in range(num_clusters_to_probe):
            start = selected_starts[i].item() # Get scalar value
            count = selected_counts[i].item() # Get scalar value
            # Check if cluster is valid (start index >= 0) and has points (count > 0)
            if start >= 0 and count > 0:
                # Append a view/slice of the GPU array (no data copy yet)
                try:
                    candidate_indices_list_gpu.append(inv_idx_values_cp[start : start + count])
                except IndexError as e:
                     print(f"Error slicing inv_idx_values_cp at query {q_idx}, probe {i}: start={start}, count={count}, total_size={inv_idx_values_cp.size}. Error: {e}")
                     # This indicates a potential mismatch in inverted index construction or assignments
                     continue # Skip this probe index


        if not candidate_indices_list_gpu:
            # print(f"Query {q_idx}: No candidates found in probed clusters.") # Optional debug
            continue # No candidates found for this query, leave results as -1/inf

        # Concatenate candidate original indices ON THE GPU
        try:
            candidate_original_indices_cp = cp.concatenate(candidate_indices_list_gpu)
        except ValueError as e:
             # This might happen if candidate_indices_list_gpu somehow ends up empty despite the check
             print(f"Warning: cp.concatenate failed for query {q_idx}. List was likely empty. Error: {e}")
             continue


        # Remove duplicate candidates that might appear if clusters overlap conceptually
        # or if a point is near boundaries. Necessary for correctness & efficiency.
        unique_candidate_original_indices_cp = cp.unique(candidate_original_indices_cp)

        num_unique_candidates = unique_candidate_original_indices_cp.size
        if num_unique_candidates == 0:
            print(f"Warning: Zero unique candidates found for query {q_idx} after concatenation/unique.")
            continue

        # --- Fetch candidate vectors (GPU indexing) ---
        # This is potentially the most memory-intensive step if many candidates are gathered
        try:
             # Ensure indices are valid before fetching
             max_idx = cp.max(unique_candidate_original_indices_cp)
             if max_idx >= N_A:
                 print(f"ERROR: Invalid candidate index {max_idx} >= N_A ({N_A}) found for query {q_idx}. Clamping indices.")
                 # Option 1: Clamp (might return wrong neighbors)
                 # unique_candidate_original_indices_cp = cp.clip(unique_candidate_original_indices_cp, 0, N_A - 1)
                 # Option 2: Filter (safer but might reduce candidates)
                 valid_mask = unique_candidate_original_indices_cp < N_A
                 unique_candidate_original_indices_cp = unique_candidate_original_indices_cp[valid_mask]
                 if unique_candidate_original_indices_cp.size == 0:
                     print(f"ERROR: No valid candidates left after filtering for query {q_idx}")
                     continue # Skip query if no valid candidates remain
             candidate_vectors_cp = A_cp[unique_candidate_original_indices_cp] # Shape (num_unique_candidates, D)

        except cp.cuda.memory.OutOfMemoryError as e:
            print(f"OOM Error fetching candidate vectors: Query {q_idx}, Num unique candidates={num_unique_candidates}, D={D}")
            mem_gb = num_unique_candidates * D * 4 / (1024**3)
            print(f"Estimated memory for candidate vectors: {mem_gb:.2f} GB")
            # Options: Skip query, try smaller batch, or raise error
            print("Skipping query due to OOM.")
            continue # Skip this query
        except IndexError as e:
             # This catch is less likely now with the check above, but good practice
             print(f"Error indexing A_cp at query {q_idx}. Max index: {cp.max(unique_candidate_original_indices_cp)}, N_A: {N_A}. Error: {e}")
             continue
        except Exception as e:
            print(f"Error fetching candidate vectors for query {q_idx}: {e}")
            continue


        # --- Calculate exact distances to candidates (vectorized on GPU) ---
        try:
            query_candidate_dists_sq = pairwise_l2_squared_cupy(query_cp, candidate_vectors_cp) # Shape (1, num_unique_candidates)
        except cp.cuda.memory.OutOfMemoryError as e:
            print(f"OOM Error calculating query-candidate distances: Query {q_idx}, Num candidates={num_unique_candidates}")
            mem_gb = num_unique_candidates * 4 / (1024**3) # Distance vector
            print(f"Estimated memory for distance vector: {mem_gb:.2f} GB")
            print("Skipping query due to OOM.")
            continue
        except Exception as e:
            print(f"Error calculating query-candidate distances for query {q_idx}: {e}")
            continue


        # --- Find top K_final among candidates (vectorized on GPU) ---
        # Determine the actual number of neighbors we can return for this query
        actual_k_final = min(K_final, num_unique_candidates)

        if actual_k_final > 0:
            try:
                # Get indices of K_final smallest distances *within the unique candidate subset*
                # Need k < N for argpartition
                final_k_partition = min(actual_k_final, num_unique_candidates - 1) if num_unique_candidates > 0 else 0
                if final_k_partition < 0: final_k_partition = 0

                if actual_k_final >= num_unique_candidates: # If K includes all candidates
                    topk_relative_indices = cp.argsort(query_candidate_dists_sq[0])[:actual_k_final]
                else:
                     # Ensure argpartition is called only if num_unique_candidates > 0
                     if num_unique_candidates > 0:
                          topk_relative_indices = cp.argpartition(query_candidate_dists_sq[0], final_k_partition)[:actual_k_final]
                     else: # Should not happen if actual_k_final > 0, but safety
                          topk_relative_indices = cp.empty((0,), dtype=cp.int64)

                # Indices relative to unique_candidate_original_indices_cp

                # Get the distances corresponding to these top k relative indices
                topk_distances_sq = query_candidate_dists_sq[0, topk_relative_indices]

                # Sort these K_final results by distance to ensure order
                sort_order = cp.argsort(topk_distances_sq)

                # Apply the sort order to get the final relative indices and distances
                final_topk_relative_indices = topk_relative_indices[sort_order]
                final_topk_distances_sq = topk_distances_sq[sort_order]

                # Map relative indices back to the original database indices from A_cp
                final_topk_original_indices = unique_candidate_original_indices_cp[final_topk_relative_indices]

                # Store results for this query in the pre-allocated arrays
                all_indices_cp[q_idx, :actual_k_final] = final_topk_original_indices
                all_distances_sq_cp[q_idx, :actual_k_final] = final_topk_distances_sq

            except Exception as e:
                print(f"Error during top-K selection/sorting for query {q_idx}: {e}")
                # Leave results as -1/inf for this query

    # --- Final Synchronization and Timing ---
    cp.cuda.Stream.null.synchronize() # Sync after all queries are processed
    search_time = time.time() - search_start_time
    print(f"ANN search time: {search_time:.4f} seconds")
    if search_time > 0 and Q > 0: print(f"-> Throughput: {Q / search_time:.2f} queries/sec")

    # Return original data point indices and their SQUARED L2 distances
    return all_indices_cp, all_distances_sq_cp, build_time_total, search_time


# ============================================================================
# Brute-Force k-NN (CuPy version - MODIFIED WITH QUERY BATCHING)
# ============================================================================
def cupy_knn_bruteforce(N_A, D, A_cp, X_cp, K, batch_size_q=256): # Add batch_size_q parameter
    """
    Finds the K nearest neighbors using brute-force pairwise SQUARED L2 distance (CuPy).
    Uses query batching to handle large Q * N distance matrices.
    Returns original indices (int64) and SQUARED L2 distances (float32).
    Handles K > N_A by padding results.
    """
    if not cupy_device_ok: raise RuntimeError("CuPy device not available.")
    # --- Input Validation ---
    if not isinstance(A_cp, cp.ndarray): A_cp = cp.asarray(A_cp, dtype=cp.float32)
    elif A_cp.dtype != cp.float32: A_cp = A_cp.astype(cp.float32)
    if not isinstance(X_cp, cp.ndarray): X_cp = cp.asarray(X_cp, dtype=cp.float32)
    elif X_cp.dtype != cp.float32: X_cp = X_cp.astype(cp.float32)

    if A_cp.ndim != 2: raise ValueError(f"Database A_cp must be 2D (N, D), got shape {A_cp.shape}")
    if X_cp.ndim != 2: raise ValueError(f"Queries X_cp must be 2D (Q, D), got shape {X_cp.shape}")

    actual_N_A, actual_D = A_cp.shape
    Q, query_D = X_cp.shape

    if query_D != actual_D: raise ValueError(f"Dimension mismatch: A_cp D={actual_D}, X_cp D={query_D}")

    # Use actual N_A
    N_A = actual_N_A

    # Handle empty database or query set
    if N_A == 0:
         print("Warning: Brute force called with empty database A_cp.")
         return cp.full((Q, K), -1, dtype=cp.int64), cp.full((Q, K), cp.inf, dtype=cp.float32)
    if Q == 0:
         print("Warning: Brute force called with empty query set X_cp.")
         return cp.empty((0, K), dtype=cp.int64), cp.empty((0, K), dtype=cp.float32)

    if not K > 0: raise ValueError("K must be positive")

    # Adjust K if it's larger than the number of data points
    effective_K = min(K, N_A)
    if effective_K != K:
         print(f"Note: Brute force K={K} requested > N_A={N_A}. Using K={effective_K}.")

    if effective_K == 0: # Handle K=0 or effective_K=0 case
         return cp.empty((Q, 0), dtype=cp.int64), cp.empty((Q, 0), dtype=cp.float32)

    print(f"Running k-NN Brute Force (CuPy, Batched): Q={Q}, N={N_A}, D={actual_D}, K={effective_K}, BatchSize={batch_size_q}")
    start_time = time.time()

    # Pre-allocate result arrays
    all_topk_indices_cp = cp.full((Q, effective_K), -1, dtype=cp.int64)
    all_topk_distances_sq_cp = cp.full((Q, effective_K), cp.inf, dtype=cp.float32)

    # --- Batch Processing Loop ---
    for q_start in range(0, Q, batch_size_q):
        q_end = min(q_start + batch_size_q, Q)
        batch_q_indices = slice(q_start, q_end) # Slice for indexing results
        X_batch_cp = X_cp[batch_q_indices]     # Current batch of queries
        current_batch_size = X_batch_cp.shape[0]
        if current_batch_size == 0: continue

        # print(f"  Processing query batch {q_start}-{q_end-1}...") # Optional progress

        # Calculate SQUARED L2 distances for the current batch
        try:
            batch_distances_sq = pairwise_l2_squared_cupy(X_batch_cp, A_cp) # Shape (current_batch_size, N_A)
        except cp.cuda.memory.OutOfMemoryError as e:
            batch_mem_gb = current_batch_size * N_A * 4 / (1024**3)
            print(f"OOM Error during Brute Force batch distance calculation:")
            print(f"  Batch Q={current_batch_size}, N={N_A}. Estimated matrix memory: {batch_mem_gb:.2f} GB")
            print(f"  Try reducing batch_size_q (current={batch_size_q}).")
            raise e # Re-raise
        except Exception as e:
            print(f"Error during Brute Force batch distance calculation: {e}")
            raise e


        # Find top K for the current batch
        # Need k < N for argpartition
        k_partition = min(effective_K, N_A - 1) if N_A > 0 else 0
        if k_partition < 0: k_partition = 0

        batch_topk_indices = None
        batch_topk_distances_sq = None

        try:
            if effective_K >= N_A: # If K includes all points, just sort all
                batch_topk_indices = cp.argsort(batch_distances_sq, axis=1)[:, :effective_K]
            else:
                 # Ensure N_A > 0 before calling argpartition
                 if N_A > 0:
                      topk_indices_unstructured = cp.argpartition(batch_distances_sq, k_partition, axis=1)[:, :effective_K]
                      topk_distances_sq_unstructured = cp.take_along_axis(batch_distances_sq, topk_indices_unstructured, axis=1)
                      sorted_order_in_k = cp.argsort(topk_distances_sq_unstructured, axis=1)
                      batch_topk_indices = cp.take_along_axis(topk_indices_unstructured, sorted_order_in_k, axis=1)
                 else: # Should not happen if N_A=0 check passed
                      batch_topk_indices = cp.empty((current_batch_size, 0), dtype=cp.int64)

            # Retrieve the final sorted distances
            if batch_topk_indices is not None and batch_topk_indices.size > 0:
                 batch_topk_distances_sq = cp.take_along_axis(batch_distances_sq, batch_topk_indices, axis=1)
            elif effective_K == 0:
                 batch_topk_distances_sq = cp.empty((current_batch_size, 0), dtype=cp.float32)
            else: # Handle unexpected empty indices
                 print(f"Warning: batch_topk_indices empty/None in batch {q_start}-{q_end-1}")
                 batch_topk_distances_sq = cp.full((current_batch_size, effective_K), cp.inf, dtype=cp.float32)
                 if batch_topk_indices is None: batch_topk_indices = cp.full((current_batch_size, effective_K), -1, dtype=cp.int64)

            # Store batch results in the main arrays
            all_topk_indices_cp[batch_q_indices] = batch_topk_indices
            all_topk_distances_sq_cp[batch_q_indices] = batch_topk_distances_sq

            # Optional: Clear intermediate batch results to free memory sooner
            del batch_distances_sq, batch_topk_indices, batch_topk_distances_sq
            if 'topk_indices_unstructured' in locals(): del topk_indices_unstructured
            if 'topk_distances_sq_unstructured' in locals(): del topk_distances_sq_unstructured
            if 'sorted_order_in_k' in locals(): del sorted_order_in_k
            # cp.get_default_memory_pool().free_all_blocks() # Use cautiously, can impact performance


        except Exception as e:
            print(f"Error during Brute Force batch top-K ({q_start}-{q_end-1}): {e}")
            # Decide how to handle: continue with potentially missing batch results or raise error?
            # For recall, probably better to raise the error
            raise e
    # --- End Batch Loop ---

    cp.cuda.Stream.null.synchronize()
    end_time = time.time()
    print(f"k-NN Brute Force (CuPy, Batched) total computation time: {end_time - start_time:.4f} seconds")

    # Pad results if original K > effective_K (i.e., K > N_A)
    if K > effective_K:
        pad_width = K - effective_K
        indices_pad = cp.full((Q, pad_width), -1, dtype=cp.int64)
        dists_pad = cp.full((Q, pad_width), cp.inf, dtype=cp.float32)
        all_topk_indices_cp = cp.hstack((all_topk_indices_cp, indices_pad))
        all_topk_distances_sq_cp = cp.hstack((all_topk_distances_sq_cp, dists_pad))

    return all_topk_indices_cp.astype(cp.int64), all_topk_distances_sq_cp.astype(cp.float32)

if __name__ == "__main__":
    if not cupy_device_ok:
        print("CuPy device not available. Exiting.")
        exit()

    # --- Parameters ---
    N_data = 1000000    # Number of database points
    Dim = 128           # Dimension
    N_queries = 10000      # Number of query points
    K_final_neighbors = 10 # Final number of neighbors to find (Output K)

    # ANN Parameters based on new pseudocode interpretation
    num_clusters_kmeans = 1000  # K for KMeans (Step 1)
    num_clusters_probe = 50    # K1 for finding probe clusters (Step 2)
                               # K2 from pseudocode is ignored based on interpretation

    kmeans_max_iters = 50      # Max iterations for KMeans

    # Recall threshold
    RECALL_THRESHOLD = 0.70

    print("="*50)
    print("Generating Test Data (CuPy)...")
    print(f"N={N_data}, D={Dim}, Q={N_queries}, K_final={K_final_neighbors}")
    print(f"ANN Params: num_clusters={num_clusters_kmeans}, nprobe={num_clusters_probe}")
    print("="*50)
    A_data_cp = None # Define outside try block
    X_queries_cp = None
    try:
        # Generate data directly on GPU
        print("Allocating memory for A_data_cp...")
        A_data_cp = cp.random.random((N_data, Dim), dtype=cp.float32)
        print("Allocating memory for X_queries_cp...")
        X_queries_cp = cp.random.random((N_queries, Dim), dtype=cp.float32)

        # Optional: Normalize data? Often helps distance-based methods.
        # print("Normalizing data...")
        # A_data_cp /= cp.linalg.norm(A_data_cp, axis=1, keepdims=True) + 1e-9
        # X_queries_cp /= cp.linalg.norm(X_queries_cp, axis=1, keepdims=True) + 1e-9

        cp.cuda.Stream.null.synchronize() # Wait for operations
        print("Data generated successfully on GPU.")
        # Check memory usage (example)
        pool = cp.get_default_memory_pool()
        print(f"CuPy Memory Usage: {pool.used_bytes() / (1024**3):.2f} GB used / {pool.total_bytes() / (1024**3):.2f} GB total")

    except cp.cuda.memory.OutOfMemoryError as e:
        print(f"\nError: Out of GPU memory during data generation: {e}")
        print("Try reducing N_data or Dim.")
        # Clean up memory if possible
        del A_data_cp
        del X_queries_cp
        cp.get_default_memory_pool().free_all_blocks()
        exit()
    except Exception as e:
        print(f"\nError generating data: {e}")
        exit()

    # --- Run ANN (User Pseudocode - IVF-like) ---
    print("\n" + "="*50)
    print(f"Testing ANN (User Pseudocode - IVF-like)...")
    print("="*50)
    ann_indices = None # Define outside try block
    ann_dists_sq = None
    build_t = 0
    search_t = 0
    try:
        ann_indices, ann_dists_sq, build_t, search_t = ann_user_pseudocode_ivf_like(
            N_A=N_data, D=Dim, A_cp=A_data_cp, X_cp=X_queries_cp,
            K_final=K_final_neighbors,
            num_clusters=num_clusters_kmeans,
            num_clusters_to_probe=num_clusters_probe,
            max_kmeans_iters=kmeans_max_iters
            # Example of using precomputed:
            # precomputed_centroids=centroids_from_earlier,
            # precomputed_assignments=assignments_from_earlier
        )
        print("\nANN Results:")
        if ann_indices is not None:
            print(f"  Indices shape: {ann_indices.shape}") # Should be (N_queries, K_final_neighbors)
        if ann_dists_sq is not None:
            print(f"  Sq Distances shape: {ann_dists_sq.shape}")
        print(f"  Build Time: {build_t:.4f}s")
        print(f"  Search Time: {search_t:.4f}s")

    except cp.cuda.memory.OutOfMemoryError as e:
        print(f"\nError: Out of GPU memory during ANN execution: {e}")
        print("Consider reducing N_data, num_clusters, or num_clusters_to_probe.")
        ann_indices = None # Prevent recall calculation
    except Exception as e:
        print(f"\nError during ANN execution: {e}")
        import traceback
        traceback.print_exc()
        ann_indices = None # Prevent recall calculation
    finally:
         # Intermediate cleanup if needed, although full cleanup happens later
         pass


    # --- Run Brute-Force KNN for Ground Truth ---
    true_knn_indices = None # Define outside try/except
    if ann_indices is not None: # Only run if ANN succeeded and needed for recall
        print("\n" + "="*50)
        print(f"Calculating Ground Truth (Brute-Force k-NN)...")
        print("="*50)
        try:
            true_knn_indices, true_knn_dists_sq = cupy_knn_bruteforce(
                N_A=N_data, D=Dim, A_cp=A_data_cp, X_cp=X_queries_cp, K=K_final_neighbors
            )
            print("\nGround Truth Results:")
            if true_knn_indices is not None:
                print(f"  Indices shape: {true_knn_indices.shape}")
            if true_knn_dists_sq is not None:
                print(f"  Sq Distances shape: {true_knn_dists_sq.shape}")

        except cp.cuda.memory.OutOfMemoryError as e:
            print(f"\nError: Out of GPU memory during Brute Force k-NN: {e}")
            print("Try reducing N_data or N_queries.")
            true_knn_indices = None # Prevent recall calculation
        except Exception as e:
            print(f"\nError during Brute Force k-NN execution: {e}")
            import traceback
            traceback.print_exc()
            true_knn_indices = None # Prevent recall calculation
        finally:
             # Intermediate cleanup if needed
             if 'true_knn_dists_sq' in locals(): del true_knn_dists_sq

    # --- Calculate Recall ---
    if ann_indices is not None and true_knn_indices is not None:
        print("\n" + "="*50)
        print(f"Calculating Recall@{K_final_neighbors}...")
        print("="*50)

        try:
            # Comparison can be done on GPU for speed if needed, but CPU is simpler here
            print("Transferring indices to CPU for comparison...")
            start_recall_calc = time.time()
            # Ensure arrays are not None before transfer
            ann_indices_np = cp.asnumpy(ann_indices) if ann_indices is not None else None
            true_indices_np = cp.asnumpy(true_knn_indices) if true_knn_indices is not None else None
            print(f"Transfer time: {time.time() - start_recall_calc:.4f}s")

            if ann_indices_np is None or true_indices_np is None:
                 raise ValueError("Cannot compare recall, indices arrays are None.")

            total_intersect = 0
            if N_queries > 0 and K_final_neighbors > 0: # Proceed only if comparison is meaningful
                for i in range(N_queries):
                    # Ignore potential -1 placeholders in ANN results if K > num_candidates found
                    ann_set = set(idx for idx in ann_indices_np[i] if idx >= 0)
                    # True KNN results might have -1 if K > N_A was handled by padding
                    true_set = set(idx for idx in true_indices_np[i] if idx >= 0)

                    # Correct handling if true_set is empty (e.g. K>N_A and padding was the only result)
                    if not true_set:
                        # If true neighbors set is empty, intersection is 0 unless ann_set is also empty
                        if not ann_set:
                             total_intersect += 0 # Or handle differently if empty match matters
                        else:
                             total_intersect += 0
                    else:
                        total_intersect += len(ann_set.intersection(true_set))

                # Recall = (Total Intersecting Neighbors) / (Total True Neighbors Expected)
                # Total expected = N_queries * actual number of true neighbors requested (min(K, N_A))
                expected_neighbors_per_query = min(K_final_neighbors, N_data)
                if expected_neighbors_per_query == 0:
                     avg_recall = 1.0 if total_intersect == 0 else 0.0 # Define recall=1 if K=0? Or avoid division.
                     print("\nRecall @ 0: Undefined or 100% by convention.")
                else:
                     denominator = N_queries * expected_neighbors_per_query
                     avg_recall = total_intersect / denominator if denominator > 0 else 1.0 # Avoid division by zero if N_queries=0

                     print(f"\nAverage Recall @ {K_final_neighbors} (vs {expected_neighbors_per_query} possible): {avg_recall:.4f} ({avg_recall:.2%})")

                     if avg_recall >= RECALL_THRESHOLD:
                         print(f"Recall meets the threshold ({RECALL_THRESHOLD:.2%}). Result CORRECT.")
                     else:
                         print(f"Recall is BELOW the threshold ({RECALL_THRESHOLD:.2%}). Result INCORRECT.")
                         print("Suggestions to improve recall:")
                         print(f" - Increase `num_clusters_to_probe` (currently {num_clusters_probe}). More probes = higher chance of finding true neighbors, but slower search.")
                         print(f" - Increase `num_clusters_kmeans` (currently {num_clusters_kmeans}). Finer clustering might isolate neighbors better.")
                         print(" - Ensure data preprocessing (e.g., normalization) is appropriate.")
            else:
                print("\nCannot calculate recall (N_queries=0 or K_final_neighbors=0).")

        except Exception as e:
            print(f"\nError during Recall calculation: {e}")
            import traceback
            traceback.print_exc()

    elif ann_indices is None:
         print("\nSkipping Recall calculation because ANN execution failed or produced None.")
    elif true_knn_indices is None:
         print("\nSkipping Recall calculation because Brute Force k-NN failed or produced None.")

    print("\n--- Execution Finished ---")
