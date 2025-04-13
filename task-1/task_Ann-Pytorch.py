import torch
# import triton # REMOVED
# import triton.language as tl # REMOVED
import time
import traceback # For error printing
import math # For ceil
import numpy as np # For recall calculation comparison
import gc # For garbage collection

# --- Device Setup ---
if not torch.cuda.is_available():
    print("CUDA not available, exiting.")
    exit()
device = torch.device("cuda:0")
print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")

# ============================================================================
# Helper Functions (Same as before)
# ============================================================================
def _prepare_tensors(*tensors, target_device=device):
    """Ensure tensors are float32, contiguous, and on the correct device."""
    prepared = []
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            try: t = torch.tensor(t, dtype=torch.float32, device=target_device)
            except Exception as e: raise TypeError(f"Failed conversion: {e}")
        if t.device != target_device: t = t.to(target_device)
        if t.dtype != torch.float32: t = t.to(dtype=torch.float32)
        if not t.is_contiguous(): t = t.contiguous()
        prepared.append(t)
    # Return single tensor directly if only one was passed
    return prepared[0] if len(prepared) == 1 else prepared

# ============================================================================
# Distance Functions (Pure PyTorch)
# ============================================================================

def distance_l2_squared_pytorch_native(X, A):
    """
    Computes pairwise SQUARED L2 distances using native PyTorch functions.
    Leverages torch.cdist for optimized pairwise distance calculation.

    Args:
        X (torch.Tensor): Queries (Q, D), float32, GPU.
        A (torch.Tensor): Database points (N, D), float32, GPU.

    Returns:
        torch.Tensor: Squared distances (Q, N), float32, GPU.
    """
    X_prep, A_prep = _prepare_tensors(X, A)
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    if D != D_A: raise ValueError(f"Dimension mismatch: X({D}) vs A({D_A})")
    if Q == 0 or N == 0: return torch.empty((Q, N), dtype=torch.float32, device=device)

    try:
        # torch.cdist computes standard L2 distance (sqrt of sum of squares)
        # It's generally highly optimized and handles batching internally.
        l2_dist = torch.cdist(X_prep, A_prep, p=2)

        # Square the result to get squared L2 distance
        dist_sq = l2_dist.square_() # In-place square
        # Clamp shouldn't be necessary if cdist is numerically stable, but added for safety
        dist_sq.clamp_(min=0.0)
        return dist_sq

    except RuntimeError as e: # Catch potential CUDA OOM errors from cdist
        print(f"Error in distance_l2_squared_pytorch_native (shapes X:{X.shape}, A:{A.shape}): {e}")
        if "CUDA out of memory" in str(e):
             # cdist might use less peak memory than matmul approach, but can still fail
             print(f"torch.cdist failed. Q={Q}, N={N}, D={D}")
        raise e # Re-raise

# ============================================================================
# K-Means Implementation (Pure PyTorch)
# ============================================================================

def our_kmeans_pytorch(N_A, D, A, K, max_iters=100, tol=1e-4, verbose=False):
    """
    Performs K-means clustering using pure PyTorch operations.
    Uses torch.cdist for assignment and scatter_add_ for update.

    Args:
        N_A (int): Number of data points (inferred).
        D (int): Dimension (inferred).
        A (torch.Tensor): Data points (N_A, D), float32, on GPU.
        K (int): Number of clusters.
        max_iters (int): Maximum iterations.
        tol (float): Convergence tolerance for centroid movement.
        verbose (bool): If True, prints iteration details.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - centroids (torch.Tensor): Final centroids (K_actual, D), float32, GPU.
            - assignments (torch.Tensor): Final assignments (N_A,), int64, GPU.
    """
    A_prep = _prepare_tensors(A)
    actual_N_A, actual_D = A_prep.shape
    N_A = actual_N_A
    D = actual_D

    if not (K > 0): raise ValueError("K must be positive.")
    if K > N_A:
        print(f"Warning: Requested K ({K}) > N_A ({N_A}). Using K={N_A}.")
        K = N_A
    if N_A == 0:
         print("Warning: KMeans called with empty data A.")
         return torch.empty((0, D), dtype=torch.float32, device=device), \
                torch.empty((0,), dtype=torch.int64, device=device)
    if K == 0:
         return torch.empty((0, D), dtype=torch.float32, device=device), \
                torch.empty((N_A,), dtype=torch.int64, device=device)

    print(f"Running K-Means (Pure PyTorch): N={N_A}, D={D}, K={K}")
    start_time_total = time.time()

    # --- Initialization ---
    initial_indices = torch.randperm(N_A, device=device)[:K]
    centroids = A_prep[initial_indices].clone() # Shape (K, D)
    assignments = torch.empty(N_A, dtype=torch.int64, device=device) # Final assignments type

    # --- Timing variables ---
    assignment_times = []
    update_times = []

    for i in range(max_iters):
        t_iter_start = time.time()
        old_centroids = centroids.clone()

        # --- 1. Assignment Step (PyTorch cdist + argmin) ---
        t_assign_start = time.time()
        try:
            # Calculate squared L2 distances using the native PyTorch function
            all_dist_sq = distance_l2_squared_pytorch_native(A_prep, centroids) # Shape (N_A, K)
            # Find the index of the minimum distance (nearest centroid)
            new_assignments_int64 = torch.argmin(all_dist_sq, dim=1) # Returns int64
        except RuntimeError as e: # Catch OOM from distance calc or argmin
            print(f"OOM Error during KMeans assignment step (Iter {i+1}): Calculating ({N_A}, {K}) distance matrix.")
            mem_gb = N_A * K * 4 / (1024**3)
            print(f"Estimated memory for distance matrix: {mem_gb:.2f} GB")
            raise e
        except Exception as e:
            print(f"Error during KMeans assignment step (Iter {i+1}): {e}")
            raise e

        torch.cuda.synchronize(device=device) # Wait for argmin
        t_assign_end = time.time()
        assignment_times.append(t_assign_end - t_assign_start)

        assignments = new_assignments_int64 # Update assignments

        # --- 2. Update Step (PyTorch scatter_add_) ---
        t_update_start = time.time()
        new_sums = torch.zeros_like(centroids) # (K, D)
        cluster_counts = torch.zeros(K, dtype=torch.float32, device=device) # (K,)
        idx_expand = assignments.unsqueeze(1).expand(-1, D) # Shape (N_A, D)
        new_sums.scatter_add_(dim=0, index=idx_expand, src=A_prep)
        cluster_counts.scatter_add_(dim=0, index=assignments, src=torch.ones_like(assignments, dtype=torch.float32))
        torch.cuda.synchronize(device=device) # Wait for scatter ops
        final_counts_safe = cluster_counts.clamp(min=1.0)
        new_centroids = new_sums / final_counts_safe.unsqueeze(1)
        empty_cluster_mask = (cluster_counts == 0)
        num_empty = torch.sum(empty_cluster_mask).item()
        if num_empty > 0:
            new_centroids[empty_cluster_mask] = old_centroids[empty_cluster_mask]
            if num_empty == K:
                 print(f"Warning: Iter {i+1}, ALL clusters are empty. Stopping iteration.")
                 centroids = old_centroids; break
        if not torch.all(torch.isfinite(new_centroids)):
             print(f"Warning: Iter {i+1}, found non-finite values in new_centroids. Replacing with old.")
             non_finite_mask = ~torch.isfinite(new_centroids)
             new_centroids[non_finite_mask] = old_centroids[non_finite_mask]
        t_update_end = time.time()
        update_times.append(t_update_end - t_update_start)

        # --- Check Convergence ---
        centroid_diff = torch.linalg.norm(new_centroids - old_centroids)
        centroids = new_centroids # Update for next iteration

        if verbose and ((i+1) % 10 == 0 or centroid_diff < tol or i == max_iters -1):
            print(f"  Iter {i+1}/{max_iters} | Centroid Diff: {centroid_diff:.4f} | Assign Time: {assignment_times[-1]:.4f}s | Update Time: {update_times[-1]:.4f}s")

        if centroid_diff < tol:
            print(f"Converged after {i+1} iterations (centroid movement < {tol}).")
            break

    if i == max_iters - 1: print(f"Reached max iterations ({max_iters}).")
    total_time = time.time() - start_time_total
    print(f"Total K-Means time: {total_time:.4f}s")

    # --- Basic Benchmarking Output ---
    if assignment_times: print(f"Avg Assign Step Time (PyTorch): {sum(assignment_times)/len(assignment_times):.6f}s")
    if update_times: print(f"Avg Update Step Time (PyTorch): {sum(update_times)/len(update_times):.6f}s")

    final_counts = torch.bincount(assignments, minlength=K)
    actual_k = torch.sum(final_counts > 0).item()
    print(f"K-Means finished. Found {actual_k} non-empty clusters out of {K} requested.")

    return centroids, assignments # Return final assignments (int64)

# ============================================================================
# ANN Function (Pure PyTorch Version of IVF-like)
# ============================================================================

def ann_ivf_like_pytorch( # Renamed from _triton
    N_A, D, A, X, K_final,
    num_clusters, num_clusters_to_probe,
    max_kmeans_iters=100,
    precomputed_centroids=None,
    precomputed_assignments=None,
    verbose_kmeans=False
    ):
    """
    Performs ANN search based on user's pseudocode using pure PyTorch.
    Finds K_final nearest DATA POINTS.

    Args:
        (Same as before, but expects PyTorch Tensors)

    Returns:
        tuple[torch.Tensor, torch.Tensor, float, float]:
            - all_indices (torch.Tensor): Indices (in A) of K_final nearest neighbors (Q, K_final). Int64.
            - all_distances_sq (torch.Tensor): **Squared** L2 distances (Q, K_final). Float32.
            - build_time (float): Time for K-Means + Inverted Index.
            - search_time (float): Time for searching all queries.
    """
    print("--- Starting Pure PyTorch ANN ---")
    # --- Input Validation & Data Prep (Same as before) ---
    A_prep = _prepare_tensors(A)
    X_prep = _prepare_tensors(X)
    actual_N_A, actual_D = A_prep.shape; Q, query_D = X_prep.shape
    N_A = actual_N_A; D = actual_D
    if query_D != D: raise ValueError(f"Dimension mismatch: A D={D}, X D={query_D}")
    if N_A == 0: raise ValueError("Database A cannot be empty.")
    if Q == 0: print("Warning: Query set X is empty.");
    if not (K_final > 0): raise ValueError("K_final must be positive")
    if not (num_clusters > 0): raise ValueError("num_clusters must be positive")
    if not (num_clusters_to_probe > 0): raise ValueError("num_clusters_to_probe must be positive")
    print(f"Running ANN (Pure PyTorch IVF-like): Q={Q}, N={N_A}, D={D}, K_final={K_final}")
    print(f"Params: num_clusters={num_clusters}, nprobe={num_clusters_to_probe}")
    build_time_total = 0.0
    build_start_time = time.time()

    # --- Step 1: K-Means Clustering & Index Setup ---
    assignments = None; centroids = None; kmeans_run_time = 0.0
    if precomputed_centroids is not None and precomputed_assignments is not None:
        # --- Using precomputed (Same logic as before) ---
        print("Using precomputed centroids and assignments.")
        centroids = _prepare_tensors(precomputed_centroids)
        assignments = _prepare_tensors(precomputed_assignments).to(torch.int64)
        # (Validation logic same as before) ...
        if centroids.ndim != 2 or centroids.shape[1] != D: raise ValueError("Invalid precomputed centroids shape/dim.")
        if assignments.ndim != 1 or assignments.shape[0] != N_A: raise ValueError("Invalid precomputed assignments shape.")
        actual_num_clusters = centroids.shape[0]
        if actual_num_clusters < num_clusters: print(f"Warning: Provided {actual_num_clusters} centroids < requested {num_clusters}.")
        elif actual_num_clusters > num_clusters:
            print(f"Warning: Using first {num_clusters} of {actual_num_clusters} provided centroids.")
            centroids = centroids[:num_clusters]
            if assignments.max() >= num_clusters: raise ValueError("Assignments index out of bounds after truncating centroids.")
        num_clusters = centroids.shape[0] # Use actual number
    elif precomputed_centroids is not None or precomputed_assignments is not None:
        raise ValueError("Provide both or neither of precomputed centroids/assignments.")
    else:
        # --- Run Pure PyTorch KMeans ---
        print("Running KMeans (Pure PyTorch)...")
        kmeans_start = time.time()
        # *** Call the new PyTorch KMeans function ***
        centroids, assignments = our_kmeans_pytorch( # <--- CHANGED HERE
            N_A, D, A_prep, num_clusters, max_iters=max_kmeans_iters, verbose=verbose_kmeans
        )
        kmeans_run_time = time.time() - kmeans_start
        actual_num_clusters = centroids.shape[0]
        if actual_num_clusters < num_clusters: print(f"Note: KMeans used K={actual_num_clusters}.")
        num_clusters = actual_num_clusters # Use actual number

    # --- Error Handling & Parameter Adjustment (Same as before) ---
    if num_clusters == 0 or centroids is None or assignments is None:
        print("Error: No clusters found or provided. Cannot proceed.")
        # (Return empty logic same as before) ...
        empty_indices = torch.full((Q, K_final), -1, dtype=torch.int64, device=device)
        empty_dists = torch.full((Q, K_final), float('inf'), dtype=torch.float32, device=device)
        build_time_total = time.time() - build_start_time
        return empty_indices, empty_dists, build_time_total, 0.0
    num_clusters_to_probe = min(num_clusters_to_probe, num_clusters)
    if num_clusters_to_probe <= 0:
        print("Error: num_clusters_to_probe is 0. Cannot proceed.")
        # (Return empty logic same as before) ...
        empty_indices = torch.full((Q, K_final), -1, dtype=torch.int64, device=device)
        empty_dists = torch.full((Q, K_final), float('inf'), dtype=torch.float32, device=device)
        build_time_total = time.time() - build_start_time
        return empty_indices, empty_dists, build_time_total, 0.0

    # --- Build Inverted Index (Same PyTorch logic as before) ---
    print(f"Building Inverted Index (PyTorch) for {num_clusters} clusters...")
    invidx_start_time = time.time()
    # (Inverted index logic using argsort, unique_consecutive, cumsum same as before) ...
    original_indices = torch.arange(N_A, dtype=torch.int64, device=device)
    assignments = assignments.to(torch.int64)
    sort_permutation = torch.argsort(assignments)
    inv_idx_values = original_indices[sort_permutation]
    sorted_assignments = assignments[sort_permutation]
    unique_clusters, cluster_counts = torch.unique_consecutive(sorted_assignments, return_inverse=False, return_counts=True)
    cluster_starts = torch.zeros_like(cluster_counts)
    cluster_starts[1:] = torch.cumsum(cluster_counts[:-1], dim=0)
    full_inv_idx_starts = torch.full((num_clusters,), -1, dtype=torch.int64, device=device)
    full_inv_idx_counts = torch.zeros((num_clusters,), dtype=torch.int64, device=device)
    valid_unique_mask = unique_clusters < num_clusters
    valid_unique_clusters = unique_clusters[valid_unique_mask]
    if valid_unique_clusters.numel() > 0:
         full_inv_idx_starts[valid_unique_clusters] = cluster_starts[valid_unique_mask]
         full_inv_idx_counts[valid_unique_clusters] = cluster_counts[valid_unique_mask]
    else: print("Warning: No valid unique clusters found during inverted index build.")
    invidx_end_time = time.time()
    build_time_total = kmeans_run_time + (invidx_end_time - invidx_start_time)
    print(f"Index build time (Total): {build_time_total:.4f}s (KMeans: {kmeans_run_time:.4f}s, InvIdx: {invidx_end_time - invidx_start_time:.4f}s)")

    # --- Search Phase ---
    search_start_time = time.time()
    all_indices = torch.full((Q, K_final), -1, dtype=torch.int64, device=device)
    all_distances_sq = torch.full((Q, K_final), float('inf'), dtype=torch.float32, device=device)
    if Q == 0: return all_indices, all_distances_sq, build_time_total, 0.0 # Handle empty query

    # --- Step 2: Find nearest clusters (Use PyTorch Distance) ---
    print("Calculating query-centroid distances (PyTorch)...")
    try:
        # *** Use the native PyTorch distance function ***
        all_query_centroid_dists_sq = distance_l2_squared_pytorch_native(X_prep, centroids) # <--- CHANGED HERE
    except RuntimeError as e: print(f"Error calculating query-centroid distances: {e}"); raise e
    print("Finding nearest clusters...")
    try: _, all_nearest_cluster_indices = torch.topk(all_query_centroid_dists_sq, k=num_clusters_to_probe, dim=1, largest=False, sorted=False)
    except Exception as e: print(f"Error during torch.topk for nearest clusters: {e}"); raise e

    # --- Step 3 & 4: Gather/Search (Same logic, but uses PyTorch distance internally) ---
    print(f"Searching {Q} queries...")
    for q_idx in range(Q):
        # --- Candidate Gathering (Same logic) ---
        query = X_prep[q_idx:q_idx+1]
        probed_cluster_indices = all_nearest_cluster_indices[q_idx]
        # (Gathering candidates using lists and torch.cat same as before) ...
        selected_starts = full_inv_idx_starts[probed_cluster_indices]
        selected_counts = full_inv_idx_counts[probed_cluster_indices]
        valid_probe_mask = selected_starts >= 0
        if not torch.any(valid_probe_mask): continue
        valid_starts = selected_starts[valid_probe_mask]; valid_counts = selected_counts[valid_probe_mask]
        num_valid_probes = valid_starts.numel()
        candidate_indices_list = []
        for i in range(num_valid_probes):
             start = valid_starts[i].item(); count = valid_counts[i].item()
             if count > 0: candidate_indices_list.append(inv_idx_values[start : start + count])
        if not candidate_indices_list: continue
        try: # Add try-except for potentially empty list cat
             candidate_original_indices = torch.cat(candidate_indices_list)
        except RuntimeError as e:
             if len(candidate_indices_list) == 0: continue # Skip if list became empty
             else: raise e # Re-raise other errors
        unique_candidate_original_indices = torch.unique(candidate_original_indices)
        num_unique_candidates = unique_candidate_original_indices.numel()
        if num_unique_candidates == 0: continue


        # --- Fetching Vectors (Same logic + error handling) ---
        try:
            max_idx = torch.max(unique_candidate_original_indices) # Check validity
            if max_idx >= N_A:
                 valid_cand_mask = unique_candidate_original_indices < N_A
                 unique_candidate_original_indices = unique_candidate_original_indices[valid_cand_mask]
                 num_unique_candidates = unique_candidate_original_indices.numel()
                 if num_unique_candidates == 0: continue
            candidate_vectors = A_prep[unique_candidate_original_indices]
        # (Error handling OOM, IndexError, etc. same as before) ...
        except RuntimeError as e: print(f"OOM fetching candidates (Query {q_idx}, {num_unique_candidates} candidates): {e}"); continue
        except IndexError as e: print(f"IndexError fetching candidates (Query {q_idx}): {e}"); continue
        except Exception as e: print(f"Error fetching candidates (Query {q_idx}): {e}"); continue


        # --- Distance Calculation (Use PyTorch Distance) ---
        try:
             # *** Use the native PyTorch distance function ***
             query_candidate_dists_sq = distance_l2_squared_pytorch_native(query, candidate_vectors) # <--- CHANGED HERE
        # (Error handling OOM, etc. same as before) ...
        except RuntimeError as e: print(f"OOM calculating query-candidate dists (Query {q_idx}, {num_unique_candidates} candidates): {e}"); continue
        except Exception as e: print(f"Error calculating query-candidate dists (Query {q_idx}): {e}"); continue


        # --- Top K Search (Same logic + error handling) ---
        actual_k_final = min(K_final, num_unique_candidates)
        if actual_k_final > 0:
            try:
                topk_dists_sq, topk_relative_indices = torch.topk(query_candidate_dists_sq[0], k=actual_k_final, largest=False, sorted=True)
                final_topk_original_indices = unique_candidate_original_indices[topk_relative_indices]
                all_indices[q_idx, :actual_k_final] = final_topk_original_indices
                all_distances_sq[q_idx, :actual_k_final] = topk_dists_sq
            except Exception as e: print(f"Error during top-K selection for query {q_idx}: {e}"); continue

    # --- Final Sync & Timing (Same logic) ---
    torch.cuda.synchronize(device=device)
    search_time = time.time() - search_start_time
    print(f"ANN search time: {search_time:.4f} seconds")
    if search_time > 0 and Q > 0: print(f"-> Throughput: {Q / search_time:.2f} queries/sec")

    return all_indices, all_distances_sq, build_time_total, search_time


# ============================================================================
# Brute-Force k-NN (Pure PyTorch Version - Batched)
# ============================================================================
def pytorch_knn_bruteforce_native(N_A, D, A, X, K, batch_size_q=256): # Renamed slightly
    """
    Finds the K nearest neighbors using pure PyTorch distance (torch.cdist)
    with query batching.
    Returns original indices (int64) and SQUARED L2 distances (float32).
    Handles K > N_A by padding results.
    """
    print(f"Running k-NN Brute Force (Pure PyTorch, Batched)...")
    A_prep = _prepare_tensors(A); X_prep = _prepare_tensors(X)
    actual_N_A, actual_D = A_prep.shape; Q, query_D = X_prep.shape
    N_A = actual_N_A; D = actual_D
    if query_D != D: raise ValueError(f"Dimension mismatch: A D={D}, X D={query_D}")
    # (Handle empty A, X, K=0 cases same as before) ...
    if N_A == 0: return torch.full((Q, K), -1, dtype=torch.int64, device=device), torch.full((Q, K), float('inf'), dtype=torch.float32, device=device)
    if Q == 0: return torch.empty((0, K), dtype=torch.int64, device=device), torch.empty((0, K), dtype=torch.float32, device=device)
    if not K > 0: raise ValueError("K must be positive")
    effective_K = min(K, N_A)
    if effective_K != K: print(f"Note: Brute force K={K} > N_A={N_A}. Using K={effective_K}.")
    if effective_K == 0: return torch.empty((Q, 0), dtype=torch.int64, device=device), torch.empty((0, 0), dtype=torch.float32, device=device)

    print(f"Params: Q={Q}, N={N_A}, D={D}, K={effective_K}, BatchSize={batch_size_q}")
    start_time = time.time()
    all_topk_indices = torch.full((Q, effective_K), -1, dtype=torch.int64, device=device)
    all_topk_distances_sq = torch.full((Q, effective_K), float('inf'), dtype=torch.float32, device=device)

    # --- Batch Processing Loop ---
    for q_start in range(0, Q, batch_size_q):
        q_end = min(q_start + batch_size_q, Q); batch_q_indices = slice(q_start, q_end)
        X_batch = X_prep[batch_q_indices]; current_batch_size = X_batch.shape[0]
        if current_batch_size == 0: continue

        try:
            # *** Use the native PyTorch distance function ***
            batch_distances_sq = distance_l2_squared_pytorch_native(X_batch, A_prep) # <--- CHANGED HERE
        except RuntimeError as e:
            # (OOM Error handling same as before) ...
             batch_mem_gb = current_batch_size * N_A * 4 / (1024**3); print(f"OOM Error during Brute Force batch distance calculation:\n Batch Q={current_batch_size}, N={N_A}. Estimated matrix memory: {batch_mem_gb:.2f} GB"); torch.cuda.empty_cache(); raise e
        except Exception as e: print(f"Error during Brute Force batch distance calculation: {e}"); raise e

        try:
            # (TopK logic same as before) ...
            batch_topk_distances_sq, batch_topk_indices = torch.topk(batch_distances_sq, k=effective_K, dim=1, largest=False, sorted=True)
        except Exception as e: print(f"Error during Brute Force batch topk ({q_start}-{q_end-1}): {e}"); raise e

        # (Storing results same as before) ...
        all_topk_indices[batch_q_indices] = batch_topk_indices
        all_topk_distances_sq[batch_q_indices] = batch_topk_distances_sq
        del batch_distances_sq, batch_topk_indices, batch_topk_distances_sq, X_batch

    # --- Post-Loop (Sync, Timing, Padding - Same as before) ---
    torch.cuda.synchronize(device=device)
    end_time = time.time(); print(f"k-NN Brute Force (Pure PyTorch, Batched) total computation time: {end_time - start_time:.4f} seconds")
    # (Padding logic same as before) ...
    if K > effective_K:
        pad_width = K - effective_K; indices_pad = torch.full((Q, pad_width), -1, dtype=torch.int64, device=device); dists_pad = torch.full((Q, pad_width), float('inf'), dtype=torch.float32, device=device)
        if all_topk_indices is None: all_topk_indices = torch.full((Q, effective_K), -1, dtype=torch.int64, device=device)
        if all_topk_distances_sq is None: all_topk_distances_sq = torch.full((Q, effective_K), float('inf'), dtype=torch.float32, device=device)
        all_topk_indices = torch.cat((all_topk_indices, indices_pad), dim=1); all_topk_distances_sq = torch.cat((all_topk_distances_sq, dists_pad), dim=1)
    if all_topk_indices is None: all_topk_indices = torch.full((Q, K), -1, dtype=torch.int64, device=device)
    if all_topk_distances_sq is None: all_topk_distances_sq = torch.full((Q, K), float('inf'), dtype=torch.float32, device=device)

    return all_topk_indices.to(torch.int64), all_topk_distances_sq.to(torch.float32)


# ============================================================================
# Main Execution Block (Pure PyTorch - Warmup & Dimension Loop)
# ============================================================================
if __name__ == "__main__":
    # --- Fixed Parameters (Same as before) ---
    N_data = 1000000
    N_queries = 1
    K_final_neighbors = 10
    num_clusters_kmeans = 1000
    num_clusters_probe = 300 # Keeping increased probe
    kmeans_max_iters = 50
    RECALL_THRESHOLD = 0.70
    dimensions_to_test = [2, 4, 64, 256, 1024]

    print("\n" + "="*60)
    print("--- Pure PyTorch ANN Full Test ---") # Updated Title
    print("="*60)
    # (Fixed Params printout same as before) ...
    print(f"Fixed Params: N={N_data}, Q={N_queries}, K_final={K_final_neighbors}")
    print(f"ANN Params: num_clusters={num_clusters_kmeans}, nprobe={num_clusters_probe}")
    print(f"Testing Dimensions: {dimensions_to_test}")

    # --- Warmup Phase ---
    print("\n" + "="*60)
    print("--- Warmup Run (PyTorch) ---")
    print("="*60)
    try:
        WARMUP_N = 10000; WARMUP_Q = 100; WARMUP_DIM = 32
        WARMUP_K_CLUSTERS = 50; WARMUP_NPROBE = 5; WARMUP_KFINAL = 5
        print(f"Warmup Params: N={WARMUP_N}, D={WARMUP_DIM}, Q={WARMUP_Q}, K={WARMUP_KFINAL}, Clusters={WARMUP_K_CLUSTERS}, NProbe={WARMUP_NPROBE}")
        A_warmup = torch.randn((WARMUP_N, WARMUP_DIM), dtype=torch.float32, device=device)
        X_warmup = torch.randn((WARMUP_Q, WARMUP_DIM), dtype=torch.float32, device=device)
        torch.cuda.synchronize(device=device)
        print("Warmup: Running ANN (includes KMeans)...")
        # *** Call the PyTorch ANN function ***
        _, _, _, _ = ann_ivf_like_pytorch( # <--- CHANGED HERE
            N_A=WARMUP_N, D=WARMUP_DIM, A=A_warmup, X=X_warmup,
            K_final=WARMUP_KFINAL, num_clusters=WARMUP_K_CLUSTERS,
            num_clusters_to_probe=WARMUP_NPROBE, max_kmeans_iters=5,
            verbose_kmeans=False
        )
        print("Warmup: Running Brute Force KNN...")
        # *** Call the PyTorch Brute Force function ***
        _, _ = pytorch_knn_bruteforce_native( # <--- CHANGED HERE
            N_A=WARMUP_N, D=WARMUP_DIM, A=A_warmup, X=X_warmup, K=WARMUP_KFINAL, batch_size_q=64
        )
        torch.cuda.synchronize(device=device)
        print("Warmup complete.")
    except Exception as e: print(f"An error occurred during warmup: {e}\nContinuing..."); traceback.print_exc()
    finally: # Cleanup warmup data
        if 'A_warmup' in locals(): del A_warmup
        if 'X_warmup' in locals(): del X_warmup
        torch.cuda.empty_cache()

    # --- Main Loop Over Dimensions ---
    print("\n" + "="*60)
    print("--- Starting Dimension Tests ---")
    print("="*60)

    for Dim in dimensions_to_test:
        print("\n" + "#"*70)
        print(f"# Testing Dimension D = {Dim}")
        print("#"*70)
        # (Per-Dimension Variables definition same as before) ...
        A_data = None; X_queries = None; ann_indices = None; ann_dists_sq = None
        true_knn_indices = None; build_t = 0; search_t = 0

        try:
            # --- Generate Data (Same PyTorch code as before) ---
            print(f"\n[D={Dim}] Generating Test Data (PyTorch)...")
            try:
                A_data = torch.randn((N_data, Dim), dtype=torch.float32, device=device)
                X_queries = torch.randn((N_queries, Dim), dtype=torch.float32, device=device)
                torch.cuda.synchronize(device=device)
                print(f"[D={Dim}] Data generated. Mem Allocated: {torch.cuda.memory_allocated(device)/(1024**3):.2f} GB")
            # (Error handling for data gen same as before) ...
            except RuntimeError as e: print(f"\n[D={Dim}] ERROR: OOM during data generation: {e}"); torch.cuda.empty_cache(); continue
            except Exception as e: print(f"\n[D={Dim}] ERROR generating data: {e}"); continue

            # --- Run ANN (Pure PyTorch version) ---
            print(f"\n[D={Dim}] Testing ANN (Pure PyTorch IVF-like)...")
            try:
                 # *** Call the PyTorch ANN function ***
                ann_indices, ann_dists_sq, build_t, search_t = ann_ivf_like_pytorch( # <--- CHANGED HERE
                    N_A=N_data, D=Dim, A=A_data, X=X_queries,
                    K_final=K_final_neighbors,
                    num_clusters=num_clusters_kmeans,
                    num_clusters_to_probe=num_clusters_probe,
                    max_kmeans_iters=kmeans_max_iters,
                    verbose_kmeans=False
                )
                print(f"\n[D={Dim}] ANN Results:")
                if ann_indices is not None: print(f"  Indices shape: {ann_indices.shape}")
                if ann_dists_sq is not None: print(f"  Sq Distances shape: {ann_dists_sq.shape}")
                print(f"  Build Time: {build_t:.4f}s")
                print(f"  Search Time: {search_t:.4f}s")
            # (Error handling for ANN same as before) ...
            except RuntimeError as e: print(f"\n[D={Dim}] ERROR: OOM during ANN execution: {e}"); ann_indices = None; torch.cuda.empty_cache()
            except Exception as e: print(f"\n[D={Dim}] ERROR during ANN execution: {e}"); traceback.print_exc(); ann_indices = None

            # --- Run Brute-Force KNN (Pure PyTorch version) ---
            if ann_indices is not None:
                print(f"\n[D={Dim}] Calculating Ground Truth (Pure PyTorch k-NN)...")
                try:
                    # *** Call the PyTorch Brute Force function ***
                    true_knn_indices, true_knn_dists_sq = pytorch_knn_bruteforce_native( # <--- CHANGED HERE
                        N_A=N_data, D=Dim, A=A_data, X=X_queries, K=K_final_neighbors
                    )
                    print(f"\n[D={Dim}] Ground Truth Results:")
                    if true_knn_indices is not None: print(f"  Indices shape: {true_knn_indices.shape}")
                # (Error handling for KNN same as before) ...
                except RuntimeError as e: print(f"\n[D={Dim}] ERROR: OOM during Brute Force k-NN: {e}"); true_knn_indices = None; torch.cuda.empty_cache()
                except Exception as e: print(f"\n[D={Dim}] ERROR during Brute Force k-NN execution: {e}"); traceback.print_exc(); true_knn_indices = None
                finally:
                     if 'true_knn_dists_sq' in locals(): del true_knn_dists_sq

            # --- Calculate Recall (Same NumPy comparison logic as before) ---
            if ann_indices is not None and true_knn_indices is not None:
                print(f"\n[D={Dim}] Calculating Recall@{K_final_neighbors}...")
                try:
                    # (Transfer to NumPy and comparison logic same as before) ...
                    start_recall_calc = time.time()
                    ann_indices_np = ann_indices.cpu().numpy()
                    true_indices_np = true_knn_indices.cpu().numpy()
                    total_intersect = 0
                    expected_neighbors_per_query = min(K_final_neighbors, N_data)
                    if N_queries > 0 and expected_neighbors_per_query > 0:
                        for i in range(N_queries):
                            ann_set = set(idx for idx in ann_indices_np[i] if idx >= 0)
                            true_set = set(idx for idx in true_indices_np[i] if idx >= 0)
                            total_intersect += len(ann_set.intersection(true_set))
                        denominator = N_queries * expected_neighbors_per_query
                        avg_recall = total_intersect / denominator if denominator > 0 else 1.0
                        print(f"\n[D={Dim}] Average Recall @ {K_final_neighbors}: {avg_recall:.4f} ({avg_recall:.2%})")
                        if avg_recall >= RECALL_THRESHOLD: print(f"[D={Dim}] Recall meets threshold ({RECALL_THRESHOLD:.2%}). CORRECT.")
                        else: print(f"[D={Dim}] Recall BELOW threshold ({RECALL_THRESHOLD:.2%}). INCORRECT.")
                    else: print(f"\n[D={Dim}] Cannot calculate recall.")
                except Exception as e: print(f"\n[D={Dim}] ERROR during Recall calculation: {e}"); traceback.print_exc()
            elif ann_indices is None: print(f"\n[D={Dim}] Skipping Recall: ANN failed.")
            elif true_knn_indices is None: print(f"\n[D={Dim}] Skipping Recall: Brute Force failed.")

        finally:
            # --- Cleanup for Current Dimension (Same logic) ---
            print(f"\n[D={Dim}] Cleaning up tensors...")
            del A_data; del X_queries; del ann_indices; del ann_dists_sq; del true_knn_indices
            if 'ann_indices_np' in locals(): del ann_indices_np
            if 'true_indices_np' in locals(): del true_indices_np
            # Also delete centroids/assignments if they exist from KMeans run
            if 'centroids' in locals(): del centroids
            if 'assignments' in locals(): del assignments
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[D={Dim}] Cleanup complete. Mem Allocated: {torch.cuda.memory_allocated(device)/(1024**3):.2f} GB")

        print(f"\n--- Finished Test for Dimension D = {Dim} ---")

    print("\n" + "="*60)
    print("--- ALL Pure PyTorch DIMENSION TESTS FINISHED ---")
    print("="*60)