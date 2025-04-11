# task-1/task_Ann.py (Corrected)

# Keep PyTorch and Triton imports if needed by HNSW or other parts of the file
import torch
import triton
import triton.language as tl
import math
import heapq # For HNSW priority queues
import random
import time
import cupy as cp
import cupyx # Required for cupyx.scatter_add

# --- Device Setup ---
# Keep PyTorch device setup if HNSW/other parts use it
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"PyTorch using device: {device}")
else:
    # Fallback for PyTorch if needed, but CuPy requires CUDA
    device = torch.device("cpu")
    print("PyTorch falling back to CPU.")

# Check and set CuPy device
try:
    cp.cuda.Device(0).use()
    print(f"CuPy using GPU: {cp.cuda.Device(0)}")
except cp.cuda.runtime.CUDARuntimeError as e:
    print(f"CuPy CUDA Error: {e}")
    print("Cannot run CuPy K-Means without CUDA.")
    # Decide how to handle this - exit or let subsequent CuPy calls fail?
    # exit()


# --- Helper Functions (Keep if needed by HNSW/Other PyTorch parts) ---
def _prepare_tensors(*tensors, target_device=device):
    """Ensure tensors are float32, contiguous, and on the correct device."""
    prepared = []
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            # If input is numpy or list, convert to tensor first
            try:
                t = torch.tensor(t, dtype=torch.float32, device=target_device)
            except Exception as e_conv: # Catch potential conversion errors
                raise TypeError(f"Could not convert input of type {type(t)} to torch.Tensor: {e_conv}")

        if t.device != target_device:
            t = t.to(target_device)
        if t.dtype != torch.float32:
            t = t.to(dtype=torch.float32)
        if not t.is_contiguous(): # Check contiguity after potential conversions
            t = t.contiguous()
        prepared.append(t)
    return prepared


def normalize_vectors(vectors, epsilon=1e-12):
    """L2 normalize vectors row-wise using PyTorch."""
    # Keep this if HNSW needs it
    norms = torch.linalg.norm(vectors, dim=1, keepdim=True)
    return vectors / (norms + epsilon)

# --- Distance Functions & Kernels (Keep if needed by HNSW/Other Parts) ---
DEFAULT_BLOCK_D = 128 # Default for simple kernels if used

@triton.jit
def l2_dist_kernel_1_vs_M( # Keep if HNSW uses it
    query_ptr, candidates_ptr, output_ptr,
    M, D, stride_cand_m, stride_cand_d,
    BLOCK_SIZE_D: tl.constexpr,
):
    # --- Kernel code remains the same ---
    pid_m = tl.program_id(axis=0)
    dist_sq = tl.zeros((), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_SIZE_D):
        offs_d = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offs_d < D
        query_d_ptr = query_ptr + offs_d
        query_vals = tl.load(query_d_ptr, mask=mask_d, other=0.0)
        cand_d_ptr = candidates_ptr + pid_m * stride_cand_m + offs_d * stride_cand_d
        cand_vals = tl.load(cand_d_ptr, mask=mask_d, other=0.0)
        diff = query_vals - cand_vals
        dist_sq += tl.sum(diff * diff, axis=0)
    tl.store(output_ptr + pid_m, dist_sq)
    pass

# ============================================================================
# Task 2.1: K-Means Clustering (Pure CuPy Implementation - Corrected)
# ============================================================================

def pairwise_l2_squared_cupy(X_cp, C_cp):
    """
    Computes pairwise squared L2 distances between points in X and centroids C using CuPy.
    X_cp: (N, D) data points OR (Q, D) query points
    C_cp: (K, D) centroids OR (N, D) database points
    Returns: (N, K) or (Q, K) or (Q, N) tensor of squared distances.
    """
    # Ensure inputs are CuPy arrays and float32
    if not isinstance(X_cp, cp.ndarray): X_cp = cp.asarray(X_cp)
    if not isinstance(C_cp, cp.ndarray): C_cp = cp.asarray(C_cp)
    if X_cp.dtype != cp.float32: X_cp = X_cp.astype(cp.float32)
    if C_cp.dtype != cp.float32: C_cp = C_cp.astype(cp.float32)

    # ||x - c||^2 = ||x||^2 - 2<x, c> + ||c||^2
    X_norm_sq = cp.sum(X_cp**2, axis=1, keepdims=True) # Shape (N|Q, 1)
    C_norm_sq = cp.sum(C_cp**2, axis=1, keepdims=True) # Shape (K|N, 1)

    # Use gemm for dot product (generally fastest)
    # Note: CuPy matmul uses cuBLAS
    dot_products = cp.matmul(X_cp, C_cp.T) # Shape (N|Q, K|N)

    # Broadcasting: (N|Q, 1) - 2*(N|Q, K|N) + (1, K|N) -> (N|Q, K|N)
    dist_sq = X_norm_sq - 2 * dot_products + C_norm_sq.T
    return cp.maximum(0, dist_sq) # Clamp to avoid small numerical negatives due to precision

def our_kmeans(N_A, D, A_cp, K, max_iters=100, tol=1e-4):
    """
    Performs K-means clustering entirely using CuPy. Corrected scatter_add usage.
    (Code from the original prompt - assumed correct and used by our_ann_cupy_ivf)
    """
    # --- Input Validation ---
    if not isinstance(A_cp, cp.ndarray):
        raise TypeError("Input data 'A_cp' must be a CuPy ndarray.")
    if A_cp.shape[0] != N_A or A_cp.shape[1] != D:
         print(f"Warning: N_A/D ({N_A}/{D}) mismatch with A_cp shape {A_cp.shape}. Using shape from A_cp.")
         N_A, D = A_cp.shape
    if not (K > 0 and K <= N_A):
         raise ValueError("K must be positive and less than or equal to N_A.")
    if A_cp.dtype != cp.float32:
        print(f"Warning: Input data dtype is {A_cp.dtype}. Converting to float32.")
        A_cp = A_cp.astype(cp.float32)

    print(f"Running K-Means (Pure CuPy): N={N_A}, D={D}, K={K}")
    start_time_total = time.time()

    # --- Initialization (using CuPy) ---
    initial_indices = cp.random.permutation(N_A)[:K]
    centroids_cp = A_cp[initial_indices].copy()

    assignments_cp = cp.empty(N_A, dtype=cp.int32) # Use int32 for assignments

    for i in range(max_iters):
        iter_start_time = time.time()

        # --- 1. Assignment Step (CuPy) ---
        all_dist_sq_cp = pairwise_l2_squared_cupy(A_cp, centroids_cp) # Shape (N_A, K)
        assignments_cp = cp.argmin(all_dist_sq_cp, axis=1).astype(cp.int32) # Shape (N_A,)
        cp.cuda.Stream.null.synchronize()
        assign_time = time.time() - iter_start_time

        # --- 2. Update Step (CuPy using cupyx.scatter_add) ---
        update_start_time = time.time()

        new_sums_cp = cp.zeros((K, D), dtype=cp.float32)
        # Ensure cluster_counts is float for scatter_add and division
        cluster_counts_cp = cp.zeros(K, dtype=cp.float32)

        # Use cupyx.scatter_add
        cupyx.scatter_add(new_sums_cp, assignments_cp, A_cp) # Add vectors to assigned centroid sum
        cupyx.scatter_add(cluster_counts_cp, assignments_cp, 1.0) # Increment count for assigned centroid

        # Avoid division by zero for empty clusters
        # Add small epsilon or use maximum(count, 1) before division
        final_counts_safe_cp = cp.maximum(cluster_counts_cp, 1.0) # Avoid div by zero
        new_centroids_cp = new_sums_cp / final_counts_safe_cp[:, None] # Broadcast division

        # Handle empty clusters: Re-assign old centroid or a random point
        # Here, we keep the old centroid position if a cluster becomes empty
        empty_cluster_mask = (cluster_counts_cp == 0)
        if cp.any(empty_cluster_mask):
             new_centroids_cp[empty_cluster_mask] = centroids_cp[empty_cluster_mask]

        cp.cuda.Stream.null.synchronize()
        update_time = time.time() - update_start_time

        # --- Check Convergence ---
        centroid_diff_cp = cp.linalg.norm(new_centroids_cp - centroids_cp)
        centroids_cp = new_centroids_cp

        # Optional: Print iteration stats
        # print(f"  Iter {i+1}/{max_iters} | Centroid Diff: {centroid_diff_cp:.4f} | Assign Time: {assign_time:.4f}s | Update Time: {update_time:.4f}s")

        if centroid_diff_cp < tol:
            print(f"Converged after {i+1} iterations.")
            break

    if i == max_iters - 1:
        print(f"Reached max iterations ({max_iters}).")

    total_time = time.time() - start_time_total
    print(f"Total K-Means time: {total_time:.4f}s")

    # Return assignments as int64 if required by subsequent code, otherwise int32 is fine
    return centroids_cp, assignments_cp # .astype(cp.int64)

# ============================================================================
# Task 2.2: Approximate Nearest Neighbors (ANN - HNSW - Keep if used)
# ============================================================================
# class SimpleHNSW_for_ANN: ... (Keep original HNSW code if needed)
# def our_ann(...): ... (Keep original HNSW wrapper if needed)

# ============================================================================
# Task 2.2: Approximate Nearest Neighbors (ANN - KMeans IVF - CuPy Optimized)
# ============================================================================

def our_ann_cupy_ivf_optimized(N_A, D, A_cp, X_cp, K, k_clusters=100, nprobe=5, max_kmeans_iters=100):
    """
    Performs Approximate Nearest Neighbor search using KMeans-IVF with CuPy,
    optimized for GPU execution by minimizing CPU transfers.

    Args:
        N_A (int): Number of database points.
        D (int): Dimensionality.
        A_cp (cp.ndarray): Database vectors (N_A, D) on GPU. Should be float32.
        X_cp (cp.ndarray): Query vectors (Q, D) on GPU. Should be float32.
        K (int): Number of neighbors to find for each query.
        k_clusters (int): Number of clusters for the K-Means index.
        nprobe (int): Number of clusters to probe during search.
        max_kmeans_iters (int): Max iterations for K-Means during index build.

    Returns:
        tuple[cp.ndarray, cp.ndarray, float, float]:
            - all_indices_cp (cp.ndarray): Original indices of the K nearest neighbors (Q, K).
            - all_distances_cp (cp.ndarray): Squared L2 distances of the K nearest neighbors (Q, K).
            - build_time (float): Time taken to build the index (KMeans + Inverted Index).
            - search_time (float): Time taken for searching all queries.
    """
    # --- Input Validation ---
    if not isinstance(A_cp, cp.ndarray) or not isinstance(X_cp, cp.ndarray):
        raise TypeError("Input data 'A_cp' and queries 'X_cp' must be CuPy ndarrays.")
    if A_cp.dtype != cp.float32:
        print("Warning: A_cp is not float32. Converting...")
        A_cp = A_cp.astype(cp.float32)
    if X_cp.dtype != cp.float32:
        print("Warning: X_cp is not float32. Converting...")
        X_cp = X_cp.astype(cp.float32)

    Q = X_cp.shape[0]
    if X_cp.shape[1] != D: raise ValueError(f"Query dimension mismatch: X_cp D={X_cp.shape[1]}, expected D={D}")
    if A_cp.shape[0] != N_A: print(f"Warning: N_A mismatch ({N_A}) with A_cp shape ({A_cp.shape[0]}). Using shape from A_cp."); N_A = A_cp.shape[0]
    if A_cp.shape[1] != D: raise ValueError(f"Database dimension mismatch: A_cp D={A_cp.shape[1]}, expected D={D}")
    if not (K > 0): raise ValueError("K must be positive")
    if not (k_clusters > 0): raise ValueError("k_clusters must be positive")
    if not (nprobe > 0 and nprobe <= k_clusters): raise ValueError("nprobe must be between 1 and k_clusters")
    K = min(K, N_A) # Cannot return more neighbors than exist

    print(f"Running ANN (KMeans-IVF / CuPy Optimized): Q={Q}, N={N_A}, D={D}, K={K}")
    print(f"IVF Params: k_clusters={k_clusters}, nprobe={nprobe}")

    # --- 1. Build Index (K-Means) ---
    build_start_time = time.time()
    centroids_cp, assignments_cp = our_kmeans(N_A, D, A_cp, k_clusters, max_iters=max_kmeans_iters)
    # Ensure assignments are int32 for sorting/indexing
    assignments_cp = assignments_cp.astype(cp.int32)
    cp.cuda.Stream.null.synchronize() # Sync after k-means
    build_time_kmeans = time.time() - build_start_time
    print(f"Index build time (KMeans): {build_time_kmeans:.4f}s")

    # --- 2. Build Inverted Index (GPU Optimized) ---
    build_invidx_start_time = time.time()

    original_indices = cp.arange(N_A, dtype=cp.int64) # Keep original indices as int64

    # Sort original indices based on cluster assignments
    sort_permutation = cp.argsort(assignments_cp)
    # Contains original indices sorted by the cluster they belong to
    inv_idx_values_cp = original_indices[sort_permutation]
    # Cluster assignments sorted according to the permutation
    sorted_assignments = assignments_cp[sort_permutation]

    # Find start positions and counts for each cluster ID that is actually present
    # unique_clusters will be sorted cluster IDs that are actually present
    unique_clusters, inv_idx_starts_cp, inv_idx_counts_cp = cp.unique(
        sorted_assignments, return_index=True, return_counts=True
    )

    # Create mappings from potential cluster ID (0 to k_clusters-1) to the location
    # in the unique_clusters/inv_idx_starts_cp/inv_idx_counts_cp arrays.
    # This handles cases where some cluster IDs might be empty (not present in assignments_cp).
    # We use -1 to indicate a cluster ID is not present.
    cluster_id_to_unique_idx = cp.full(k_clusters, -1, dtype=cp.int32)
    cluster_id_to_unique_idx[unique_clusters] = cp.arange(len(unique_clusters), dtype=cp.int32)

    # Precompute starts and counts mapped to the full 0..k_clusters-1 range
    # Use -1 index and 0 count for clusters that are empty/missing
    full_inv_idx_starts = cp.full(k_clusters, -1, dtype=cp.int32) # Default: invalid start index
    full_inv_idx_counts = cp.zeros(k_clusters, dtype=cp.int32)   # Default: zero count

    present_mask = (cluster_id_to_unique_idx != -1) # Mask for clusters that are present
    # Indices into unique_clusters/inv_idx_starts/counts for the present clusters
    indices_in_unique = cluster_id_to_unique_idx[present_mask]

    if indices_in_unique.size > 0:
        # Get the actual cluster IDs (0..k_clusters-1) that are present
        present_cluster_ids = cp.where(present_mask)[0]
        # Use advanced indexing to populate the full arrays only for present clusters
        full_inv_idx_starts[present_cluster_ids] = inv_idx_starts_cp[indices_in_unique]
        full_inv_idx_counts[present_cluster_ids] = inv_idx_counts_cp[indices_in_unique]

    # Cleanup intermediate arrays if memory is tight (optional)
    # del sort_permutation, sorted_assignments, unique_clusters
    # del inv_idx_starts_cp, inv_idx_counts_cp, cluster_id_to_unique_idx, present_mask, indices_in_unique

    cp.cuda.Stream.null.synchronize() # Sync after index build
    build_time_invidx = time.time() - build_invidx_start_time
    build_time = build_time_kmeans + build_time_invidx # Total build time
    print(f"Index build time (Inverted Index GPU): {build_time_invidx:.4f}s")
    print(f"Index build time (Total): {build_time:.4f}s")

    # --- 3. Perform Search ---
    search_start_time = time.time()
    all_indices_cp = cp.full((Q, K), -1, dtype=cp.int64)
    all_distances_cp = cp.full((Q, K), cp.inf, dtype=cp.float32)

    # --- Calculate all query-centroid distances at once (vectorized) ---
    # Shape (Q, k_clusters)
    all_query_centroid_dists_sq = pairwise_l2_squared_cupy(X_cp, centroids_cp)

    # --- Find nprobe nearest clusters for all queries at once (vectorized) ---
    # Shape (Q, nprobe) - Indices of the nearest clusters for each query
    all_nearest_cluster_indices = cp.argpartition(all_query_centroid_dists_sq, nprobe, axis=1)[:, :nprobe]
    # Ensure indices are int32 for indexing our helper arrays
    all_nearest_cluster_indices = all_nearest_cluster_indices.astype(cp.int32)

    # --- Iterate through queries (CPU loop, but GPU work inside) ---
    # Fully vectorizing the rest across queries is complex due to variable candidate counts
    for q_idx in range(Q):
        query_cp = X_cp[q_idx:q_idx+1] # Keep it 2D: (1, D) for pairwise_l2_squared_cupy
        # Indices of the nprobe nearest clusters for *this* query
        nearest_cluster_indices = all_nearest_cluster_indices[q_idx] # Shape (nprobe,)

        # b. Gather candidate points from selected clusters (GPU Optimized)
        # Get the start positions and counts for the selected clusters
        selected_starts = full_inv_idx_starts[nearest_cluster_indices] # Shape (nprobe,)
        selected_counts = full_inv_idx_counts[nearest_cluster_indices] # Shape (nprobe,)

        # Use a list to collect candidate arrays from GPU, then concatenate on GPU
        candidate_indices_list_gpu = []
        # This loop iterates nprobe times (CPU controlled, but small)
        for i in range(nprobe):
            start = selected_starts[i].item() # Get scalar value
            count = selected_counts[i].item() # Get scalar value
            # Check if cluster is valid (start index >= 0) and has points (count > 0)
            if start >= 0 and count > 0:
                # Append a view/slice of the GPU array (no data copy yet)
                candidate_indices_list_gpu.append(inv_idx_values_cp[start : start + count])

        if not candidate_indices_list_gpu:
             # print(f"Query {q_idx}: No candidates found in probed clusters.") # Optional debug
             continue # No candidates found for this query

        # Concatenate candidates from different clusters ON THE GPU
        candidate_indices_cp = cp.concatenate(candidate_indices_list_gpu)

        # Remove duplicate candidates that might appear if a point is near boundaries probed
        # This can be costly if candidate_indices_cp is huge, but necessary for correctness
        # and potentially reduces work for distance calculation.
        unique_candidate_indices_cp = cp.unique(candidate_indices_cp)

        num_unique_candidates = unique_candidate_indices_cp.size
        if num_unique_candidates == 0:
            continue

        # c. Fetch candidate vectors (GPU indexing)
        candidate_vectors_cp = A_cp[unique_candidate_indices_cp]

        # d. Calculate exact distances to candidates (vectorized on GPU)
        query_candidate_dists_sq = pairwise_l2_squared_cupy(query_cp, candidate_vectors_cp) # Shape (1, num_unique_candidates)

        # e. Find top K among candidates (vectorized on GPU)
        actual_k = min(K, num_unique_candidates) # We can't return more candidates than we found

        if actual_k > 0:
            # Get indices of K smallest distances *within the unique candidate subset*
            # argpartition is usually faster than argsort for finding k smallest
            topk_relative_indices = cp.argpartition(query_candidate_dists_sq[0], actual_k)[:actual_k] # Indices relative to unique_candidate_indices_cp

            # Get the distances corresponding to these top k relative indices
            topk_distances_sq = query_candidate_dists_sq[0, topk_relative_indices]

            # Sort these K results by distance
            sort_order = cp.argsort(topk_distances_sq)

            # Apply the sort order to get the final relative indices and distances
            final_topk_relative_indices = topk_relative_indices[sort_order]
            final_topk_distances_sq = topk_distances_sq[sort_order] # Or query_candidate_dists_sq[0, final_topk_relative_indices]

            # Map relative indices back to the original database indices
            final_topk_original_indices = unique_candidate_indices_cp[final_topk_relative_indices]

            # Store results for this query
            all_indices_cp[q_idx, :actual_k] = final_topk_original_indices
            all_distances_cp[q_idx, :actual_k] = final_topk_distances_sq

    cp.cuda.Stream.null.synchronize() # Sync after all queries are processed
    search_time = time.time() - search_start_time
    print(f"ANN search time (Optimized): {search_time:.4f} seconds")

    return all_indices_cp, all_distances_cp, build_time, search_time


# ============================================================================
# Brute-Force k-NN (CuPy version for Recall Calculation - Keep as is)
# ============================================================================
def cupy_knn_bruteforce(N_A, D, A_cp, X_cp, K):
    """
    Finds the K nearest neighbors using brute-force pairwise L2 distance (CuPy).
    (Code from the original prompt - assumed correct)
    """
    # --- Input Validation ---
    if not isinstance(A_cp, cp.ndarray): A_cp = cp.asarray(A_cp)
    if not isinstance(X_cp, cp.ndarray): X_cp = cp.asarray(X_cp)
    if A_cp.dtype != cp.float32: A_cp = A_cp.astype(cp.float32)
    if X_cp.dtype != cp.float32: X_cp = X_cp.astype(cp.float32)

    Q = X_cp.shape[0]
    if X_cp.shape[1] != D: raise ValueError("Query dimension mismatch")
    if A_cp.shape[0] != N_A: print(f"Warning: N_A mismatch. Using A_cp.shape[0]={A_cp.shape[0]}"); N_A = A_cp.shape[0]
    if A_cp.shape[1] != D: raise ValueError("Database dimension mismatch")
    if not K > 0: raise ValueError("K must be positive")
    K = min(K, N_A) # Cannot return more neighbors than exist

    print(f"Running k-NN Brute Force (CuPy): Q={Q}, N={N_A}, D={D}, K={K}")
    start_time = time.time()

    all_distances_sq = pairwise_l2_squared_cupy(X_cp, A_cp) # Shape (Q, N_A)

    # Partitioning is generally faster than full sort for finding top K
    # Find indices of K smallest distances for each query (row)
    # Ensure K is not larger than N_A before partitioning
    k_partition = min(K, N_A -1) if N_A > 0 else 0 # argpartition needs k < N
    if k_partition < 0: k_partition = 0 # Handle N_A=0 case

    if N_A == 0: # Handle empty database case
        topk_indices_cp = cp.full((Q, K), -1, dtype=cp.int64)
        topk_distances_sq_cp = cp.full((Q, K), cp.inf, dtype=cp.float32)
    elif K >= N_A: # If K is larger or equal N_A, just sort all distances
        topk_indices_cp = cp.argsort(all_distances_sq, axis=1)[:, :K]
        topk_distances_sq_cp = cp.take_along_axis(all_distances_sq, topk_indices_cp, axis=1)
    else:
        # Use argpartition to find the K smallest elements (indices)
        topk_indices_unstructured = cp.argpartition(all_distances_sq, k_partition, axis=1)[:, :K] # Shape (Q, K)

        # Get the actual distances for these K indices
        topk_distances_unstructured = cp.take_along_axis(all_distances_sq, topk_indices_unstructured, axis=1) # Shape (Q, K)

        # Sort within the retrieved K elements for each query based on distance
        sorted_order_in_k = cp.argsort(topk_distances_unstructured, axis=1) # Shape (Q, K)

        # Apply the sort order to get the final indices and distances
        topk_indices_cp = cp.take_along_axis(topk_indices_unstructured, sorted_order_in_k, axis=1)
        topk_distances_sq_cp = cp.take_along_axis(topk_distances_unstructured, sorted_order_in_k, axis=1)


    cp.cuda.Stream.null.synchronize()
    end_time = time.time()
    print(f"k-NN Brute Force (CuPy) computation time: {end_time - start_time:.4f} seconds")

    return topk_indices_cp, topk_distances_sq_cp


# ============================================================================
# Example Usage (Modified to use the optimized function)
# ============================================================================


class SimpleHNSW_for_ANN:
    def __init__(self, dim, M=16, ef_construction=200, ef_search=50, mL=0.5):
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.mL = mL
        self.vectors = torch.empty((0, dim), dtype=torch.float32, device=device)
        self.graph = []
        self.level_assignments = []
        self.node_count = 0
        self.entry_point = -1
        self.max_level = -1
        self.BLOCK_SIZE_D_DIST = 128 # Default block size for 1-vs-M kernel

    def _get_level_for_new_node(self):
        level = int(-math.log(random.uniform(0, 1)) * self.mL)
        return level

    def _distance(self, query_vec, candidate_indices):
        """Internal distance calc using the 1-vs-M Triton kernel (Squared L2).
           CORRECTED: Always returns tensors, even if empty.
        """
        target_device = query_vec.device if isinstance(query_vec, torch.Tensor) else self.vectors.device
        empty_indices = torch.empty(0, dtype=torch.long, device=target_device)
        empty_distances = torch.empty(0, device=target_device)

        if not isinstance(candidate_indices, list): candidate_indices = list(candidate_indices)
        if not candidate_indices: return empty_distances, empty_indices # Return empty tensors

        query_vec_prep, = _prepare_tensors(query_vec.flatten(), target_device=target_device)
        valid_indices_list = [idx for idx in candidate_indices if idx < self.node_count and idx >= 0]

        if not valid_indices_list:
            return empty_distances, empty_indices # Return empty tensors

        # Convert valid indices to tensor
        valid_indices = torch.tensor(valid_indices_list, dtype=torch.long, device=target_device)
        num_valid_candidates = len(valid_indices_list)
        candidate_vectors, = _prepare_tensors(self.vectors[valid_indices], target_device=target_device) # Index with tensor

        distances_out = torch.empty(num_valid_candidates, dtype=torch.float32, device=device)
        grid = (num_valid_candidates,)
        # Ensure l2_dist_kernel_1_vs_M is defined globally
        l2_dist_kernel_1_vs_M[grid](
            query_vec_prep, candidate_vectors, distances_out,
            num_valid_candidates, self.dim,
            candidate_vectors.stride(0), candidate_vectors.stride(1),
            BLOCK_SIZE_D=self.BLOCK_SIZE_D_DIST
        )
        return distances_out, valid_indices # Returns squared L2 and VALID INDICES TENSOR

    def _distance_batch(self, query_indices, candidate_indices):
        """
        Calculates pairwise L2 distances between batches using PyTorch cdist.
        Returns a tensor of L2 distances.
        """
        target_device = self.vectors.device
        empty_result = torch.empty((0, 0), device=target_device) # Define empty tensor based on device

        if not isinstance(query_indices, list): query_indices = list(query_indices)
        if not isinstance(candidate_indices, list): candidate_indices = list(candidate_indices)
        if not query_indices or not candidate_indices:
             return empty_result # Return empty tensor matching device

        valid_query_indices = [idx for idx in query_indices if idx >= 0 and idx < self.node_count]
        valid_candidate_indices = [idx for idx in candidate_indices if idx >= 0 and idx < self.node_count]

        if not valid_query_indices or not valid_candidate_indices:
            # Return shape (actual_query, actual_candidate)
            return torch.empty((len(valid_query_indices), len(valid_candidate_indices)), device=target_device)

        # Indexing needs lists or tensors, not sets
        query_vectors = self.vectors[valid_query_indices]
        candidate_vectors = self.vectors[valid_candidate_indices]

        try:
            query_vec_prep, cand_vec_prep = _prepare_tensors(query_vectors, candidate_vectors, target_device=target_device)
            pairwise_l2_distances = torch.cdist(query_vec_prep, cand_vec_prep, p=2) # L2 distance
        except Exception as e:
            print(f"Error in _distance_batch using torch.cdist: {e}")
            return torch.empty((len(valid_query_indices), len(valid_candidate_indices)), device=target_device)

        return pairwise_l2_distances # Shape (len(valid_query), len(valid_candidate))

    def _select_neighbors_heuristic(self, query_vec, candidates, M_target):
        """Selects M_target neighbors (using squared L2 internally)."""
        selected_neighbors = []
        working_candidates_heap = [(dist, nid) for dist, nid in candidates]
        heapq.heapify(working_candidates_heap)
        discarded_candidates = set()

        while working_candidates_heap and len(selected_neighbors) < M_target:
            dist_best_sq, best_nid = heapq.heappop(working_candidates_heap)
            if best_nid in discarded_candidates: continue
            selected_neighbors.append(best_nid)

            remaining_candidates_info = {}
            temp_heap = []
            while working_candidates_heap:
                 dist_r_sq, nid_r = heapq.heappop(working_candidates_heap)
                 if nid_r not in discarded_candidates:
                      remaining_candidates_info[nid_r] = dist_r_sq
                      temp_heap.append((dist_r_sq, nid_r))
            working_candidates_heap = temp_heap
            heapq.heapify(working_candidates_heap)
            remaining_nids = list(remaining_candidates_info.keys())

            if remaining_nids:
                dists_best_to_remaining = self._distance_batch([best_nid], remaining_nids) # Actual L2
                if dists_best_to_remaining.numel() > 0:
                    dists_best_to_remaining_sq = (dists_best_to_remaining**2).squeeze(0) # Squared L2

                    for i, r_nid in enumerate(remaining_nids):
                        dist_r_query_sq = remaining_candidates_info[r_nid] # Already squared L2
                        if i < len(dists_best_to_remaining_sq): # Bounds check
                           dist_r_best_sq = dists_best_to_remaining_sq[i].item()
                           if dist_r_best_sq < dist_r_query_sq:
                               discarded_candidates.add(r_nid)
        return selected_neighbors

    def add_point(self, point_vec):
        """Adds a single point to the graph."""
        target_device = self.vectors.device if self.node_count > 0 else device
        point_vec_prep, = _prepare_tensors(point_vec.flatten(), target_device=target_device)
        new_node_id = self.node_count

        if self.node_count == 0: self.vectors = point_vec_prep.unsqueeze(0)
        else: self.vectors = torch.cat((self.vectors, point_vec_prep.unsqueeze(0)), dim=0)
        self.node_count += 1
        node_level = self._get_level_for_new_node()
        self.level_assignments.append(node_level)

        while node_level >= len(self.graph): self.graph.append([])
        for lvl in range(len(self.graph)):
             while len(self.graph[lvl]) <= new_node_id: self.graph[lvl].append([])

        current_entry_point = self.entry_point
        current_max_level = self.max_level
        if current_entry_point == -1:
            self.entry_point = new_node_id; self.max_level = node_level; return new_node_id

        ep = [current_entry_point]
        for level in range(current_max_level, node_level, -1):
            if level >= len(self.graph) or not ep or ep[0] < 0 or ep[0] >= len(self.graph[level]): continue # Basic check
            search_results = self._search_layer(point_vec_prep, ep, level, ef=1)
            if not search_results:
                 # If search fails, maybe ep was invalid? Try global EP as fallback?
                 if current_entry_point >= 0 and current_entry_point < len(self.graph[level]):
                      search_results = self._search_layer(point_vec_prep, [current_entry_point], level, ef=1)
                      if search_results: ep = [search_results[0][1]]
                      else: break # Stop if still no results
                 else: break # Cannot proceed
            else: ep = [search_results[0][1]]

        for level in range(min(node_level, current_max_level), -1, -1):
            if level >= len(self.graph): continue # Should not happen

            # Ensure ep is valid for this level
            current_ep = []
            if ep: # Check if ep list is not empty
                 valid_eps_in_list = [idx for idx in ep if idx >= 0 and idx < len(self.graph[level])]
                 if valid_eps_in_list:
                     current_ep = valid_eps_in_list
                 elif current_entry_point >= 0 and current_entry_point < len(self.graph[level]):
                     current_ep = [current_entry_point] # Fallback to global EP
                 else:
                     continue # Cannot proceed with search at this level

            neighbors_found_with_dist_sq = self._search_layer(point_vec_prep, current_ep, level, self.ef_construction) # Uses squared L2
            if not neighbors_found_with_dist_sq: 
                continue

            selected_neighbor_ids = self._select_neighbors_heuristic(point_vec_prep, neighbors_found_with_dist_sq, self.M)
            self.graph[level][new_node_id] = selected_neighbor_ids

            for neighbor_id in selected_neighbor_ids:
                 if neighbor_id < 0 or neighbor_id >= len(self.graph[level]): continue # Check index validity
                 neighbor_connections = self.graph[level][neighbor_id]
                 if new_node_id not in neighbor_connections:
                     if len(neighbor_connections) < self.M:
                         neighbor_connections.append(new_node_id)
                     else:
                         # Pruning logic (uses squared L2 from _distance)
                         dist_new_sq, valid_new = self._distance(self.vectors[neighbor_id], [new_node_id])
                         if not valid_new or dist_new_sq.numel() == 0: continue
                         dist_new_sq = dist_new_sq[0].item()

                         current_neighbor_ids = list(neighbor_connections)
                         dists_to_current_sq, valid_curr_ids = self._distance(self.vectors[neighbor_id], current_neighbor_ids)

                         if dists_to_current_sq.numel() > 0:
                              furthest_dist_sq = -1.0; furthest_idx_in_list = -1
                              dist_map = {nid.item(): d.item() for nid, d in zip(valid_curr_ids, dists_to_current_sq)}
                              for list_idx, current_nid in enumerate(current_neighbor_ids):
                                   d_sq = dist_map.get(current_nid, float('inf'))
                                   if d_sq > furthest_dist_sq: furthest_dist_sq = d_sq; furthest_idx_in_list = list_idx
                              if furthest_idx_in_list != -1 and dist_new_sq < furthest_dist_sq:
                                   neighbor_connections[furthest_idx_in_list] = new_node_id

            ep = selected_neighbor_ids
            if not ep: ep = [nid for _, nid in neighbors_found_with_dist_sq[:1]]

        if node_level > self.max_level: self.max_level = node_level; self.entry_point = new_node_id
        return new_node_id

    def _search_layer(self, query_vec, entry_points, target_level, ef):
        """Performs greedy search on a single layer. Returns list of (squared_dist, node_id).
           CORRECTED: Checks valid_indices.numel() instead of distances.numel().
        """
        if self.entry_point == -1: return []
        target_device = query_vec.device if isinstance(query_vec, torch.Tensor) else self.vectors.device
        valid_entry_points = [ep for ep in entry_points if ep >= 0 and ep < self.node_count]
        if not valid_entry_points:
             if self.entry_point >= 0 and self.entry_point < self.node_count: valid_entry_points = [self.entry_point]
             else: return []

        # _distance returns (squared_distances, valid_indices_tensor)
        initial_distances_sq, valid_indices_init = self._distance(query_vec, valid_entry_points) # Squared L2

        # --- CORRECTED CHECK ---
        if valid_indices_init.numel() == 0: return [] # Check if ANY valid EPs were found

        dist_map_init = {nid.item(): d.item() for nid, d in zip(valid_indices_init, initial_distances_sq)}
        # Use only the valid entry points for heap initialization
        candidate_heap = [(dist_map_init[ep], ep) for ep in valid_indices_init.tolist()]
        heapq.heapify(candidate_heap)
        results_heap = [(-dist_sq, node_id) for dist_sq, node_id in candidate_heap] # Use squared dist
        heapq.heapify(results_heap)
        visited = set(valid_indices_init.tolist()) # Start visited set with valid EPs

        while candidate_heap:
            dist_candidate_sq, current_node_id = heapq.heappop(candidate_heap)
            # No need to check for inf here if heap initialized correctly

            furthest_dist_sq = -results_heap[0][0] if results_heap else float('inf')
            if dist_candidate_sq > furthest_dist_sq and len(results_heap) >= ef: break

            try: neighbors = self.graph[target_level][current_node_id]
            except IndexError: neighbors = []

            unvisited_neighbors = [n for n in neighbors if n not in visited]
            if unvisited_neighbors:
                 visited.update(unvisited_neighbors)
                 # _distance returns (squared_distances, valid_indices_tensor)
                 neighbor_distances_sq, valid_neighbor_indices = self._distance(query_vec, unvisited_neighbors) # Squared L2

                 # --- CORRECTED CHECK ---
                 if valid_neighbor_indices.numel() == 0: continue # Check if ANY valid neighbors had dist calculated

                 dist_map_neighbors = {nid.item(): d.item() for nid, d in zip(valid_neighbor_indices, neighbor_distances_sq)}
                 for neighbor_id_tensor in valid_neighbor_indices:
                      neighbor_id = neighbor_id_tensor.item()
                      neighbor_dist_sq_val = dist_map_neighbors[neighbor_id]
                      furthest_dist_sq = -results_heap[0][0] if results_heap else float('inf')

                      if len(results_heap) < ef or neighbor_dist_sq_val < furthest_dist_sq:
                           heapq.heappush(results_heap, (-neighbor_dist_sq_val, neighbor_id))
                           if len(results_heap) > ef: heapq.heappop(results_heap)
                           heapq.heappush(candidate_heap, (neighbor_dist_sq_val, neighbor_id))

        # Return sorted list: (squared_dist, node_id)
        final_results = sorted([(abs(neg_dist_sq), node_id) for neg_dist_sq, node_id in results_heap])
        return final_results

    def search_knn(self, query_vec, k):
        """Searches for k nearest neighbors (returns squared L2 dists)."""
        if self.entry_point == -1: return []
        target_device = self.vectors.device if self.node_count > 0 else device
        query_vec_prep, = _prepare_tensors(query_vec.flatten(), target_device=target_device)
        ep = [self.entry_point]
        current_max_level = self.max_level
        for level in range(current_max_level, 0, -1):
             if level >= len(self.graph): continue # Skip non-existent levels
             valid_ep = []
             if ep and ep[0] >=0 and ep[0] < len(self.graph[level]):
                 valid_ep = ep
             elif self.entry_point >=0 and self.entry_point < len(self.graph[level]):
                 valid_ep = [self.entry_point]
             else: break # Cannot proceed
             if not valid_ep : break # Cannot proceed

             search_results = self._search_layer(query_vec_prep, valid_ep, level, ef=1) # Squared L2
             if not search_results: break
             ep = [search_results[0][1]]

        # Final search at level 0
        if 0 >= len(self.graph): return [] # Level 0 doesn't exist

        final_ep = []
        if ep and ep[0] >=0 and ep[0] < len(self.graph[0]):
            final_ep = ep
        elif self.entry_point != -1 and 0 < len(self.graph) and self.entry_point < len(self.graph[0]):
            final_ep = [self.entry_point]
        else:
             return [] # Cannot search level 0

        neighbors_found = self._search_layer(query_vec_prep, final_ep, 0, self.ef_search) # Squared L2
        return neighbors_found[:k] # Returns (squared_dist, node_id)


# --- our_ann function wrapper (remains unchanged) ---
def our_ann(N_A, D, A, X, K, M=16, ef_construction=100, ef_search=50):
     # This function assumes A and X are PyTorch tensors
     target_device = X.device
     A_prep, X_prep = _prepare_tensors(A, X, target_device=target_device)
     Q = X_prep.shape[0]
     assert A_prep.shape[0]==N_A and A_prep.shape[1]==D and X_prep.shape[1]==D and K>0
     print(f"Running ANN (HNSW): Q={Q}, N={N_A}, D={D}, K={K}, M={M}, efC={ef_construction}, efS={ef_search}")
     start_build = time.time()
     hnsw_index = SimpleHNSW_for_ANN(dim=D, M=M, ef_construction=ef_construction, ef_search=ef_search)
     print("Building index..."); i=0
     for i in range(N_A): hnsw_index.add_point(A_prep[i]) #; if (i+1)%(N_A//10+1)==0: print(f"  Added {i+1}/{N_A}...")
     end_build = time.time()
     build_time = end_build - start_build
     print(f"Index build time: {build_time:.2f} seconds")
     if hnsw_index.node_count == 0 or hnsw_index.entry_point == -1 : print("Error: Index build failed."); return torch.full((Q, K), -1, dtype=torch.int64, device=device), torch.full((Q, K), float('inf'), dtype=torch.float32, device=device), build_time, 0.0 # Added time return on failure
     start_search = time.time()
     all_indices = torch.full((Q, K), -1, dtype=torch.int64, device=device)
     all_distances = torch.full((Q, K), float('inf'), dtype=torch.float32, device=device)
     print("Searching queries..."); i=0
     for i in range(Q):
          results = hnsw_index.search_knn(X_prep[i], K) # Returns (squared_dist, node_id)
          num_results = len(results); k_actual = min(num_results, K)
          if num_results > 0:
               all_distances[i, :k_actual] = torch.tensor([res[0] for res in results[:k_actual]], device=device) # Store squared L2
               all_indices[i, :k_actual] = torch.tensor([res[1] for res in results[:k_actual]], device=device)
          # if (i+1)%(Q//10+1)==0: print(f"  Searched {i+1}/{Q}...")
     end_search = time.time()
     search_time = end_search - start_search
     print(f"ANN search time: {search_time:.4f} seconds")
     return all_indices, all_distances, build_time, search_time # Returns indices and SQUARED L2 distances

def our_ann_user_pseudocode_impl(N_A, D, A_cp, X_cp, k_clusters, K1, K2, max_kmeans_iters=100):
    """
    Implements the user's specific 4-step pseudocode using CuPy.
    Note: This finds the nearest K2 CLUSTER CENTERS according to the logic,
          not necessarily the nearest actual data points.

    Args:
        N_A (int): Number of database points (used for KMeans).
        D (int): Dimensionality.
        A_cp (cp.ndarray): Database vectors (N_A, D) on GPU (for KMeans).
        X_cp (cp.ndarray): Query vectors (Q, D) on GPU.
        k_clusters (int): Number of clusters for KMeans (Step 1).
        K1 (int): Number of nearest cluster centers to initially identify (Step 2).
        K2 (int): Number of nearest cluster centers to finally return from the K1 set (Step 3/4).
                  The final output size K will be K2.
        max_kmeans_iters (int): Max iterations for K-Means.

    Returns:
        tuple[cp.ndarray, cp.ndarray, float, float]:
            - topk2_centroid_indices_cp (cp.ndarray): Indices of the final K2 nearest *centroids* (Q, K2).
            - topk2_centroid_distances_sq_cp (cp.ndarray): Squared L2 distances to these K2 *centroids* (Q, K2).
            - build_time (float): Time taken for KMeans.
            - search_time (float): Time taken for the search logic (steps 2-4).
    """
    # --- Input Validation ---
    if not isinstance(A_cp, cp.ndarray): raise TypeError("A_cp must be CuPy ndarray.")
    if not isinstance(X_cp, cp.ndarray): raise TypeError("X_cp must be CuPy ndarray.")
    if A_cp.dtype != cp.float32:
        print("Warning: A_cp not float32. Converting.")
        A_cp = A_cp.astype(cp.float32)
    if X_cp.dtype != cp.float32:
        print("Warning: X_cp not float32. Converting.")
        X_cp = X_cp.astype(cp.float32)

    Q = X_cp.shape[0]
    if X_cp.shape[1] != D: raise ValueError(f"Query dimension mismatch: X D={X_cp.shape[1]}, expected D={D}")
    if A_cp.shape[0] != N_A: print(f"Warning: N_A mismatch ({N_A}) with A_cp shape ({A_cp.shape[0]}). Using shape from A_cp."); N_A = A_cp.shape[0]
    if A_cp.shape[1] != D: raise ValueError(f"Database dimension mismatch: A D={A_cp.shape[1]}, expected D={D}")
    if not (k_clusters > 0): raise ValueError("k_clusters must be positive")
    if not (K1 > 0): raise ValueError("K1 must be positive")
    if not (K2 > 0): raise ValueError("K2 must be positive")
    # K1 must be <= k_clusters, K2 must be <= K1. Adjust later if KMeans returns fewer.

    print(f"Running ANN (User Pseudocode Impl): Q={Q}, N={N_A}, D={D}")
    print(f"Params: k_clusters={k_clusters}, K1={K1}, K2={K2}")

    # --- Step 1: Use KMeans to cluster the data ---
    build_start_time = time.time()
    centroids_cp, _ = our_kmeans(N_A, D, A_cp, k_clusters, max_iters=max_kmeans_iters)
    if centroids_cp.dtype != cp.float32: centroids_cp = centroids_cp.astype(cp.float32)

    # Handle case where KMeans might return fewer than k_clusters
    actual_k_clusters = centroids_cp.shape[0]
    if actual_k_clusters == 0:
        print("Error: KMeans returned 0 centroids. Cannot proceed.")
        empty_indices = cp.full((Q, K2), -1, dtype=cp.int64)
        empty_dists = cp.full((Q, K2), cp.inf, dtype=cp.float32)
        # Return build_time=0 as it failed partially, search_time=0
        return empty_indices, empty_dists, time.time() - build_start_time, 0.0

    if actual_k_clusters < k_clusters:
        print(f"Warning: KMeans returned {actual_k_clusters} centroids, fewer than requested {k_clusters}.")
        k_clusters = actual_k_clusters # Use the actual number from now on

    # Adjust K1 and K2 based on the actual number of clusters available
    K1 = min(K1, k_clusters)
    K2 = min(K2, K1)
    if K1 == 0 or K2 == 0: # Need at least 1 cluster and K1/K2 > 0
        print("Error: K1 or K2 is 0 after adjustment or initially. Cannot proceed.")
        empty_indices = cp.full((Q, K2 if K2 > 0 else 1), -1, dtype=cp.int64)
        empty_dists = cp.full((Q, K2 if K2 > 0 else 1), cp.inf, dtype=cp.float32)
        return empty_indices, empty_dists, time.time() - build_start_time, 0.0

    cp.cuda.Stream.null.synchronize()
    build_time = time.time() - build_start_time
    print(f"Build time (KMeans): {build_time:.4f}s")

    # --- Search Phase ---
    search_start_time = time.time()

    # Calculate all query-centroid distances (vectorized)
    # Shape: (Q, k_clusters)
    all_query_centroid_dists_sq = pairwise_l2_squared_cupy(X_cp, centroids_cp)

    # --- Step 2: Find the nearest K1 cluster centers for each query ---
    # Use argpartition for efficiency to find indices of the K1 smallest distances.
    # k_partition value needs to be < N-1 for argpartition
    k1_partition = min(K1, k_clusters - 1) if k_clusters > 0 else 0
    if k1_partition < 0: k1_partition = 0 # Ensure non-negative

    if K1 >= k_clusters: # If K1 is all clusters, just sort all
        topk1_centroid_indices = cp.argsort(all_query_centroid_dists_sq, axis=1)[:, :K1]
    else:
        topk1_centroid_indices = cp.argpartition(all_query_centroid_dists_sq, k1_partition, axis=1)[:, :K1]
    # Shape (Q, K1) - Indices are original centroid indices (0 to k_clusters-1)

    # Get the squared distances corresponding to these K1 centroids
    # Shape: (Q, K1)
    topk1_centroid_dists_sq = cp.take_along_axis(all_query_centroid_dists_sq, topk1_centroid_indices, axis=1)

    # --- Step 3: Use KNN to find the nearest K2 neighbor *from the K1 cluster centers* ---
    # We now operate only on the K1 selected centroids and their distances per query.
    # Find the indices (relative to the K1 subset) of the K2 smallest distances among the K1 distances.
    k2_partition = min(K2, K1 - 1) if K1 > 0 else 0
    if k2_partition < 0: k2_partition = 0

    if K2 >= K1: # If K2 includes all K1, just sort the K1 distances
        relative_indices_k2 = cp.argsort(topk1_centroid_dists_sq, axis=1)[:, :K2] # Indices 0..K1-1
    else:
        relative_indices_k2 = cp.argpartition(topk1_centroid_dists_sq, k2_partition, axis=1)[:, :K2]
    # Shape (Q, K2) - Indices are relative to the K1 subset (0 to K1-1)

    # Get the distances for the final K2 centroids (values from topk1_centroid_dists_sq)
    # Shape: (Q, K2)
    topk2_subset_dists_sq = cp.take_along_axis(topk1_centroid_dists_sq, relative_indices_k2, axis=1)

    # Get the original centroid indices for the final K2 centroids by mapping back
    # Shape: (Q, K2)
    topk2_centroid_indices_cp = cp.take_along_axis(topk1_centroid_indices, relative_indices_k2, axis=1)

    # --- Step 4: Merge K1 * K2 vectors and find top K neighbors ---
    # Interpreting this as: return the K2 results found in Step 3, sorted by distance.
    # We already have the K2 indices and distances, just need to sort them.

    # Sort the final K2 results based on distance
    sort_order_k2 = cp.argsort(topk2_subset_dists_sq, axis=1) # Shape (Q, K2)

    # Apply the sort order to both the centroid indices and their distances
    final_topk2_centroid_indices_cp = cp.take_along_axis(topk2_centroid_indices_cp, sort_order_k2, axis=1)
    final_topk2_centroid_distances_sq_cp = cp.take_along_axis(topk2_subset_dists_sq, sort_order_k2, axis=1)

    cp.cuda.Stream.null.synchronize()
    search_time = time.time() - search_start_time
    print(f"Search time (User Pseudocode): {search_time:.4f}s")

    # Return the **centroid indices** (as int64 for consistency) and their distances
    return final_topk2_centroid_indices_cp.astype(cp.int64),final_topk2_centroid_distances_sq_cp,centroids_cp,build_time, search_time
# ============================================================================
# Example Usage (for the user's pseudocode implementation)
# ============================================================================
if __name__ == "__main__":
    # --- (Setup code: check cupy, set params, generate data A_cp, X_queries_cp) ---
    # ... (previous setup code remains the same) ...
    N_data = 100000
    Dim = 128
    N_queries = 500
    num_clusters_for_kmeans = 200
    K1_probe = 20 # K1: Probe nearest 20 centroids
    K2_final = 10 # K2: Final desired number of nearest centroids

    print("="*40)
    print("Generating Test Data (CuPy)...")
    # ... (data generation code) ...
    try:
        A_data_cp = cp.random.randn(N_data, Dim, dtype=cp.float32)
        X_queries_cp = cp.random.randn(N_queries, Dim, dtype=cp.float32)
    except Exception as e:
        print(f"Error generating data: {e}")
        exit()


    # --- Run ANN based on User's Pseudocode ---
    print("\n" + "="*40)
    print(f"Testing our_ann_user_pseudocode_impl (k_clusters={num_clusters_for_kmeans}, K1={K1_probe}, K2={K2_final})...")
    print("="*40)
    ann_indices_centroids = None # Initialize in case of error
    ann_dists_centroids = None
    centroids_cp = None
    build_t = 0
    search_t = 0
    try:
        # Call the modified function that now returns centroids
        ann_indices_centroids, ann_dists_centroids, centroids_cp, build_t, search_t = our_ann_user_pseudocode_impl(
            N_A=N_data, D=Dim, A_cp=A_data_cp, X_cp=X_queries_cp,
            k_clusters=num_clusters_for_kmeans,
            K1=K1_probe,
            K2=K2_final,
            max_kmeans_iters=50
        )
        # ... (print results as before) ...
        print("User Pseudocode ANN results shape (Centroid Indices):", ann_indices_centroids.shape)
        print("User Pseudocode ANN results shape (Squared Distances to Centroids):", ann_dists_centroids.shape)
        print(f"User Pseudocode ANN Build Time: {build_t:.4f}s")
        print(f"User Pseudocode ANN Search Time: {search_t:.4f}s")
        print(f"-> Throughput: {N_queries / search_t:.2f} queries/sec")

    # ... (exception handling as before) ...
    except Exception as e:
        print(f"Error during ANN execution: {e}")
        import traceback
        traceback.print_exc()


    # --- Calculate Recall against True Nearest Centroids ---
    if ann_indices_centroids is not None and centroids_cp is not None and centroids_cp.shape[0] > 0:
        print("\n" + "="*40)
        print("Calculating Recall vs. True Nearest Centroids...")
        print("="*40)

        K_recall = K2_final # Recall is @K2 for this algorithm

        try:
            # Calculate ground truth: True K2 nearest centroids for each query
            print("Calculating ground truth (Brute-force nearest centroids)...")
            start_gt = time.time()
            # Recompute all query-centroid distances (or get from function if modified)
            all_query_centroid_dists_sq_gt = pairwise_l2_squared_cupy(X_queries_cp, centroids_cp)

            # Find the indices of the true K2 nearest centroids using argsort
            # Ensure K_recall <= actual number of centroids
            actual_num_centroids = centroids_cp.shape[0]
            k_recall_adjusted = min(K_recall, actual_num_centroids)

            if k_recall_adjusted > 0:
                true_knn_centroid_indices = cp.argsort(all_query_centroid_dists_sq_gt, axis=1)[:, :k_recall_adjusted]
            else:
                true_knn_centroid_indices = cp.empty((N_queries, 0), dtype=cp.int64) # Handle zero case

            cp.cuda.Stream.null.synchronize()
            print(f"Ground truth calculation time: {time.time() - start_gt:.4f}s")


            # Compare results
            total_intersect_centroids = 0
            # Transfer to CPU for set operations
            ann_indices_np = cp.asnumpy(ann_indices_centroids[:, :k_recall_adjusted])
            true_indices_np = cp.asnumpy(true_knn_centroid_indices)

            for i in range(N_queries):
                # Get sets of centroid indices
                # Algorithm results might contain -1 if errors occurred, ignore them
                approx_centroid_ids = set(idx for idx in ann_indices_np[i] if idx >= 0)
                true_centroid_ids = set(true_indices_np[i]) # Ground truth won't have -1 unless k_recall_adjusted=0

                total_intersect_centroids += len(approx_centroid_ids.intersection(true_centroid_ids))

            if N_queries > 0 and k_recall_adjusted > 0:
                avg_recall_centroids = total_intersect_centroids / (N_queries * k_recall_adjusted)
                print(f"\nAverage Recall @ {k_recall_adjusted} (vs brute-force CENTROIDS): {avg_recall_centroids:.4f} ({avg_recall_centroids:.2%})")
                if avg_recall_centroids < 1.0 and K1_probe >= actual_num_centroids:
                     print("INFO: Recall is < 100% even though K1 >= num_clusters. Check logic if this occurs.")
                elif avg_recall_centroids < 0.9: # Example threshold
                     print(f"INFO: Recall is below target (e.g., 90%). This means the true {k_recall_adjusted} nearest centroids")
                     print(f"      were often not found within the initial {K1_probe} centroids probed.")
                     print(f"      Consider increasing K1_probe to improve centroid recall.")

            else:
                print("\nCannot calculate recall (N_queries=0 or K2=0).")

        except cp.cuda.memory.OutOfMemoryError as e:
            print(f"Error: Out of GPU memory during recall calculation. Details: {e}")
        except Exception as e:
            print(f"Error during recall calculation: {e}")
            import traceback
            traceback.print_exc()

    elif centroids_cp is None or centroids_cp.shape[0] == 0:
         print("\nCannot calculate recall: Centroids are not available or empty.")
    else:
         print("\nCannot calculate recall: ANN results are not available.")


'''
if __name__ == "__main__":
    # Ensure CuPy is available
    if device is None:
        print("CuPy device not available. Exiting example.")
        exit()

    N_data = 100000 # Increased size for better performance comparison
    Dim = 1024
    N_queries = 500
    K_val = 10
    K_clusters_ann = 200 # Number of clusters for ANN index (adjust as needed)
    N_probe_ann = 60     # Number of clusters to probe (adjust as needed)

    print("="*40)
    print("Generating Test Data (CuPy)...")
    print("="*40)
    # Generate data directly as CuPy arrays (float32 is crucial for speed)
    try:
        A_data_cp = cp.random.randn(N_data, Dim, dtype=cp.float32)
        X_queries_cp = cp.random.randn(N_queries, Dim, dtype=cp.float32)
        print(f"Database shape: {A_data_cp.shape}, Query shape: {X_queries_cp.shape}")
    except cp.cuda.memory.OutOfMemoryError as e:
        print(f"Error: Out of GPU memory generating data. Reduce N_data or Dim. Details: {e}")
        exit()
    except Exception as e:
        print(f"Error generating data: {e}")
        exit()

    # --- Run Brute-Force k-NN (CuPy) for ground truth ---
    print("\n" + "="*40)
    print(f"Running Brute-Force k-NN (CuPy) (K={K_val})...")
    print("="*40)
    knn_indices_cp = None # Initialize in case of error
    try:
        knn_indices_cp, knn_dists_sq_cp = cupy_knn_bruteforce(N_data, Dim, A_data_cp, X_queries_cp, K_val)
        print("KNN results shape (Indices):", knn_indices_cp.shape)
        print("KNN results shape (Squared Distances):", knn_dists_sq_cp.shape)
    except cp.cuda.memory.OutOfMemoryError as e:
        print(f"Error: Out of GPU memory during Brute-Force KNN. Reduce N_data or Q. Details: {e}")
    except Exception as e:
        print(f"Error during Brute-Force k-NN execution: {e}")
        import traceback
        traceback.print_exc()

    # --- Run Optimized ANN (KMeans-IVF / CuPy) ---
    print("\n" + "="*40)
    print(f"Testing our_ann_cupy_ivf_optimized (K={K_val})...")
    print("="*40)
    try:
        ann_indices_cp, ann_dists_sq_cp, build_t, search_t = our_ann_cupy_ivf_optimized(
            N_data, Dim, A_data_cp, X_queries_cp, K_val,
            k_clusters=K_clusters_ann, nprobe=N_probe_ann, max_kmeans_iters=50 # Reduce kmeans iters for faster example
        )
        print("ANN results shape (Indices):", ann_indices_cp.shape)
        print("ANN results shape (Squared Distances):", ann_dists_sq_cp.shape)
        print(f"ANN Build Time (Optimized): {build_t:.4f}s")
        print(f"ANN Search Time (Optimized): {search_t:.4f}s")
        print(f"-> Throughput: {N_queries / search_t:.2f} queries/sec")


        # --- Recall Calculation ---
        if knn_indices_cp is not None and ann_indices_cp is not None:
            # Ensure K used for recall matches K used in functions
            recall_K = K_val
            total_intersect = 0
            # Move necessary parts to CPU for set operations
            # Only move the top-K indices needed for recall calculation
            knn_indices_np = cp.asnumpy(knn_indices_cp[:, :recall_K])
            ann_indices_np = cp.asnumpy(ann_indices_cp[:, :recall_K])

            for i in range(N_queries):
                # Consider only valid indices (>= 0) from ANN results
                true_knn_ids = set(knn_indices_np[i])
                approx_ann_ids = set(idx for idx in ann_indices_np[i] if idx >= 0)

                intersect_count = len(true_knn_ids.intersection(approx_ann_ids))
                total_intersect += intersect_count

            if N_queries > 0 and recall_K > 0:
                # Recall is defined as (number of true neighbors found) / (total number of true neighbors)
                avg_recall = total_intersect / (N_queries * recall_K)
                print(f"\nAverage ANN Recall @ {recall_K} (vs brute-force CuPy KNN): {avg_recall:.4f} ({avg_recall:.2%})")
                # Set a reasonable recall target
                if avg_recall < 0.80: # Example target
                     print("INFO: Recall is below target threshold (e.g., 80%). Consider increasing nprobe.")
                if N_probe_ann >= K_clusters_ann and avg_recall < 0.999:
                    print("Warning: nprobe >= k_clusters, but recall is not ~100%. Check logic.")

            else:
                print("\nCannot calculate recall (N_queries or K_val is zero).")
        else:
            print("\nCannot calculate recall: Brute-force or ANN results unavailable.")

    except ImportError:
         print("Error: cupyx not found. Cannot use cupyx.scatter_add in K-Means.")
    except cp.cuda.memory.OutOfMemoryError as e:
        print(f"Error: Out of GPU memory during ANN. Reduce N_data, Q, K, or k_clusters. Details: {e}")
    except cp.cuda.runtime.CUDARuntimeError as e:
         print(f"CUDA runtime error during ANN execution: {e}")
    except Exception as e:
        print(f"Error during ANN execution: {e}")
        import traceback
        traceback.print_exc()
    D = 1024  # The dimension of vectors in your index (A_cp)
    Q_new = 500 # How many new queries you want to generate

# --- Generate ---
# Option A: Standard normal distribution (mean 0, variance 1)
    new_queries_cp = cp.random.randn(Q_new, D, dtype=cp.float32)

# Option B: Uniform distribution between [0, 1)
# new_queries_cp = cp.random.rand(Q_new, D, dtype=cp.float32)

    print(f"Generated {Q_new} new random queries with shape: {new_queries_cp.shape}")
    print(f"\nSearching with {new_queries_cp.shape[0]} newly generated queries...")

    try:
        D = 1024
        Q_new = 500
        noise_level = 0.05 # How much noise to add (adjust magnitude)
# Assume A_cp (the database used for indexing) is available
        N_A = A_data_cp.shape[0]

    # Call the search function with the new queries
        new_ann_indices, new_ann_dists, build_t_ignored, search_t_new = our_ann_cupy_ivf_optimized(
        N_A, D, A_data_cp,           # Original database info
        new_queries_cp,       # Your NEW query vectors
        K_val,                    # Number of neighbors
        k_clusters=K_clusters_ann, # Same index parameters
        nprobe=N_probe_ann,        # Same index parameters
        max_kmeans_iters=50        # K-Means iterations don't matter for search
        )

        print(f"New ANN search finished in: {search_t_new:.4f} seconds")
        print(f"-> Throughput: {new_queries_cp.shape[0] / search_t_new:.2f} queries/sec")
        print("New ANN results shape (Indices):", new_ann_indices.shape)
        print("New ANN results shape (Squared Distances):", new_ann_dists.shape)

    # You can now use new_ann_indices and new_ann_dists

    except Exception as e:
        print(f"Error during search with new queries: {e}")
        import traceback
        traceback.print_exc()
'''