import torch
import triton
import triton.language as tl
import time
import traceback # For error printing
import math # For ceil

# --- Device Setup ---
if not torch.cuda.is_available():
    print("CUDA not available, exiting.")
    exit()
device = torch.device("cuda:0")
print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")

# ============================================================================
# Helper Functions
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
# Triton Kernels & Distance Functions (PyTorch/Triton)
# ============================================================================

# --- Dot Product Kernel ---
@triton.jit
def dot_kernel_pairwise(
    X_ptr, A_ptr, Out_ptr,
    Q, N, D,
    stride_xq, stride_xd,
    stride_an, stride_ad,
    stride_outq, stride_outn,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Calculates pairwise dot products between X (Q, D) and A (N, D)."""
    pid_q = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    dot_prod = tl.zeros((), dtype=tl.float32)

    # Loop over dimension D in blocks
    for d_start in range(0, tl.cdiv(D, BLOCK_SIZE_D)):
        offs_d = d_start * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offs_d < D

        # Load X values for the current query and dimension block
        x_ptrs = X_ptr + pid_q * stride_xq + offs_d * stride_xd
        x_vals = tl.load(x_ptrs, mask=mask_d, other=0.0).to(tl.float32)

        # Load A values for the current db item and dimension block
        a_ptrs = A_ptr + pid_n * stride_an + offs_d * stride_ad
        a_vals = tl.load(a_ptrs, mask=mask_d, other=0.0).to(tl.float32)

        # Accumulate dot product
        dot_prod += tl.sum(x_vals * a_vals) # Sum over the dimension block

    # Write the final dot product to the output tensor
    out_offset = pid_q * stride_outq + pid_n * stride_outn
    tl.store(Out_ptr + out_offset, dot_prod)

# --- Tiled Dot Product Wrapper ---
DEFAULT_BLOCK_D_DOT = 128 # Can be tuned

def distance_dot_tiled(X, A, N_TILE=16384, prep=True):
    """ Computes pairwise dot products using the tiled Triton kernel. """
    if prep: X_prep, A_prep = _prepare_tensors(X, A)
    else: X_prep, A_prep = X, A

    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    if D != D_A: raise ValueError(f"Dimension mismatch: X({D}) vs A({D_A})")
    if Q == 0 or N == 0: return torch.empty((Q, N), dtype=torch.float32, device=device)

    Out = torch.empty((Q, N), dtype=torch.float32, device=device)

    # Process A in chunks to manage memory/grid size
    for n_start in range(0, N, N_TILE):
        n_end = min(n_start + N_TILE, N)
        N_chunk = n_end - n_start
        if N_chunk <= 0: continue

        A_chunk = A_prep[n_start:n_end, :] # Slice of A
        Out_chunk = Out[:, n_start:n_end] # Slice of Output

        grid = (Q, N_chunk)
        if grid[0] * grid[1] == 0: continue # Skip if empty grid

        # Launch Triton kernel for the chunk
        dot_kernel_pairwise[grid](
            X_prep, A_chunk, Out_chunk,
            Q, N_chunk, D,
            X_prep.stride(0), X_prep.stride(1),  # X strides
            A_chunk.stride(0), A_chunk.stride(1), # A_chunk strides
            Out_chunk.stride(0), Out_chunk.stride(1), # Out_chunk strides
            BLOCK_SIZE_D=DEFAULT_BLOCK_D_DOT # Pass block size
        )
    return Out

# --- Squared L2 Distance Function ---
def distance_l2_squared_pytorch(X, A):
    """Computes pairwise SQUARED L2 distances using PyTorch/Triton dot."""
    X_prep, A_prep = _prepare_tensors(X, A)
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    if D != D_A: raise ValueError("Dimension mismatch")
    if Q == 0 or N == 0: return torch.empty((Q, N), dtype=torch.float32, device=device)

    # Use optimized calculation: ||x - a||^2 = ||x||^2 + ||a||^2 - 2<x, a>
    try:
        dot_products = distance_dot_tiled(X_prep, A_prep, prep=False) # Triton kernel inside
        X_norm_sq = torch.sum(X_prep**2, axis=1, keepdim=True) # Shape (Q, 1)
        A_norm_sq = torch.sum(A_prep**2, axis=1, keepdim=True) # Shape (N, 1)

        # Broadcasting: (Q, 1) + (1, N) - 2 * (Q, N) -> (Q, N)
        dist_sq = X_norm_sq + A_norm_sq.T - 2 * dot_products
        dist_sq.clamp_(min=0.0) # Clamp numerical negatives
        return dist_sq

    except RuntimeError as e: # Catch potential CUDA OOM errors
        print(f"Error in distance_l2_squared_pytorch (shapes X:{X.shape}, A:{A.shape}): {e}")
        if "CUDA out of memory" in str(e):
             # Provide more context if possible
             dot_prod_mem_gb = Q * N * 4 / (1024**3)
             print(f"Estimated memory for dot product matrix: {dot_prod_mem_gb:.2f} GB")
        raise e # Re-raise


# --- K-Means Assignment Kernel (Autotuned) ---
@triton.autotune(
    configs=[
        # Basic configs varying block sizes (powers of 2 are common)
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K_CHUNK': 32, 'BLOCK_SIZE_D': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K_CHUNK': 64, 'BLOCK_SIZE_D': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K_CHUNK': 32, 'BLOCK_SIZE_D': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K_CHUNK': 64, 'BLOCK_SIZE_D': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K_CHUNK': 128, 'BLOCK_SIZE_D': 128}, num_warps=8),
        # Add more configs based on experimentation or GPU specs
    ],
    key=['N', 'D', 'K'], # Cache tuning result based on these dimensions
)
@triton.jit
def kmeans_assign_kernel(
    A_ptr,           # Pointer to data points (N, D) float32
    centroids_ptr,   # Pointer to centroids (K, D) float32
    assignments_ptr, # Pointer to output assignments (N,) int32
    N, D, K,
    stride_an, stride_ad,
    stride_ck, stride_cd,
    # These are now automatically set by the autotuner based on the chosen config
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K_CHUNK: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Assigns each point in A to the nearest centroid (Squared L2). Autotuned."""
    # Row indices for the current block of points this program is responsible for
    pid_n_base = tl.program_id(axis=0) * BLOCK_SIZE_N
    offs_n = pid_n_base + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N # Mask for points within the valid range [0, N)

    # Initialize minimum distance squared and best assignment for each point in the block
    min_dist_sq = tl.full((BLOCK_SIZE_N,), float('inf'), dtype=tl.float32)
    best_assignment = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32) - 1 # Initialize with -1

    # Pointer to the start of the rows for the current block of points
    # Shape: (BLOCK_SIZE_N, 1) - broadcasts along dimension D later
    points_block_ptr = A_ptr + offs_n[:, None] * stride_an

    # Iterate through centroids in chunks for potentially better cache usage
    for k_start in range(0, tl.cdiv(K, BLOCK_SIZE_K_CHUNK)):
        # Iterate within the chunk
        for k_offset in range(0, BLOCK_SIZE_K_CHUNK):
            k_idx = k_start * BLOCK_SIZE_K_CHUNK + k_offset
            if k_idx < K: # Check if the current centroid index is valid
                current_dist_sq = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
                # Pointer to the start of the current centroid's row
                centroid_row_ptr = centroids_ptr + k_idx * stride_ck

                # Iterate over dimension D in blocks
                for d_start in range(0, tl.cdiv(D, BLOCK_SIZE_D)):
                    offs_d = d_start * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
                    mask_d = offs_d < D # Mask for dimensions within the valid range [0, D)

                    # Load centroid values for the current dimension block
                    # Shape: (BLOCK_SIZE_D,)
                    centroid_d_ptr = centroid_row_ptr + offs_d * stride_cd
                    centroid_vals = tl.load(centroid_d_ptr, mask=mask_d, other=0.0)

                    # Load point values for the current block of points and dimension block
                    # Shape: (BLOCK_SIZE_N, BLOCK_SIZE_D)
                    points_d_ptr = points_block_ptr + offs_d[None, :] * stride_ad
                    # Apply masks for both points (rows) and dimensions (columns)
                    points_vals = tl.load(points_d_ptr, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

                    # Calculate difference and squared difference
                    diff = points_vals - centroid_vals[None, :] # Broadcast centroid vals
                    current_dist_sq += tl.sum(diff * diff, axis=1) # Sum across dimension block

                # Update minimum distance and assignment if current centroid is closer
                is_closer = current_dist_sq < min_dist_sq
                min_dist_sq = tl.where(is_closer, current_dist_sq, min_dist_sq)
                best_assignment = tl.where(is_closer, k_idx.to(tl.int32), best_assignment)

    # Write the final best assignments for the block of points
    assignments_out_ptrs = assignments_ptr + offs_n
    tl.store(assignments_out_ptrs, best_assignment, mask=mask_n)


# ============================================================================
# K-Means Implementation (PyTorch/Triton)
# ============================================================================

def our_kmeans_triton(N_A, D, A, K, max_iters=100, tol=1e-4, verbose=False):
    """
    Performs K-means using Autotuned Triton kernel for assignment
    and PyTorch scatter_add_ for the update step. Uses Squared L2 distance.

    Args:
        N_A (int): Number of data points (can be inferred).
        D (int): Dimension (can be inferred).
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
    A_prep = _prepare_tensors(A) # Ensure correct type, device, contiguity
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
    if K == 0: # K=0 after potential adjustment
         return torch.empty((0, D), dtype=torch.float32, device=device), \
                torch.empty((N_A,), dtype=torch.int64, device=device) # Assign all to -1 or 0?

    print(f"Running K-Means (AUTOTUNED Triton Assign + PyTorch Update): N={N_A}, D={D}, K={K}")
    start_time_total = time.time()

    # --- Initialization ---
    initial_indices = torch.randperm(N_A, device=device)[:K]
    centroids = A_prep[initial_indices].clone() # Shape (K, D)
    # assignments = torch.empty(N_A, dtype=torch.int64, device=device) # Final assignments type
    assignments_int32 = torch.empty(N_A, dtype=torch.int32, device=device) # Kernel output type

    # --- Grid calculation for autotuned kernel ---
    # Grid depends on BLOCK_SIZE_N chosen by autotuner, passed via meta
    grid_assign = lambda meta: (triton.cdiv(N_A, meta['BLOCK_SIZE_N']),)

    # --- Benchmarking/Timing variables ---
    assignment_times = []
    update_times = []

    for i in range(max_iters):
        t_iter_start = time.time()
        old_centroids = centroids.clone()

        # --- 1. Assignment Step (Autotuned Triton Kernel) ---
        t_assign_start = time.time()
        # Kernel expects int32 output pointer
        kmeans_assign_kernel[grid_assign](
            A_prep, centroids, assignments_int32, # Pass int32 tensor
            N_A, D, K,
            A_prep.stride(0), A_prep.stride(1), # Use strides
            centroids.stride(0), centroids.stride(1), # Use strides
            # Autotuner selects BLOCK_SIZE_N, BLOCK_SIZE_K_CHUNK, BLOCK_SIZE_D
        )
        torch.cuda.synchronize(device=device) # Wait for kernel
        t_assign_end = time.time()
        assignment_times.append(t_assign_end - t_assign_start)

        # Convert assignments for PyTorch update step (needs int64 index)
        assignments = assignments_int32.to(torch.int64)

        # --- 2. Update Step (PyTorch scatter_add_) ---
        t_update_start = time.time()
        new_sums = torch.zeros_like(centroids) # (K, D)
        cluster_counts = torch.zeros(K, dtype=torch.float32, device=device) # (K,)

        # Expand assignments to match dimensions for scatter_add_ index
        idx_expand = assignments.unsqueeze(1).expand(-1, D) # Shape (N_A, D)

        # Perform scatter add operations
        new_sums.scatter_add_(dim=0, index=idx_expand, src=A_prep)
        cluster_counts.scatter_add_(dim=0, index=assignments, src=torch.ones_like(assignments, dtype=torch.float32))
        torch.cuda.synchronize(device=device) # Wait for scatter ops

        # Handle empty clusters and calculate new centroids
        final_counts_safe = cluster_counts.clamp(min=1.0) # Avoid division by zero
        new_centroids = new_sums / final_counts_safe.unsqueeze(1) # Shape (K, D)

        # Restore old centroid position for any empty clusters
        empty_cluster_mask = (cluster_counts == 0)
        num_empty = torch.sum(empty_cluster_mask).item()
        if num_empty > 0:
            # print(f"  Iter {i+1}: Found {num_empty} empty clusters. Re-using old centroids.")
            new_centroids[empty_cluster_mask] = old_centroids[empty_cluster_mask]
            if num_empty == K:
                 print(f"Warning: Iter {i+1}, ALL clusters are empty. Stopping iteration.")
                 centroids = old_centroids # Restore previous state
                 break # Stop if all clusters became empty

        # Handle potential NaNs/Infs if scatter_add resulted in issues (less likely but possible)
        if not torch.all(torch.isfinite(new_centroids)):
             print(f"Warning: Iter {i+1}, found non-finite values in new_centroids. Replacing with old.")
             non_finite_mask = ~torch.isfinite(new_centroids)
             new_centroids[non_finite_mask] = old_centroids[non_finite_mask] # Replace only non-finite

        t_update_end = time.time()
        update_times.append(t_update_end - t_update_start)

        # --- Check Convergence ---
        # Compare new centroids to the ones from the start of the iteration
        centroid_diff = torch.linalg.norm(new_centroids - old_centroids)
        centroids = new_centroids # Update for next iteration

        if verbose and ((i+1) % 10 == 0 or centroid_diff < tol or i == max_iters -1):
            print(f"  Iter {i+1}/{max_iters} | Centroid Diff: {centroid_diff:.4f} | Assign Time: {assignment_times[-1]:.4f}s | Update Time: {update_times[-1]:.4f}s")

        if centroid_diff < tol:
            print(f"Converged after {i+1} iterations (centroid movement < {tol}).")
            break
        # Add stable check as alternative convergence
        # if torch.allclose(centroids, old_centroids, atol=tol/10):
        #      print(f"Converged after {i+1} iterations (centroids stable).")
        #      break


    if i == max_iters - 1: print(f"Reached max iterations ({max_iters}).")
    total_time = time.time() - start_time_total
    print(f"Total K-Means time: {total_time:.4f}s")

    # --- Basic Benchmarking Output ---
    if assignment_times: print(f"Avg Assign Step Time (Triton): {sum(assignment_times)/len(assignment_times):.6f}s")
    if update_times: print(f"Avg Update Step Time (PyTorch): {sum(update_times)/len(update_times):.6f}s")
    # Print the best config found by the autotuner for the assignment kernel
    print("Best config for kmeans_assign_kernel:", kmeans_assign_kernel.best_config)

    # Final check for non-empty clusters
    final_counts = torch.bincount(assignments.to(torch.int64), minlength=K)
    actual_k = torch.sum(final_counts > 0).item()
    print(f"K-Means finished. Found {actual_k} non-empty clusters out of {K} requested.")


    return centroids, assignments # Return final assignments (int64)


# ============================================================================
# ANN Function (PyTorch/Triton Version of IVF-like)
# ============================================================================

def ann_user_pseudocode_ivf_like_triton(
    N_A, D, A, X, K_final,
    num_clusters, num_clusters_to_probe,
    max_kmeans_iters=100,
    precomputed_centroids=None,
    precomputed_assignments=None,
    verbose_kmeans=False # Pass verbosity to kmeans
    ):
    """
    Performs ANN search based on user's pseudocode using PyTorch/Triton.
    Finds K_final nearest DATA POINTS.

    Args:
        N_A (int): Number of database points (inferred).
        D (int): Dimension (inferred).
        A (torch.Tensor): Database vectors (N_A, D), float32, GPU.
        X (torch.Tensor): Query vectors (Q, D), float32, GPU.
        K_final (int): Final number of nearest *data point* neighbors.
        num_clusters (int): Number of clusters for K-Means.
        num_clusters_to_probe (int): Number of nearest clusters to probe (K1).
        max_kmeans_iters (int): Max iterations for K-Means.
        precomputed_centroids (torch.Tensor, optional): (num_clusters, D) centroids.
        precomputed_assignments (torch.Tensor, optional): (N_A,) assignments (int64).
        verbose_kmeans (bool): Verbosity for KMeans function.

    Returns:
        tuple[torch.Tensor, torch.Tensor, float, float]:
            - all_indices (torch.Tensor): Indices (in A) of K_final nearest neighbors (Q, K_final). Int64.
            - all_distances_sq (torch.Tensor): **Squared** L2 distances (Q, K_final). Float32.
            - build_time (float): Time for K-Means + Inverted Index.
            - search_time (float): Time for searching all queries.
    """
    print("--- Starting Triton/PyTorch ANN ---")
    # --- Input Validation & Data Prep ---
    A_prep = _prepare_tensors(A)
    X_prep = _prepare_tensors(X)

    actual_N_A, actual_D = A_prep.shape
    Q, query_D = X_prep.shape

    N_A = actual_N_A
    D = actual_D
    if query_D != D: raise ValueError(f"Dimension mismatch: A D={D}, X D={query_D}")

    if N_A == 0: raise ValueError("Database A cannot be empty.")
    if Q == 0: print("Warning: Query set X is empty."); # Will return empty results

    if not (K_final > 0): raise ValueError("K_final must be positive")
    if not (num_clusters > 0): raise ValueError("num_clusters must be positive")
    if not (num_clusters_to_probe > 0): raise ValueError("num_clusters_to_probe must be positive")

    print(f"Running ANN (Triton/PyTorch IVF-like): Q={Q}, N={N_A}, D={D}, K_final={K_final}")
    print(f"Params: num_clusters={num_clusters}, nprobe={num_clusters_to_probe}")

    build_time_total = 0.0
    build_start_time = time.time()

    # --- Step 1: K-Means Clustering & Index Setup ---
    assignments = None
    centroids = None
    kmeans_run_time = 0.0
    if precomputed_centroids is not None and precomputed_assignments is not None:
        print("Using precomputed centroids and assignments.")
        centroids = _prepare_tensors(precomputed_centroids)
        assignments = _prepare_tensors(precomputed_assignments).to(torch.int64) # Ensure int64
        # Basic validation
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
        print("Running KMeans (Triton Assign)...")
        kmeans_start = time.time()
        # Cap K if needed (handled inside kmeans)
        centroids, assignments = our_kmeans_triton(N_A, D, A_prep, num_clusters, max_iters=max_kmeans_iters, verbose=verbose_kmeans)
        kmeans_run_time = time.time() - kmeans_start
        actual_num_clusters = centroids.shape[0]
        if actual_num_clusters < num_clusters: print(f"Note: KMeans used K={actual_num_clusters}.")
        num_clusters = actual_num_clusters # Use actual number

    # Handle case where KMeans might fail or return no clusters
    if num_clusters == 0 or centroids is None or assignments is None:
        print("Error: No clusters found or provided. Cannot proceed.")
        empty_indices = torch.full((Q, K_final), -1, dtype=torch.int64, device=device)
        empty_dists = torch.full((Q, K_final), float('inf'), dtype=torch.float32, device=device)
        build_time_total = time.time() - build_start_time # Include failed KMeans time
        return empty_indices, empty_dists, build_time_total, 0.0

    # Adjust num_clusters_to_probe
    num_clusters_to_probe = min(num_clusters_to_probe, num_clusters)
    if num_clusters_to_probe <= 0:
        print("Error: num_clusters_to_probe is 0. Cannot proceed.")
        # Handle returning empty results
        empty_indices = torch.full((Q, K_final), -1, dtype=torch.int64, device=device)
        empty_dists = torch.full((Q, K_final), float('inf'), dtype=torch.float32, device=device)
        build_time_total = time.time() - build_start_time
        return empty_indices, empty_dists, build_time_total, 0.0

    print(f"Building Inverted Index (PyTorch) for {num_clusters} clusters...")
    # --- Build Inverted Index (PyTorch) ---
    invidx_start_time = time.time()
    original_indices = torch.arange(N_A, dtype=torch.int64, device=device)

    # Sort original indices based on cluster assignments
    assignments = assignments.to(torch.int64) # Ensure int64 for sorting
    sort_permutation = torch.argsort(assignments)
    inv_idx_values = original_indices[sort_permutation] # Original indices sorted by cluster
    sorted_assignments = assignments[sort_permutation] # Cluster IDs sorted

    # Find unique cluster IDs present and their counts/first occurrences
    # unique_consecutive returns unique values, inverse indices, and counts
    unique_clusters, _, cluster_counts = torch.unique_consecutive(sorted_assignments, return_inverse=False, return_counts=True)

    # Calculate start indices: cumulative sum of counts of *previous* clusters
    cluster_starts = torch.zeros_like(cluster_counts)
    cluster_starts[1:] = torch.cumsum(cluster_counts[:-1], dim=0)

    # Create full lookup tables (size num_clusters) for starts and counts
    # Initialize with default values (-1 for start, 0 for count)
    full_inv_idx_starts = torch.full((num_clusters,), -1, dtype=torch.int64, device=device)
    full_inv_idx_counts = torch.zeros((num_clusters,), dtype=torch.int64, device=device)

    # Populate the tables using the unique cluster information
    # Ensure unique_clusters indices are within the valid range [0, num_clusters)
    valid_unique_mask = unique_clusters < num_clusters
    valid_unique_clusters = unique_clusters[valid_unique_mask]
    if valid_unique_clusters.numel() > 0: # Check if any valid clusters exist
         full_inv_idx_starts[valid_unique_clusters] = cluster_starts[valid_unique_mask]
         full_inv_idx_counts[valid_unique_clusters] = cluster_counts[valid_unique_mask]
    else:
         print("Warning: No valid unique clusters found during inverted index build.")


    invidx_end_time = time.time()
    build_time_total = kmeans_run_time + (invidx_end_time - invidx_start_time)
    print(f"Index build time (Total): {build_time_total:.4f}s (KMeans: {kmeans_run_time:.4f}s, InvIdx: {invidx_end_time - invidx_start_time:.4f}s)")

    # --- Search Phase ---
    search_start_time = time.time()
    all_indices = torch.full((Q, K_final), -1, dtype=torch.int64, device=device)
    all_distances_sq = torch.full((Q, K_final), float('inf'), dtype=torch.float32, device=device)

    # Handle empty query case
    if Q == 0:
         print("Empty query set, returning empty results.")
         return all_indices, all_distances_sq, build_time_total, 0.0

    # --- Step 2: Find nearest `num_clusters_to_probe` cluster centers ---
    print("Calculating query-centroid distances...")
    try:
        all_query_centroid_dists_sq = distance_l2_squared_pytorch(X_prep, centroids) # Shape (Q, num_clusters)
    except RuntimeError as e: # Catch OOM etc.
         print(f"Error calculating query-centroid distances: {e}")
         raise e

    print("Finding nearest clusters...")
    # Use torch.topk for finding nearest centroids (simpler than partition+sort here)
    # Returns distances and indices directly sorted
    try:
        _, all_nearest_cluster_indices = torch.topk(
            all_query_centroid_dists_sq,
            k=num_clusters_to_probe,
            dim=1,
            largest=False, # Find smallest distances
            sorted=False # Don't strictly need them sorted here
        ) # Shape (Q, num_clusters_to_probe)
    except Exception as e:
         print(f"Error during torch.topk for nearest clusters: {e}")
         raise e


    # --- Step 3 & 4: Gather Candidates & Find Top K_final data points ---
    print(f"Searching {Q} queries...")
    # Iterate through queries (CPU loop, GPU work inside)
    # Batching queries might be faster but complicates candidate handling
    for q_idx in range(Q):
        query = X_prep[q_idx:q_idx+1] # Keep 2D: (1, D)
        # Indices of clusters to probe for this query
        probed_cluster_indices = all_nearest_cluster_indices[q_idx] # Shape (num_clusters_to_probe,)

        # --- Gather candidate original indices (from A) ---
        # Get starts/counts for probed clusters using advanced indexing
        selected_starts = full_inv_idx_starts[probed_cluster_indices]
        selected_counts = full_inv_idx_counts[probed_cluster_indices]

        # Filter out potentially empty/invalid clusters (-1 start)
        valid_probe_mask = selected_starts >= 0
        if not torch.any(valid_probe_mask): continue # Skip if no valid clusters probed

        valid_starts = selected_starts[valid_probe_mask]
        valid_counts = selected_counts[valid_probe_mask]
        num_valid_probes = valid_starts.numel()

        # Efficiently create slices and concatenate candidate indices
        # This is tricky in PyTorch without explicit loops or list comprehensions.
        # Approach: Generate all indices using arange and masks based on starts/counts.
        # 1. Calculate end indices: ends = starts + counts
        # 2. Create a large arange covering max possible index range (potentially large) - maybe inefficient.
        # Alternative: List comprehension (CPU overhead) then torch.cat
        candidate_indices_list = []
        for i in range(num_valid_probes):
             start = valid_starts[i].item()
             count = valid_counts[i].item()
             if count > 0: # Ensure count is positive
                  candidate_indices_list.append(inv_idx_values[start : start + count])

        if not candidate_indices_list: continue # Skip if list is empty

        candidate_original_indices = torch.cat(candidate_indices_list)

        # Remove duplicates
        unique_candidate_original_indices = torch.unique(candidate_original_indices)
        num_unique_candidates = unique_candidate_original_indices.numel()
        if num_unique_candidates == 0: continue

        # --- Fetch candidate vectors ---
        try:
            # Ensure indices are valid before fetching (already done in inv idx build?)
            # Add check just in case:
            max_idx = torch.max(unique_candidate_original_indices)
            if max_idx >= N_A:
                 print(f"ERROR: Invalid candidate index {max_idx.item()} >= N_A ({N_A}) for query {q_idx}. Filtering.")
                 valid_cand_mask = unique_candidate_original_indices < N_A
                 unique_candidate_original_indices = unique_candidate_original_indices[valid_cand_mask]
                 num_unique_candidates = unique_candidate_original_indices.numel()
                 if num_unique_candidates == 0: continue

            candidate_vectors = A_prep[unique_candidate_original_indices] # Shape (num_unique, D)
        except RuntimeError as e: # Catch OOM
            print(f"OOM fetching candidates (Query {q_idx}, {num_unique_candidates} candidates): {e}")
            continue # Skip query
        except IndexError as e:
             print(f"IndexError fetching candidates (Query {q_idx}): {e}. Max index={max_idx.item() if 'max_idx' in locals() else 'N/A'}")
             continue # Skip query
        except Exception as e:
             print(f"Error fetching candidates (Query {q_idx}): {e}")
             continue


        # --- Calculate exact distances to candidates ---
        try:
            query_candidate_dists_sq = distance_l2_squared_pytorch(query, candidate_vectors) # Shape (1, num_unique)
        except RuntimeError as e: # Catch OOM
             print(f"OOM calculating query-candidate dists (Query {q_idx}, {num_unique_candidates} candidates): {e}")
             continue # Skip query
        except Exception as e:
             print(f"Error calculating query-candidate dists (Query {q_idx}): {e}")
             continue


        # --- Find top K_final among candidates ---
        actual_k_final = min(K_final, num_unique_candidates)
        if actual_k_final > 0:
            try:
                # Use torch.topk to get K smallest distances and their indices relative to candidates
                topk_dists_sq, topk_relative_indices = torch.topk(
                    query_candidate_dists_sq[0], # Operate on the 1D tensor of distances
                    k=actual_k_final,
                    largest=False, # Find smallest
                    sorted=True    # Get them sorted
                )

                # Map relative indices back to original indices from A_prep
                final_topk_original_indices = unique_candidate_original_indices[topk_relative_indices]

                # Store results
                all_indices[q_idx, :actual_k_final] = final_topk_original_indices
                all_distances_sq[q_idx, :actual_k_final] = topk_dists_sq

            except Exception as e:
                print(f"Error during top-K selection for query {q_idx}: {e}")
                # Leave results as -1/inf

    # --- Final Synchronization and Timing ---
    torch.cuda.synchronize(device=device) # Sync after all query loops
    search_time = time.time() - search_start_time
    print(f"ANN search time: {search_time:.4f} seconds")
    if search_time > 0 and Q > 0: print(f"-> Throughput: {Q / search_time:.2f} queries/sec")

    return all_indices, all_distances_sq, build_time_total, search_time


# ============================================================================
# Brute-Force k-NN (PyTorch/Triton Version)
# ============================================================================

def pytorch_knn_bruteforce(N_A, D, A, X, K):
    """
    Finds the K nearest neighbors using brute-force PyTorch/Triton distance.
    Returns original indices (int64) and SQUARED L2 distances (float32).
    Handles K > N_A by padding results.
    """
    print(f"Running k-NN Brute Force (PyTorch/Triton)...")
    A_prep = _prepare_tensors(A)
    X_prep = _prepare_tensors(X)

    actual_N_A, actual_D = A_prep.shape
    Q, query_D = X_prep.shape

    N_A = actual_N_A
    D = actual_D
    if query_D != D: raise ValueError(f"Dimension mismatch: A D={D}, X D={query_D}")

    # Handle empty cases
    if N_A == 0:
         print("Warning: Brute force called with empty database A.")
         return torch.full((Q, K), -1, dtype=torch.int64, device=device), \
                torch.full((Q, K), float('inf'), dtype=torch.float32, device=device)
    if Q == 0:
         print("Warning: Brute force called with empty query set X.")
         return torch.empty((0, K), dtype=torch.int64, device=device), \
                torch.empty((0, K), dtype=torch.float32, device=device)

    if not K > 0: raise ValueError("K must be positive")

    effective_K = min(K, N_A)
    if effective_K != K:
         print(f"Note: Brute force K={K} > N_A={N_A}. Using K={effective_K}.")
    if effective_K == 0:
         return torch.empty((Q, 0), dtype=torch.int64, device=device), \
                torch.empty((0, 0), dtype=torch.float32, device=device)

    print(f"Params: Q={Q}, N={N_A}, D={D}, K={effective_K}")
    start_time = time.time()

    # Calculate SQUARED L2 distances
    try:
        all_distances_sq = distance_l2_squared_pytorch(X_prep, A_prep) # Shape (Q, N_A)
    except RuntimeError as e: # Catch OOM
         print(f"OOM Error during Brute Force distance calculation: {e}")
         raise e
    except Exception as e:
         print(f"Error during Brute Force distance calculation: {e}")
         raise e


    # Find top K distances and indices using torch.topk
    try:
        topk_distances_sq, topk_indices = torch.topk(
            all_distances_sq,
            k=effective_K,
            dim=1,          # Along the N_A dimension
            largest=False,  # Find smallest distances
            sorted=True     # Ensure results are sorted by distance
        )
    except Exception as e:
         print(f"Error during Brute Force topk: {e}")
         raise e

    torch.cuda.synchronize(device=device) # Wait for topk
    end_time = time.time()
    print(f"k-NN Brute Force (PyTorch/Triton) computation time: {end_time - start_time:.4f} seconds")

    # Pad results if original K > effective_K (i.e., K > N_A)
    if K > effective_K:
        pad_width = K - effective_K
        indices_pad = torch.full((Q, pad_width), -1, dtype=torch.int64, device=device)
        dists_pad = torch.full((Q, pad_width), float('inf'), dtype=torch.float32, device=device)
        topk_indices = torch.cat((topk_indices, indices_pad), dim=1)
        topk_distances_sq = torch.cat((topk_distances_sq, dists_pad), dim=1)

    return topk_indices.to(torch.int64), topk_distances_sq


# ============================================================================
# Example Usage & Recall Calculation (Triton/PyTorch Version)
# ============================================================================

if __name__ == "__main__":
    # --- Parameters ---
    N_data = 1_000_000    # Database size
    Dim = 128           # Dimension
    N_queries = 10_000      # Queries
    K_final_neighbors = 10 # Final K for output

    # ANN Parameters
    num_clusters_kmeans = 1000  # K for KMeans (Step 1)
    num_clusters_probe = 50    # K1 (nprobe) for cluster probing (Step 2)
    kmeans_max_iters = 50      # Max iterations for KMeans

    # Recall threshold
    RECALL_THRESHOLD = 0.70

    print("\n" + "="*60)
    print("--- Triton/PyTorch ANN Example ---")
    print("="*60)
    print("Generating Test Data (PyTorch)...")
    print(f"N={N_data}, D={Dim}, Q={N_queries}, K_final={K_final_neighbors}")
    print(f"ANN Params: num_clusters={num_clusters_kmeans}, nprobe={num_clusters_probe}")
    print("="*60)
    A_data = None # Define outside try
    X_queries = None
    try:
        # Generate data directly on GPU
        print("Allocating memory for A_data...")
        A_data = torch.randn((N_data, Dim), dtype=torch.float32, device=device)
        print("Allocating memory for X_queries...")
        X_queries = torch.randn((N_queries, Dim), dtype=torch.float32, device=device)
        torch.cuda.synchronize(device=device) # Wait
        print("Data generated successfully on GPU.")
        # Optional: Check memory
        print(f"PyTorch Memory Allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB")
        print(f"PyTorch Memory Reserved: {torch.cuda.memory_reserved(device) / (1024**3):.2f} GB")

    except RuntimeError as e: # Catch CUDA OOM
        print(f"\nError: Out of GPU memory during data generation: {e}")
        print("Try reducing N_data or Dim.")
        torch.cuda.empty_cache() # Attempt to clear cache
        exit()
    except Exception as e:
        print(f"\nError generating data: {e}")
        exit()

    # --- Run ANN (Triton/PyTorch IVF-like) ---
    print("\n" + "="*60)
    print(f"Testing ANN (Triton/PyTorch IVF-like)...")
    print("="*60)
    ann_indices = None # Define outside try
    ann_dists_sq = None
    build_t = 0
    search_t = 0
    try:
        ann_indices, ann_dists_sq, build_t, search_t = ann_user_pseudocode_ivf_like_triton(
            N_A=N_data, D=Dim, A=A_data, X=X_queries,
            K_final=K_final_neighbors,
            num_clusters=num_clusters_kmeans,
            num_clusters_to_probe=num_clusters_probe,
            max_kmeans_iters=kmeans_max_iters,
            verbose_kmeans=False # Set to True for KMeans details
        )
        print("\nANN Results:")
        if ann_indices is not None: print(f"  Indices shape: {ann_indices.shape}")
        if ann_dists_sq is not None: print(f"  Sq Distances shape: {ann_dists_sq.shape}")
        print(f"  Build Time: {build_t:.4f}s")
        print(f"  Search Time: {search_t:.4f}s")

    except RuntimeError as e: # Catch OOM
        print(f"\nError: Out of GPU memory during ANN execution: {e}")
        ann_indices = None # Prevent recall
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\nError during ANN execution: {e}")
        traceback.print_exc()
        ann_indices = None # Prevent recall

    # --- Run Brute-Force KNN (PyTorch/Triton) ---
    true_knn_indices = None # Define outside try
    if ann_indices is not None: # Only run if ANN succeeded
        print("\n" + "="*60)
        print(f"Calculating Ground Truth (PyTorch/Triton k-NN)...")
        print("="*60)
        try:
            true_knn_indices, true_knn_dists_sq = pytorch_knn_bruteforce(
                N_A=N_data, D=Dim, A=A_data, X=X_queries, K=K_final_neighbors
            )
            print("\nGround Truth Results:")
            if true_knn_indices is not None: print(f"  Indices shape: {true_knn_indices.shape}")
            # if true_knn_dists_sq is not None: print(f"  Sq Distances shape: {true_knn_dists_sq.shape}") # Less interesting

        except RuntimeError as e: # Catch OOM
            print(f"\nError: Out of GPU memory during Brute Force k-NN: {e}")
            true_knn_indices = None # Prevent recall
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"\nError during Brute Force k-NN execution: {e}")
            traceback.print_exc()
            true_knn_indices = None # Prevent recall
        finally:
             if 'true_knn_dists_sq' in locals(): del true_knn_dists_sq # Clear dists

    # --- Calculate Recall ---
    if ann_indices is not None and true_knn_indices is not None:
        print("\n" + "="*60)
        print(f"Calculating Recall@{K_final_neighbors}...")
        print("="*60)

        try:
            print("Transferring indices to CPU for comparison...")
            start_recall_calc = time.time()
            # Transfer to CPU and convert to NumPy for set operations
            ann_indices_np = ann_indices.cpu().numpy()
            true_indices_np = true_knn_indices.cpu().numpy()
            print(f"Transfer time: {time.time() - start_recall_calc:.4f}s")

            total_intersect = 0
            # Calculate expected neighbors carefully, considering N_A
            expected_neighbors_per_query = min(K_final_neighbors, N_data)

            if N_queries > 0 and expected_neighbors_per_query > 0:
                for i in range(N_queries):
                    # Filter out potential -1 padding
                    ann_set = set(idx for idx in ann_indices_np[i] if idx >= 0)
                    true_set = set(idx for idx in true_indices_np[i] if idx >= 0)
                    total_intersect += len(ann_set.intersection(true_set))

                denominator = N_queries * expected_neighbors_per_query
                avg_recall = total_intersect / denominator if denominator > 0 else 1.0

                print(f"\nAverage Recall @ {K_final_neighbors} (vs {expected_neighbors_per_query} possible): {avg_recall:.4f} ({avg_recall:.2%})")

                if avg_recall >= RECALL_THRESHOLD:
                    print(f"Recall meets the threshold ({RECALL_THRESHOLD:.2%}). Result CORRECT.")
                else:
                    print(f"Recall is BELOW the threshold ({RECALL_THRESHOLD:.2%}). Result INCORRECT.")
                    print("Suggestions to improve recall:")
                    print(f" - Increase `num_clusters_to_probe` (currently {num_clusters_probe}).")
                    print(f" - Increase `num_clusters_kmeans` (currently {num_clusters_kmeans}).")
                    print(" - Check data normalization/preprocessing.")
            else:
                print("\nCannot calculate recall (N_queries=0 or K_final=0 or N_A=0).")

        except Exception as e:
            print(f"\nError during Recall calculation: {e}")
            traceback.print_exc()

    elif ann_indices is None:
         print("\nSkipping Recall: ANN execution failed or produced None.")
    elif true_knn_indices is None:
         print("\nSkipping Recall: Brute Force k-NN failed or produced None.")

    print("\n--- Triton/PyTorch Execution Finished ---")

    # --- Final Cleanup ---
    print("Cleaning up GPU memory...")
    del A_data
    del X_queries
    del ann_indices
    del ann_dists_sq
    del true_knn_indices
    torch.cuda.empty_cache()
    print("PyTorch CUDA cache cleared.")