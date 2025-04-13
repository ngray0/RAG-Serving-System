import torch
import triton
import triton.language as tl
import time
import traceback # For error printing
import math # For ceil
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
# Helper Functions (Same as before)
# ============================================================================
def _prepare_tensors(*tensors, target_device=device):
    # ... (definition remains the same) ...
    prepared = []
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            try: t = torch.tensor(t, dtype=torch.float32, device=target_device)
            except Exception as e: raise TypeError(f"Failed conversion: {e}")
        if t.device != target_device: t = t.to(target_device)
        if t.dtype != torch.float32: t = t.to(dtype=torch.float32)
        if not t.is_contiguous(): t = t.contiguous()
        prepared.append(t)
    return prepared[0] if len(prepared) == 1 else prepared

# ============================================================================
# Triton Kernels & Distance Functions (PyTorch/Triton - Dot Kernel same)
# ============================================================================

# --- Dot Product Kernel (dot_kernel_pairwise - Same as before) ---
@triton.jit
def dot_kernel_pairwise(
    X_ptr, A_ptr, Out_ptr,
    Q, N, D,
    stride_xq, stride_xd,
    stride_an, stride_ad,
    stride_outq, stride_outn,
    BLOCK_SIZE_D: tl.constexpr,
):
    # ... (definition remains the same) ...
    pid_q = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    dot_prod = tl.zeros((), dtype=tl.float32)
    for d_start in range(0, tl.cdiv(D, BLOCK_SIZE_D)):
        offs_d = d_start * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offs_d < D
        x_ptrs = X_ptr + pid_q * stride_xq + offs_d * stride_xd
        x_vals = tl.load(x_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        a_ptrs = A_ptr + pid_n * stride_an + offs_d * stride_ad
        a_vals = tl.load(a_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        dot_prod += tl.sum(x_vals * a_vals)
    out_offset = pid_q * stride_outq + pid_n * stride_outn
    tl.store(Out_ptr + out_offset, dot_prod)

# --- Tiled Dot Product Wrapper (distance_dot_tiled - Same as before) ---
DEFAULT_BLOCK_D_DOT = 128
def distance_dot_tiled(X, A, N_TILE=16384, prep=True):
    # ... (definition remains the same) ...
    if prep: X_prep, A_prep = _prepare_tensors(X, A)
    else: X_prep, A_prep = X, A
    Q, D = X_prep.shape; N, D_A = A_prep.shape
    if D != D_A: raise ValueError(f"Dimension mismatch: X({D}) vs A({D_A})")
    if Q == 0 or N == 0: return torch.empty((Q, N), dtype=torch.float32, device=device)
    Out = torch.empty((Q, N), dtype=torch.float32, device=device)
    for n_start in range(0, N, N_TILE):
        n_end = min(n_start + N_TILE, N); N_chunk = n_end - n_start
        if N_chunk <= 0: continue
        A_chunk = A_prep[n_start:n_end, :]; Out_chunk = Out[:, n_start:n_end]
        grid = (Q, N_chunk)
        if grid[0] * grid[1] == 0: continue
        dot_kernel_pairwise[grid](
            X_prep, A_chunk, Out_chunk, Q, N_chunk, D,
            X_prep.stride(0), X_prep.stride(1), A_chunk.stride(0), A_chunk.stride(1),
            Out_chunk.stride(0), Out_chunk.stride(1), BLOCK_SIZE_D=DEFAULT_BLOCK_D_DOT)
    return Out

# --- Squared L2 Distance Function (distance_l2_squared_pytorch - Same as before) ---
def distance_l2_squared_pytorch(X, A):
    # ... (definition remains the same, uses distance_dot_tiled) ...
    X_prep, A_prep = _prepare_tensors(X, A)
    Q, D = X_prep.shape; N, D_A = A_prep.shape
    if D != D_A: raise ValueError("Dimension mismatch")
    if Q == 0 or N == 0: return torch.empty((Q, N), dtype=torch.float32, device=device)
    try:
        dot_products = distance_dot_tiled(X_prep, A_prep, prep=False)
        X_norm_sq = torch.sum(X_prep**2, axis=1, keepdim=True)
        A_norm_sq = torch.sum(A_prep**2, axis=1, keepdim=True)
        dist_sq = X_norm_sq + A_norm_sq.T - 2 * dot_products
        dist_sq.clamp_(min=0.0)
        return dist_sq
    except RuntimeError as e:
        print(f"Error in distance_l2_squared_pytorch (shapes X:{X.shape}, A:{A.shape}): {e}")
        if "CUDA out of memory" in str(e):
             dot_prod_mem_gb = Q * N * 4 / (1024**3)
             print(f"Estimated memory for dot product matrix: {dot_prod_mem_gb:.2f} GB")
        raise e

# --- NEW: K-Means Assignment Kernel using Precomputed Norms ---
@triton.autotune(
    configs=[ # Same configs as before, maybe add more stages or experiment
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K_CHUNK': 32, 'BLOCK_SIZE_D': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K_CHUNK': 64, 'BLOCK_SIZE_D': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K_CHUNK': 64, 'BLOCK_SIZE_D': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K_CHUNK': 32, 'BLOCK_SIZE_D': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K_CHUNK': 128, 'BLOCK_SIZE_D': 128}, num_warps=8, num_stages=3),
        # Add a config with smaller K_CHUNK potentially?
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K_CHUNK': 16, 'BLOCK_SIZE_D': 128}, num_warps=4, num_stages=4),
    ],
    key=['N', 'D', 'K'],
)
@triton.jit
def kmeans_assign_kernel_prenorm(
    A_ptr, centroids_ptr,
    A_norm_sq_ptr, C_norm_sq_ptr, # Pointers to precomputed squared norms
    assignments_ptr,
    N, D, K,
    stride_an, stride_ad,
    stride_ck, stride_cd,
    stride_anorm, stride_cnorm, # Strides for norm arrays (usually 1)
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K_CHUNK: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Assigns points based on ||A||^2 + ||C||^2 - 2<A, C> using precomputed norms."""
    pid_n_base = tl.program_id(axis=0) * BLOCK_SIZE_N
    offs_n = pid_n_base + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N

    # Load precomputed squared norm for the current block of points
    A_norm_sq_ptrs = A_norm_sq_ptr + offs_n * stride_anorm
    a_norm_sq = tl.load(A_norm_sq_ptrs, mask=mask_n, other=0.0).to(tl.float32) # Shape (BLOCK_N,)

    min_dist_sq = tl.full((BLOCK_SIZE_N,), float('inf'), dtype=tl.float32)
    best_assignment = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32) - 1

    # Pointer to the start of the rows for the current block of points
    points_block_ptr = A_ptr + offs_n[:, None] * stride_an

    # Iterate through centroids in chunks
    for k_start in range(0, tl.cdiv(K, BLOCK_SIZE_K_CHUNK)):
        k_end = tl.minimum(k_start + BLOCK_SIZE_K_CHUNK, K)
        # Iterate within the chunk
        for k_offset in range(0, BLOCK_SIZE_K_CHUNK):
            k_idx = k_start * BLOCK_SIZE_K_CHUNK + k_offset
            if k_idx < k_end:
                # Load precomputed squared norm for the current centroid
                c_norm_sq_ptr = C_norm_sq_ptr + k_idx * stride_cnorm
                c_norm_sq = tl.load(c_norm_sq_ptr, mask=True, other=0.0).to(tl.float32) # Single value

                dot_prod = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
                centroid_row_ptr = centroids_ptr + k_idx * stride_ck

                # Compute dot product: <points_block, centroid_k>
                for d_start in range(0, tl.cdiv(D, BLOCK_SIZE_D)):
                    offs_d = d_start * BLOCK_SIZE_D + tl.arange(0, BLOCK_SIZE_D)
                    mask_d = offs_d < D

                    # Load centroid values for the current dimension block
                    centroid_d_ptr = centroid_row_ptr + offs_d * stride_cd
                    centroid_vals = tl.load(centroid_d_ptr, mask=mask_d, other=0.0) # (BLOCK_D,)

                    # Load point values for the current block of points and dimension block
                    points_d_ptr = points_block_ptr + offs_d[None, :] * stride_ad
                    points_vals = tl.load(points_d_ptr, mask=mask_n[:, None] & mask_d[None, :], other=0.0) # (BLOCK_N, BLOCK_D)

                    # Accumulate dot product using element-wise product and sum
                    dot_prod += tl.sum(points_vals * centroid_vals[None, :], axis=1) # Sum over dim block D

                # Calculate final squared distance using precomputed norms
                # dist_sq = ||A||^2 + ||C||^2 - 2 * <A, C>
                current_dist_sq = a_norm_sq + c_norm_sq - 2 * dot_prod

                # Update minimum distance and assignment
                is_closer = current_dist_sq < min_dist_sq
                min_dist_sq = tl.where(is_closer, current_dist_sq, min_dist_sq)
                best_assignment = tl.where(is_closer, k_idx.to(tl.int32), best_assignment)

    # Write the final best assignments for the block of points
    assignments_out_ptrs = assignments_ptr + offs_n
    tl.store(assignments_out_ptrs, best_assignment, mask=mask_n)

# ============================================================================
# K-Means Implementation (USING PRECOMPUTED NORM KERNEL)
# ============================================================================

def our_kmeans_triton_prenorm(N_A, D, A, K, max_iters=100, tol=1e-4, verbose=False):
    """
    Performs K-means using Autotuned Triton kernel (with precomputed norms)
    and PyTorch scatter_add_ for the update step.

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
    if K == 0:
         return torch.empty((0, D), dtype=torch.float32, device=device), \
                torch.empty((N_A,), dtype=torch.int64, device=device)

    print(f"Running K-Means (AUTOTUNED Triton PreNorm Assign + PyTorch Update): N={N_A}, D={D}, K={K}")
    start_time_total = time.time()

    # --- Initialization ---
    initial_indices = torch.randperm(N_A, device=device)[:K]
    centroids = A_prep[initial_indices].clone() # Shape (K, D)
    assignments_int32 = torch.empty(N_A, dtype=torch.int32, device=device) # Kernel output type

    # --- Precompute norms for A (once) ---
    print("Precomputing A norms...")
    t_norm_a_start = time.time()
    # Ensure norms are contiguous for the kernel
    A_norm_sq = torch.sum(A_prep**2, axis=1).contiguous() # Shape (N_A,)
    torch.cuda.synchronize(device=device)
    print(f"Precomputing A norms took: {time.time() - t_norm_a_start:.4f}s")


    # --- Grid calculation for autotuned kernel ---
    grid_assign = lambda meta: (triton.cdiv(N_A, meta['BLOCK_SIZE_N']),)

    # --- Benchmarking/Timing variables ---
    assignment_times = []
    update_times = []
    norm_c_times = []

    for i in range(max_iters):
        t_iter_start = time.time()
        old_centroids = centroids.clone()

        # --- 1. Assignment Step (Autotuned Triton PreNorm Kernel) ---
        t_assign_start = time.time()

        # --- Precompute norms for current centroids ---
        t_norm_c_start = time.time()
        C_norm_sq = torch.sum(centroids**2, axis=1).contiguous() # Shape (K,)
        torch.cuda.synchronize(device=device)
        norm_c_times.append(time.time() - t_norm_c_start)
        # --- End Precompute C norms ---

        # Call the prenorm kernel
        kmeans_assign_kernel_prenorm[grid_assign](
            A_prep, centroids,
            A_norm_sq, C_norm_sq, # Pass norms
            assignments_int32,
            N_A, D, K,
            A_prep.stride(0), A_prep.stride(1),
            centroids.stride(0), centroids.stride(1),
            A_norm_sq.stride(0), C_norm_sq.stride(0), # Norm strides (usually 1)
        )
        torch.cuda.synchronize(device=device) # Wait for kernel
        t_assign_end = time.time()
        assignment_times.append(t_assign_end - t_assign_start - norm_c_times[-1]) # Subtract norm time

        # Convert assignments for PyTorch update step (needs int64 index)
        assignments = assignments_int32.to(torch.int64)

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
                 centroids = old_centroids
                 break
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
            iter_time_total = (t_assign_end - t_assign_start) + (t_update_end - t_update_start)
            print(f"  Iter {i+1}/{max_iters} | Centroid Diff: {centroid_diff:.4f} | Assign Net: {assignment_times[-1]:.4f}s | Update: {update_times[-1]:.4f}s | C_Norm: {norm_c_times[-1]:.4f}s | Iter Total: {iter_time_total:.4f}s")

        if centroid_diff < tol:
            print(f"Converged after {i+1} iterations (centroid movement < {tol}).")
            break

    if i == max_iters - 1: print(f"Reached max iterations ({max_iters}).")
    total_time = time.time() - start_time_total
    print(f"Total K-Means time: {total_time:.4f}s")

    # --- Basic Benchmarking Output ---
    if assignment_times: print(f"Avg Assign Step Time (Triton PreNorm Kernel): {sum(assignment_times)/len(assignment_times):.6f}s")
    if norm_c_times: print(f"Avg Centroid Norm Calc Time (PyTorch): {sum(norm_c_times)/len(norm_c_times):.6f}s")
    if update_times: print(f"Avg Update Step Time (PyTorch): {sum(update_times)/len(update_times):.6f}s")
    print("Best config for kmeans_assign_kernel_prenorm:", kmeans_assign_kernel_prenorm.best_config)

    final_counts = torch.bincount(assignments.to(torch.int64), minlength=K)
    actual_k = torch.sum(final_counts > 0).item()
    print(f"K-Means finished. Found {actual_k} non-empty clusters out of {K} requested.")

    return centroids, assignments # Return final assignments (int64)


# ============================================================================
# ANN Function (PyTorch/Triton Version - UPDATED TO USE PRENORM KMEANS)
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
    Finds K_final nearest DATA POINTS. Uses optimized KMeans.

    Args:
        (Same as before)

    Returns:
        (Same as before)
    """
    print("--- Starting Triton/PyTorch ANN (Optimized KMeans) ---")
    # --- Input Validation & Data Prep (Same as before) ---
    A_prep = _prepare_tensors(A)
    X_prep = _prepare_tensors(X)
    actual_N_A, actual_D = A_prep.shape; Q, query_D = X_prep.shape
    N_A = actual_N_A; D = actual_D
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
    assignments = None; centroids = None; kmeans_run_time = 0.0
    if precomputed_centroids is not None and precomputed_assignments is not None:
        # --- Using precomputed (Same logic as before) ---
        print("Using precomputed centroids and assignments.")
        centroids = _prepare_tensors(precomputed_centroids)
        assignments = _prepare_tensors(precomputed_assignments).to(torch.int64)
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
        # --- Run Optimized KMeans ---
        print("Running KMeans (Triton PreNorm Assign)...")
        kmeans_start = time.time()
        # *** Call the new KMeans function ***
        centroids, assignments = our_kmeans_triton_prenorm(
            N_A, D, A_prep, num_clusters, max_iters=max_kmeans_iters, verbose=verbose_kmeans
        )
        kmeans_run_time = time.time() - kmeans_start
        actual_num_clusters = centroids.shape[0]
        if actual_num_clusters < num_clusters: print(f"Note: KMeans used K={actual_num_clusters}.")
        num_clusters = actual_num_clusters # Use actual number

    # --- Error Handling & Parameter Adjustment (Same as before) ---
    if num_clusters == 0 or centroids is None or assignments is None:
        print("Error: No clusters found or provided. Cannot proceed.")
        empty_indices = torch.full((Q, K_final), -1, dtype=torch.int64, device=device)
        empty_dists = torch.full((Q, K_final), float('inf'), dtype=torch.float32, device=device)
        build_time_total = time.time() - build_start_time
        return empty_indices, empty_dists, build_time_total, 0.0
    num_clusters_to_probe = min(num_clusters_to_probe, num_clusters)
    if num_clusters_to_probe <= 0:
        print("Error: num_clusters_to_probe is 0. Cannot proceed.")
        empty_indices = torch.full((Q, K_final), -1, dtype=torch.int64, device=device)
        empty_dists = torch.full((Q, K_final), float('inf'), dtype=torch.float32, device=device)
        build_time_total = time.time() - build_start_time
        return empty_indices, empty_dists, build_time_total, 0.0

    # --- Build Inverted Index (Same PyTorch logic as before) ---
    print(f"Building Inverted Index (PyTorch) for {num_clusters} clusters...")
    invidx_start_time = time.time()
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

    # --- Search Phase (Same logic as before) ---
    search_start_time = time.time()
    all_indices = torch.full((Q, K_final), -1, dtype=torch.int64, device=device)
    all_distances_sq = torch.full((Q, K_final), float('inf'), dtype=torch.float32, device=device)
    if Q == 0: return all_indices, all_distances_sq, build_time_total, 0.0 # Handle empty query

    # --- Step 2: Find nearest clusters (Same logic) ---
    print("Calculating query-centroid distances...")
    try: all_query_centroid_dists_sq = distance_l2_squared_pytorch(X_prep, centroids)
    except RuntimeError as e: print(f"Error calculating query-centroid distances: {e}"); raise e
    print("Finding nearest clusters...")
    try: _, all_nearest_cluster_indices = torch.topk(all_query_centroid_dists_sq, k=num_clusters_to_probe, dim=1, largest=False, sorted=False)
    except Exception as e: print(f"Error during torch.topk for nearest clusters: {e}"); raise e

    # --- Step 3 & 4: Gather/Search (Same logic - still has Python loop bottleneck) ---
    print(f"Searching {Q} queries...")
    for q_idx in range(Q):
        # --- Candidate Gathering (Same logic) ---
        query = X_prep[q_idx:q_idx+1]
        probed_cluster_indices = all_nearest_cluster_indices[q_idx]
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
        candidate_original_indices = torch.cat(candidate_indices_list)
        unique_candidate_original_indices = torch.unique(candidate_original_indices)
        num_unique_candidates = unique_candidate_original_indices.numel()
        if num_unique_candidates == 0: continue

        # --- Fetching Vectors (Same logic + error handling) ---
        try:
            max_idx = torch.max(unique_candidate_original_indices)
            if max_idx >= N_A:
                 valid_cand_mask = unique_candidate_original_indices < N_A
                 unique_candidate_original_indices = unique_candidate_original_indices[valid_cand_mask]
                 num_unique_candidates = unique_candidate_original_indices.numel()
                 if num_unique_candidates == 0: continue
            candidate_vectors = A_prep[unique_candidate_original_indices]
        except RuntimeError as e: print(f"OOM fetching candidates (Query {q_idx}, {num_unique_candidates} candidates): {e}"); continue
        except IndexError as e: print(f"IndexError fetching candidates (Query {q_idx}): {e}"); continue
        except Exception as e: print(f"Error fetching candidates (Query {q_idx}): {e}"); continue

        # --- Distance Calculation (Same logic + error handling) ---
        try: query_candidate_dists_sq = distance_l2_squared_pytorch(query, candidate_vectors)
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
# Brute-Force k-NN (PyTorch/Triton Version - Batched - Same as before)
# ============================================================================
def pytorch_knn_bruteforce(N_A, D, A, X, K, batch_size_q=256):
    # ... (definition remains the same as the previous batched version) ...
    print(f"Running k-NN Brute Force (PyTorch/Triton, Batched)...")
    A_prep = _prepare_tensors(A); X_prep = _prepare_tensors(X)
    actual_N_A, actual_D = A_prep.shape; Q, query_D = X_prep.shape
    N_A = actual_N_A; D = actual_D
    if query_D != D: raise ValueError(f"Dimension mismatch: A D={D}, X D={query_D}")
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
    for q_start in range(0, Q, batch_size_q):
        q_end = min(q_start + batch_size_q, Q); batch_q_indices = slice(q_start, q_end)
        X_batch = X_prep[batch_q_indices]; current_batch_size = X_batch.shape[0]
        if current_batch_size == 0: continue
        try: batch_distances_sq = distance_l2_squared_pytorch(X_batch, A_prep)
        except RuntimeError as e: batch_mem_gb = current_batch_size * N_A * 4 / (1024**3); print(f"OOM Error during Brute Force batch distance calculation:\n Batch Q={current_batch_size}, N={N_A}. Estimated matrix memory: {batch_mem_gb:.2f} GB"); torch.cuda.empty_cache(); raise e
        except Exception as e: print(f"Error during Brute Force batch distance calculation: {e}"); raise e
        try: batch_topk_distances_sq, batch_topk_indices = torch.topk(batch_distances_sq, k=effective_K, dim=1, largest=False, sorted=True)
        except Exception as e: print(f"Error during Brute Force batch topk ({q_start}-{q_end-1}): {e}"); raise e
        all_topk_indices[batch_q_indices] = batch_topk_indices; all_topk_distances_sq[batch_q_indices] = batch_topk_distances_sq
        del batch_distances_sq, batch_topk_indices, batch_topk_distances_sq, X_batch
    torch.cuda.synchronize(device=device)
    end_time = time.time(); print(f"k-NN Brute Force (PyTorch/Triton, Batched) total computation time: {end_time - start_time:.4f} seconds")
    if K > effective_K:
        pad_width = K - effective_K; indices_pad = torch.full((Q, pad_width), -1, dtype=torch.int64, device=device); dists_pad = torch.full((Q, pad_width), float('inf'), dtype=torch.float32, device=device)
        if all_topk_indices is None: all_topk_indices = torch.full((Q, effective_K), -1, dtype=torch.int64, device=device)
        if all_topk_distances_sq is None: all_topk_distances_sq = torch.full((Q, effective_K), float('inf'), dtype=torch.float32, device=device)
        all_topk_indices = torch.cat((all_topk_indices, indices_pad), dim=1); all_topk_distances_sq = torch.cat((all_topk_distances_sq, dists_pad), dim=1)
    if all_topk_indices is None: all_topk_indices = torch.full((Q, K), -1, dtype=torch.int64, device=device)
    if all_topk_distances_sq is None: all_topk_distances_sq = torch.full((Q, K), float('inf'), dtype=torch.float32, device=device)
    return all_topk_indices.to(torch.int64), all_topk_distances_sq.to(torch.float32)


# ============================================================================
# Example Usage & Recall Calculation (UPDATED Main Block)
# ============================================================================

# ============================================================================
# Main Execution Block (Triton/PyTorch - Warmup & Dimension Loop)
# ============================================================================
if __name__ == "__main__":
    # --- Fixed Parameters ---
    N_data = 1000000    # Database size
    # Dim will be set in the loop # Dim = 128
    N_queries = 1000     # Queries
    K_final_neighbors = 10 # Final K for output

    # ANN Parameters
    num_clusters_kmeans = 50000 # K for KMeans (Step 1)
    num_clusters_probe = 3   # K1 (nprobe) for cluster probing (Step 2)
    kmeans_max_iters = 100      # Max iterations for KMeans

    # Recall threshold
    RECALL_THRESHOLD = 0.70

    # Dimensions to test
    dimensions_to_test = [2,2, 2, 4,4, 4, 64, 256, 1024]

    print("\n" + "="*60)
    print("--- Triton/PyTorch ANN Full Test ---")
    print("="*60)
    print(f"Fixed Params: N={N_data}, Q={N_queries}, K_final={K_final_neighbors}")
    print(f"ANN Params: num_clusters={num_clusters_kmeans}, nprobe={num_clusters_probe}")
    print(f"Testing Dimensions: {dimensions_to_test}")

    # --- Warmup Phase ---
    print("\n" + "="*60)
    print("--- Warmup Run (Compiling Kernels...) ---")
    print("="*60)
    try:
        WARMUP_N = 10000  # Smaller N for warmup
        WARMUP_Q = 100   # Smaller Q for warmup
        WARMUP_DIM = 32  # Representative dimension
        WARMUP_K_CLUSTERS = 50 # Fewer clusters
        WARMUP_NPROBE = 5
        WARMUP_KFINAL = 5

        print(f"Warmup Params: N={WARMUP_N}, D={WARMUP_DIM}, Q={WARMUP_Q}, K={WARMUP_KFINAL}, Clusters={WARMUP_K_CLUSTERS}, NProbe={WARMUP_NPROBE}")

        # Generate small warmup data
        A_warmup = torch.randn((WARMUP_N, WARMUP_DIM), dtype=torch.float32, device=device)
        X_warmup = torch.randn((WARMUP_Q, WARMUP_DIM), dtype=torch.float32, device=device)
        torch.cuda.synchronize(device=device)

        # Run ANN function (which includes KMeans)
        print("Warmup: Running ANN (includes KMeans)...")
        _, _, _, _ = ann_user_pseudocode_ivf_like_triton(
            N_A=WARMUP_N, D=WARMUP_DIM, A=A_warmup, X=X_warmup,
            K_final=WARMUP_KFINAL,
            num_clusters=WARMUP_K_CLUSTERS,
            num_clusters_to_probe=WARMUP_NPROBE,
            max_kmeans_iters=5, # Fewer iters for warmup
            verbose_kmeans=False
        )
        print("Warmup: Running Brute Force KNN...")
        # Run Brute Force function
        _, _ = pytorch_knn_bruteforce(
            N_A=WARMUP_N, D=WARMUP_DIM, A=A_warmup, X=X_warmup, K=WARMUP_KFINAL, batch_size_q=64
        )
        torch.cuda.synchronize(device=device)
        print("Warmup complete.")

    except Exception as e:
        print(f"An error occurred during warmup: {e}")
        print("Continuing without warmup...")
    finally:
        # Cleanup warmup data
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

        # --- Per-Dimension Variables ---
        A_data = None
        X_queries = None
        ann_indices = None
        ann_dists_sq = None
        true_knn_indices = None
        build_t = 0
        search_t = 0

        try:
            # --- Generate Data for Current Dimension ---
            print(f"\n[D={Dim}] Generating Test Data (PyTorch)...")
            try:
                A_data = torch.randn((N_data, Dim), dtype=torch.float32, device=device)
                X_queries = torch.randn((N_queries, Dim), dtype=torch.float32, device=device)
                torch.cuda.synchronize(device=device)
                print(f"[D={Dim}] Data generated. Mem Allocated: {torch.cuda.memory_allocated(device)/(1024**3):.2f} GB")
            except RuntimeError as e:
                print(f"\n[D={Dim}] ERROR: OOM during data generation: {e}")
                torch.cuda.empty_cache();
                continue # Skip to next dimension
            except Exception as e:
                print(f"\n[D={Dim}] ERROR generating data: {e}")
                continue # Skip to next dimension

            # --- Run ANN for Current Dimension ---
            print(f"\n[D={Dim}] Testing ANN (Triton/PyTorch IVF-like)...")
            try:
                ann_indices, ann_dists_sq, build_t, search_t = ann_user_pseudocode_ivf_like_triton(
                    N_A=N_data, D=Dim, A=A_data, X=X_queries, # Use current Dim data
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
            except RuntimeError as e:
                print(f"\n[D={Dim}] ERROR: OOM during ANN execution: {e}")
                ann_indices = None; torch.cuda.empty_cache() # Prevent recall
            except Exception as e:
                print(f"\n[D={Dim}] ERROR during ANN execution: {e}")
                traceback.print_exc(); ann_indices = None # Prevent recall

            # --- Run Brute-Force KNN for Current Dimension ---
            if ann_indices is not None: # Only run if ANN succeeded
                print(f"\n[D={Dim}] Calculating Ground Truth (PyTorch/Triton k-NN)...")
                try:
                    true_knn_indices, true_knn_dists_sq = pytorch_knn_bruteforce(
                        N_A=N_data, D=Dim, A=A_data, X=X_queries, K=K_final_neighbors # Use current Dim data
                    )
                    print(f"\n[D={Dim}] Ground Truth Results:")
                    if true_knn_indices is not None: print(f"  Indices shape: {true_knn_indices.shape}")
                except RuntimeError as e:
                    print(f"\n[D={Dim}] ERROR: OOM during Brute Force k-NN: {e}")
                    true_knn_indices = None; torch.cuda.empty_cache() # Prevent recall
                except Exception as e:
                    print(f"\n[D={Dim}] ERROR during Brute Force k-NN execution: {e}")
                    traceback.print_exc(); true_knn_indices = None # Prevent recall
                finally:
                     if 'true_knn_dists_sq' in locals(): del true_knn_dists_sq

            # --- Calculate Recall for Current Dimension ---
            if ann_indices is not None and true_knn_indices is not None:
                print(f"\n[D={Dim}] Calculating Recall@{K_final_neighbors}...")
                try:
                    # print(f"[D={Dim}] Transferring indices to CPU...")
                    start_recall_calc = time.time()
                    ann_indices_np = ann_indices.cpu().numpy()
                    true_indices_np = true_knn_indices.cpu().numpy()
                    # print(f" Transfer time: {time.time() - start_recall_calc:.4f}s")
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
            # --- Cleanup for Current Dimension ---
            print(f"\n[D={Dim}] Cleaning up tensors...")
            del A_data
            del X_queries
            del ann_indices
            del ann_dists_sq
            del true_knn_indices
            # Conditional delete for variables created inside try blocks
            if 'ann_indices_np' in locals(): del ann_indices_np
            if 'true_indices_np' in locals(): del true_indices_np
            torch.cuda.empty_cache()
            print(f"[D={Dim}] Cleanup complete. Mem Allocated: {torch.cuda.memory_allocated(device)/(1024**3):.2f} GB")

        print(f"\n--- Finished Test for Dimension D = {Dim} ---")

    print("\n" + "="*60)
    print("--- ALL DIMENSION TESTS FINISHED ---")
    print("="*60)