import torch
import triton
import triton.language as tl
import time
import traceback # For error printing
# Remove numpy import if not needed elsewhere
# import numpy as np

# --- Device Setup ---
if not torch.cuda.is_available():
    print("CUDA not available, exiting.")
    exit()
device = torch.device("cuda:0")
print(f"Using device: {device}")

# Assume a default block size if not defined elsewhere
DEFAULT_BLOCK_D = 128 # Or choose another appropriate power of 2

# --- Triton Kernels ---
@triton.jit
def dot_kernel_pairwise(
    X_ptr, A_ptr, Out_ptr,
    Q, N, D,
    stride_xq, stride_xd,
    stride_an, stride_ad,
    stride_outq, stride_outn,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Calculates pairwise dot product: dot(X[q], A[n]) -> storing the POSITIVE dot product"""
    pid_q = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    # Use float32 for accumulation inside kernel for performance, can cast output later if needed
    dot_prod = tl.zeros((), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_SIZE_D):
        # Corrected range calculation for tl.arange
        offs_d = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offs_d < D # Correct mask based on actual dimension D

        x_ptrs = X_ptr + pid_q * stride_xq + offs_d # Stride for D is usually 1
        x_vals = tl.load(x_ptrs, mask=mask_d, other=0.0).to(tl.float32)

        a_ptrs = A_ptr + pid_n * stride_an + offs_d # Stride for D is usually 1
        a_vals = tl.load(a_ptrs, mask=mask_d, other=0.0).to(tl.float32)

        dot_prod += tl.sum(x_vals * a_vals, axis=0) # Axis=0 sums across the BLOCK_SIZE_D dimension

    out_offset = pid_q * stride_outq + pid_n * stride_outn
    tl.store(Out_ptr + out_offset, dot_prod) # Store the positive dot product

@triton.jit
def kmeans_assign_kernel(
    A_ptr,           # Pointer to data points (N, D) float32
    centroids_ptr,   # Pointer to centroids (K, D) float32
    assignments_ptr, # Pointer to output assignments (N,) int32
    N, D, K,
    stride_an, stride_ad,
    stride_ck, stride_cd,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K_CHUNK: tl.constexpr, # How many centroids to process per inner loop
    BLOCK_SIZE_D: tl.constexpr, # How many dimensions to process per innermost loop
):
    """Assigns each point in A to the nearest centroid (Squared L2)"""
    pid_n_base = tl.program_id(axis=0) * BLOCK_SIZE_N # Base index for points in this block
    offs_n = pid_n_base + tl.arange(0, BLOCK_SIZE_N) # Range of point indices for this block
    mask_n = offs_n < N # Mask for valid points in this block

    min_dist_sq = tl.full((BLOCK_SIZE_N,), float('inf'), dtype=tl.float32)
    best_assignment = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32) # Kernel outputs int32

    # Pointers to point data for this block (doesn't change with k)
    points_block_ptr = A_ptr + offs_n[:, None] * stride_an # Base ptr for each point row

    for k_start in range(0, K, BLOCK_SIZE_K_CHUNK):
        k_end = tl.minimum(k_start + BLOCK_SIZE_K_CHUNK, K)
        # Iterate through centroids within the current chunk
        for k_offset in range(BLOCK_SIZE_K_CHUNK):
            k_idx = k_start + k_offset
            if k_idx < k_end: # Process only valid centroids in the chunk
                current_dist_sq = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
                # Pointer to the current centroid row
                centroid_row_ptr = centroids_ptr + k_idx * stride_ck

                # Iterate through dimensions
                for d_start in range(0, D, BLOCK_SIZE_D):
                    offs_d = d_start + tl.arange(0, BLOCK_SIZE_D)
                    mask_d = offs_d < D

                    # Load centroid dimension block
                    centroid_d_ptr = centroid_row_ptr + offs_d * stride_cd
                    centroid_vals = tl.load(centroid_d_ptr, mask=mask_d, other=0.0) # Shape (BLOCK_D,)

                    # Load points dimension block
                    # points_block_ptr shape (BLOCK_N, 1) -> points_d_ptr shape (BLOCK_N, BLOCK_D)
                    points_d_ptr = points_block_ptr + offs_d[None, :] * stride_ad
                    # Need mask based on both n and d validity
                    points_vals = tl.load(points_d_ptr, mask=mask_n[:, None] & mask_d[None, :], other=0.0) # Shape (BLOCK_N, BLOCK_D)

                    # Calculate difference and accumulate squared distance
                    diff = points_vals - centroid_vals[None, :] # Broadcast centroid vals
                    current_dist_sq += tl.sum(diff * diff, axis=1) # Sum across dimension blocks

                # Update minimum distance and assignment
                is_closer = current_dist_sq < min_dist_sq
                min_dist_sq = tl.where(is_closer, current_dist_sq, min_dist_sq)
                best_assignment = tl.where(is_closer, k_idx, best_assignment) # Store absolute centroid index

    # Store final assignments for this block of points
    assignments_out_ptrs = assignments_ptr + offs_n
    tl.store(assignments_out_ptrs, best_assignment, mask=mask_n)


def _prepare_tensors(*tensors, target_device=device):
    """
    Ensure tensors are float32, contiguous, and on the correct device.
    Modified return to try and satisfy specific unpacking requirement.
    """
    prepared = []
    for t in tensors:
        # --- (Keep the preparation logic inside the loop exactly as provided by user) ---
        if not isinstance(t, torch.Tensor):
            try:
                t = torch.tensor(t, dtype=torch.float32, device=target_device)
            except Exception as e:
                raise TypeError(f"Failed to convert input of type {type(t)} to torch.Tensor: {e}")
        if t.device != target_device:
            t = t.to(target_device)
        if t.dtype != torch.float32:
            t = t.to(dtype=torch.float32)
        if not t.is_contiguous():
            t = t.contiguous()
        # --- (End of preparation logic) ---
        prepared.append(t) # Append the prepared tensor

    # --- Return Logic ---
    if len(prepared) == 1:
        # If only one tensor was passed in, return a TUPLE containing that single tensor
        # This explicitly creates an iterable of length 1 for the unpacking `var, = ...`
        return (prepared[0],) # Note the trailing comma to make it a tuple
    else:
        # If multiple tensors were passed in, return the list of prepared tensors
        # (Unpacking `var1, var2 = ...` works correctly with lists/tuples of length > 1)
        return prepared

# --- Distance Functions (Using Triton Dot Kernel) ---
def distance_dot_tiled(X, A, N_TILE=32768, prep=True): # Adjusted tile size, needs tuning
    """
    Computes pairwise dot product using Triton kernel, tiled over A.
    Returns POSITIVE dot product.
    """
    if prep: X_prep, A_prep = _prepare_tensors(X, A)
    else: X_prep, A_prep = X, A # Assume already prepared

    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    if D != D_A: raise ValueError(f"Dimension mismatch: X({D}) vs A({D_A})")

    # Output uses float32 matching kernel accumulation
    Out = torch.empty((Q, N), dtype=torch.float32, device=device)

    # print(f"Tiling dot product calculation with N_TILE={N_TILE}")
    for n_start in range(0, N, N_TILE):
        n_end = min(n_start + N_TILE, N)
        N_chunk = n_end - n_start
        if N_chunk <= 0: continue # Skip if somehow chunk size is zero or negative

        A_chunk = A_prep[n_start:n_end, :]
        # Create a view for the output chunk
        Out_chunk = Out[:, n_start:n_end]

        grid = (Q, N_chunk) # Grid dimensions based on current chunk
        if grid[0] == 0 or grid[1] == 0: continue # Skip empty grids

        # Use default strides (1 for dim D) assuming contiguous tensors
        dot_kernel_pairwise[grid](
            X_prep, A_chunk, Out_chunk,
            Q, N_chunk, D,
            X_prep.stride(0), 1, # Stride D is 1 for contiguous
            A_chunk.stride(0), 1, # Stride D is 1 for contiguous
            Out_chunk.stride(0), 1,# Stride D is 1 for contiguous view
            BLOCK_SIZE_D=DEFAULT_BLOCK_D
        )
        # torch.cuda.synchronize() # Debugging sync point

    return Out # Return POSITIVE dot product

def distance_l2(X, A, **kwargs):
    """
    Computes pairwise SQUARED L2 (Euclidean) distances using the tiled dot product kernel
    and PyTorch operations for norms. Returns SQUARED L2 distance.
    """
    X_prep, A_prep = _prepare_tensors(X, A) # Ensure prepared tensors
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    if D != D_A: raise ValueError(f"Dimension mismatch: X({D}) vs A({D_A})")

    # Calculate dot products (positive values)
    dot_products = distance_dot_tiled(X_prep, A_prep, prep=False, **kwargs) # Shape (Q, N)

    # Calculate squared norms
    X_norm_sq = torch.sum(X_prep**2, axis=1, keepdims=True)  # Shape (Q, 1)
    A_norm_sq = torch.sum(A_prep**2, axis=1, keepdims=True)  # Shape (N, 1)

    # ||X-A||^2 = ||X||^2 + ||A||^2 - 2*dot(X,A)
    dist_sq = X_norm_sq + A_norm_sq.T - 2 * dot_products # Shape (Q, N)

    # Clamp results to ensure non-negativity due to potential floating point issues
    dist_sq.clamp_(min=0.0)
    return dist_sq # Return squared L2 distance

# --- Triton KMeans Implementation ---
def our_kmeans(N_A, D, A, K, max_iters=100, tol=1e-4):
    """
    Performs K-means clustering on data A using Triton kernel for assignment
    and PyTorch scatter_add_ for the update step. Uses L2 distance.
    """
    A_prep, = _prepare_tensors(A) # Ensure data is prepared
    if A_prep.shape[0] != N_A or A_prep.shape[1] != D: N_A, D = A_prep.shape # Use actual shape
    if not (K > 0): raise ValueError("K must be positive.")
    if K > N_A: K = N_A # Adjust K if needed

    # print(f"Running K-Means (Triton Assign + PyTorch Update): N={N_A}, D={D}, K={K}")
    start_time_total = time.time()

    if K == 0: return torch.empty((0, D), dtype=torch.float32, device=device), \
                      torch.empty((N_A,), dtype=torch.int64, device=device)

    # --- Initialization ---
    initial_indices = torch.randperm(N_A, device=device)[:K]
    centroids = A_prep[initial_indices].clone()
    assignments = torch.empty(N_A, dtype=torch.int64, device=device) # For scatter_add_
    old_centroids = torch.empty_like(centroids)

    # --- Triton Kernel Launch Configuration ---
    # BLOCK sizes might need tuning based on GPU architecture and D
    BLOCK_SIZE_N_ASSIGN = 128
    BLOCK_SIZE_K_CHUNK_ASSIGN = 64 # Process 64 centroids per block iteration
    BLOCK_SIZE_D_ASSIGN = DEFAULT_BLOCK_D # Match dot product kernel or tune

    grid_assign = lambda meta: (triton.cdiv(N_A, meta['BLOCK_SIZE_N']),)

    for i in range(max_iters):
        old_centroids = centroids.clone() # Need clone, not just assign
        # --- 1. Assignment Step (Triton Kernel) ---
        # Kernel requires int32 pointer for assignments, but scatter_add_ needs int64 index
        assignments_int32 = torch.empty(N_A, dtype=torch.int32, device=device)
        kmeans_assign_kernel[grid_assign](
            A_prep, centroids, assignments_int32, # Pass float32 data/centroids
            N_A, D, K,
            A_prep.stride(0), 1, # Assuming contiguous stride 1 for D
            centroids.stride(0), 1, # Assuming contiguous stride 1 for D
            BLOCK_SIZE_N=BLOCK_SIZE_N_ASSIGN,
            BLOCK_SIZE_K_CHUNK=BLOCK_SIZE_K_CHUNK_ASSIGN,
            BLOCK_SIZE_D=BLOCK_SIZE_D_ASSIGN
        )
        assignments = assignments_int32.to(torch.int64) # Convert to int64 for scatter_add_
        torch.cuda.synchronize(device=device) # Sync after kernel launch

        # --- 2. Update Step (PyTorch scatter_add_) ---
        new_sums = torch.zeros_like(centroids)
        cluster_counts = torch.zeros(K, dtype=torch.float32, device=device)
        # Prepare index tensor for scatter_add_ on sums (needs to match src shape)
        idx_expand = assignments.unsqueeze(1).expand(-1, D) # Shape (N_A, D)

        # Perform scatter add operations
        new_sums.scatter_add_(dim=0, index=idx_expand, src=A_prep)
        cluster_counts.scatter_add_(dim=0, index=assignments, src=torch.ones_like(assignments, dtype=torch.float32))
        torch.cuda.synchronize(device=device) # Sync after scatter adds

        # --- Update Centroids ---
        final_counts_safe = cluster_counts.clamp(min=1.0) # Avoid division by zero
        new_centroids = new_sums / final_counts_safe.unsqueeze(1)
        empty_cluster_mask = (cluster_counts == 0)
        # If a cluster is empty, keep its previous centroid position
        new_centroids[empty_cluster_mask] = old_centroids[empty_cluster_mask]
        # Check for NaN/inf (safety)
        if not torch.all(torch.isfinite(new_centroids)):
             print(f"Warning: Non-finite values found in centroids at iteration {i+1}.")
             new_centroids = torch.nan_to_num(new_centroids, nan=old_centroids[torch.isnan(new_centroids)])

        # --- Check Convergence ---
        centroid_diff = torch.linalg.norm(new_centroids - centroids) # Use old_centroids from start of loop
        centroids = new_centroids # Update centroids for next iteration

        # Optional: Print progress
        # print(f"  Iter {i+1}/{max_iters} | Centroid Diff: {centroid_diff:.4f}")

        if centroid_diff < tol:
            # print(f"Converged after {i+1} iterations.")
            break

    # if i == max_iters - 1: print(f"Reached max iterations ({max_iters}).")
    # print(f"Total K-Means time: {time.time() - start_time_total:.4f}s")

    return centroids, assignments # Return assignments as int64


# --- KNN Function (using distance_l2) ---
# (User provided this - assumed okay, but using torch.topk might be simpler)
def our_knn(N_A, D, A, X, K):
    A_prep, X_prep = _prepare_tensors(A, X)
    Q = X_prep.shape[0]
    # ... (assertions as before) ...
    if not (K > 0 and K <= N_A): K = min(max(1, K), N_A) # Adjust K

    # print(f"Running k-NN: Q={Q}, N={N_A}, D={D}, K={K}")
    start_time = time.time()
    # distance_l2 now returns SQUARED L2 distances
    all_distances_sq = distance_l2(X_prep, A_prep, prep = False) # Shape (Q, N_A)
    # Find the top K smallest squared distances
    topk_distances_sq, topk_indices = torch.topk(all_distances_sq, k=K, dim=1, largest=False, sorted=True)
    # print(f"k-NN computation time: {time.time() - start_time:.6f} seconds")
    # Return SQUARED distances for consistency with internal ANN calculations
    return topk_indices, topk_distances_sq

# --- Normalization Helper ---
def normalize_vectors(vectors, epsilon=1e-12):
    """L2 normalize PyTorch vectors row-wise."""
    norms = torch.linalg.norm(vectors, dim=1, keepdim=True)
    return vectors / (norms + epsilon)


# ============================================================================
# NEW: ANN Function (User Pseudocode, L2, Using Triton/PyTorch)
# ============================================================================
def our_ann_user_pseudocode_impl_l2_triton(N_A, D, A, X, k_clusters, K1, K2,
                                           max_kmeans_iters=100, centroids=None):
    """
    Implements the user's specific 4-step pseudocode using L2 DISTANCE.
    Uses Triton distance kernel + PyTorch operations.
    Can optionally accept pre-computed centroids.
    Note: Finds nearest K2 CLUSTER CENTERS based on L2 distance.

    Args:
        N_A (int): Number of database points (for KMeans if centroids is None).
        D (int): Dimensionality.
        A (torch.Tensor): Database vectors (N_A, D) on GPU (for KMeans if centroids is None).
        X (torch.Tensor): Query vectors (Q, D) on GPU.
        k_clusters (int): Target number of clusters for KMeans (if centroids is None).
        K1 (int): Number of nearest cluster centers to initially identify (Step 2).
        K2 (int): Number of nearest cluster centers to finally return from the K1 set (Step 3/4).
        max_kmeans_iters (int): Max iterations for K-Means.
        centroids (torch.Tensor, optional): Pre-computed centroids (k, D). If provided, KMeans is skipped.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
            - topk2_centroid_indices (torch.Tensor): Indices of the final K2 nearest *centroids* (Q, K2).
            - topk2_centroid_distances_sq (torch.Tensor): **SQUARED** L2 distances to these K2 *centroids* (Q, K2).
            - centroids_used (torch.Tensor): The actual centroids used.
            - build_time (float): Time taken for KMeans (0 if centroids was provided).
            - search_time (float): Time taken for the search logic.
    """
    # --- Input Validation ---
    X_prep, = _prepare_tensors(X) # Prepare queries
    Q = X_prep.shape[0]
    if X_prep.shape[1] != D: raise ValueError(f"Query dimension mismatch: X D={X_prep.shape[1]}, expected D={D}")

    build_time = 0.0
    search_time = 0.0
    actual_k_clusters = 0
    centroids_used = None

    # --- Step 1: Use KMeans or provided centroids ---
    if centroids is None:
        # print("No precomputed centroids provided. Running Triton/PyTorch KMeans...")
        build_start_time = time.time()
        A_prep, = _prepare_tensors(A) # Prepare data only if building
        if A_prep.shape[1] != D: raise ValueError("A data dimension mismatch.")

        # Call the Triton/PyTorch KMeans function
        centroids_used, _ = our_kmeans(N_A, D, A_prep, k_clusters, max_iters=max_kmeans_iters)
        torch.cuda.synchronize(device=device)
        build_time = time.time() - build_start_time
        # print(f"Build time (Triton/PyTorch KMeans): {build_time:.4f}s")
    else:
        # print("Using precomputed centroids.")
        # Prepare provided centroids
        centroids_used, = _prepare_tensors(centroids)
        if centroids_used.ndim != 2 or centroids_used.shape[1] != D:
             raise ValueError(f"Provided centroids invalid shape/dim {centroids_used.shape}.")

    actual_k_clusters = centroids_used.shape[0]
    if actual_k_clusters == 0:
        print("Error: No centroids available. Cannot proceed.")
        empty_indices = torch.full((Q, K2 if K2 > 0 else 1), -1, dtype=torch.int64, device=device)
        empty_dists = torch.full((Q, K2 if K2 > 0 else 1), float('inf'), dtype=torch.float32, device=device)
        return empty_indices, empty_dists, torch.empty((0,D), dtype=torch.float32, device=device), build_time, 0.0

    # print(f"Using {actual_k_clusters} centroids for search.")
    K1 = min(K1, actual_k_clusters); K2 = min(K2, K1)
    if K1 <= 0 or K2 <= 0:
        print(f"Error: K1 ({K1}) or K2 ({K2}) is non-positive. Cannot proceed.")
        empty_indices = torch.full((Q, K2 if K2 > 0 else 1), -1, dtype=torch.int64, device=device)
        empty_dists = torch.full((Q, K2 if K2 > 0 else 1), float('inf'), dtype=torch.float32, device=device)
        return empty_indices, empty_dists, centroids_used, build_time, 0.0
    # print(f"Adjusted params: K1={K1}, K2={K2}")

    # --- Search Phase (Using SQUARED L2 Distance with Triton kernel) ---
    search_start_time = time.time()

    # Calculate all query-centroid SQUARED L2 distances using Triton dot kernel internally
    # Pass prep=False as X_prep and centroids_used are already prepared
    all_query_centroid_dists_sq = distance_l2(X_prep, centroids_used, prep=False) # Shape (Q, actual_k_clusters)

    # Step 2: Find K1 nearest centroids (based on squared L2 distance)
    # Use torch.topk which is efficient on GPU and returns sorted results
    # We need smallest distances, so largest=False
    topk1_dists_sq, topk1_indices = torch.topk(all_query_centroid_dists_sq, k=K1, dim=1, largest=False, sorted=True)
    # topk1_indices are original centroid indices (0 to actual_k_clusters-1)

    # Step 3: Find K2 nearest among K1 (based on squared L2 distance)
    # Since topk1 results are already sorted by distance, the first K2 are the ones we need
    topk2_subset_dists_sq = topk1_dists_sq[:, :K2] # Shape (Q, K2)
    # The indices corresponding to these are simply the first K2 indices from topk1_indices
    topk2_centroid_indices = topk1_indices[:, :K2] # Shape (Q, K2)

    # Step 4: Sort final K2 results - Already sorted by torch.topk

    final_topk2_centroid_indices = topk2_centroid_indices
    final_topk2_centroid_distances_sq = topk2_subset_dists_sq # SQUARED Distances

    torch.cuda.synchronize(device=device) # Ensure all GPU operations complete
    search_time = time.time() - search_start_time
    # print(f"Search time (User Pseudocode, L2 Triton): {search_time:.4f}s")

    # Return centroid indices (int64), SQUARED L2 distances (float32), centroids used, build_time, search_time
    return final_topk2_centroid_indices.to(torch.int64), \
           final_topk2_centroid_distances_sq, \
           centroids_used, \
           build_time, \
           search_time

# ============================================================================
# Main Execution Block (Triton/PyTorch ONLY, Looping Through Dimensions)
# ============================================================================
if __name__ == "__main__":
    # --- Fixed Parameters for all dimension runs ---
    N_data = 100000
    N_queries_new = 1000 # Number of new queries
    num_clusters_for_kmeans = 1000 # Target cluster count
    K1_probe = 30
    K2_final = 10
    kmeans_max_iters = 50

    # --- Dimensions to Test ---
    dimensions_to_test = [16,2, 2, 4, 64, 256, 1024]

    print(f"--- Triton/PyTorch EXECUTION LOOPING THROUGH DIMENSIONS ---")
    print(f"Fixed Params: N={N_data}, Q={N_queries_new}, k_clusters={num_clusters_for_kmeans}, K1={K1_probe}, K2={K2_final}")
    print(f"Testing Dimensions: {dimensions_to_test}")

    # Loop through each dimension
    for Dim in dimensions_to_test:
        print("\n" + "#"*70)
        print(f"# Starting Test for Dimension D = {Dim}")
        print("#"*70)

        # --- Generate Base Data (on GPU) for the current dimension ---
        print("\n" + "="*40); print(f"Generating Base Data (D={Dim}, GPU)..."); print("="*40)
        try:
            # Use PyTorch directly on the target device
            A_data = torch.randn(N_data, Dim, dtype=torch.float32, device=device)
            torch.cuda.synchronize(device=device)
            print(f"Database shape (GPU): {A_data.shape}")
            mem_gb = A_data.nelement() * A_data.element_size() / (1024**3)
            print(f"Approx. memory for A_data: {mem_gb:.2f} GB")

        except RuntimeError as e: # Catch CUDA OOM errors etc.
             if "CUDA out of memory" in str(e):
                 print(f"Error: Failed to allocate GPU memory for A_data (D={Dim}). Skipping.")
             else:
                 print(f"Error generating A_data (D={Dim}): {e}")
             # Clean up potentially allocated array before continuing
             if 'A_data' in locals(): del A_data
             torch.cuda.empty_cache()
             continue # Skip to the next dimension
        except Exception as e:
            print(f"Error generating A_data (D={Dim}): {e}");
            continue

        # --- Build Index ONCE (GPU KMeans) for the current dimension ---
        print("\n" + "="*40); print(f"Building Centroids via Triton/PyTorch KMeans (D={Dim})..."); print("="*40)
        initial_centroids = None
        actual_k_clusters = 0
        build_time_actual = 0
        try:
            build_start_actual = time.time()
            # Call the Triton/PyTorch KMeans function
            initial_centroids, _ = our_kmeans(N_data, Dim, A_data, num_clusters_for_kmeans, max_iters=kmeans_max_iters)
            actual_k_clusters = initial_centroids.shape[0]
            build_time_actual = time.time() - build_start_actual
            if actual_k_clusters == 0: raise ValueError("KMeans returned 0 centroids.")
            print(f"Triton/PyTorch KMeans completed. Found {actual_k_clusters} centroids.")
            print(f"Actual Build Time (D={Dim}): {build_time_actual:.4f}s")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e): print(f"OOM Error during KMeans build (D={Dim}): {e}")
            else: print(f"Runtime Error during KMeans build (D={Dim}): {e}")
            print("Consider reducing N_data or num_clusters_for_kmeans.")
            if 'initial_centroids' in locals(): del initial_centroids
            del A_data; torch.cuda.empty_cache(); continue
        except Exception as e:
            print(f"Error during initial KMeans build (D={Dim}): {e}"); traceback.print_exc();
            if 'initial_centroids' in locals(): del initial_centroids
            del A_data; torch.cuda.empty_cache(); continue

        # --- Generate NEW Queries (on GPU) for the current dimension ---
        print("\n" + "="*40); print(f"Generating {N_queries_new} NEW Test Queries (D={Dim}, GPU)..."); print("="*40)
        try:
            X_queries_new = torch.randn(N_queries_new, Dim, dtype=torch.float32, device=device)
            torch.cuda.synchronize(device=device)
            print(f"New query shape (GPU): {X_queries_new.shape}")
        except RuntimeError as e:
             if "CUDA out of memory" in str(e): print(f"Error: OOM allocating {N_queries_new} queries (D={Dim}). Skipping.")
             else: print(f"Runtime Error generating queries (D={Dim}): {e}")
             del initial_centroids, A_data; torch.cuda.empty_cache(); continue
        except Exception as e:
            print(f"Error generating new queries (D={Dim}): {e}");
            del initial_centroids, A_data; torch.cuda.empty_cache(); continue

        # --- Run ANN Search (Triton/PyTorch Function) for the current dimension ---
        print("\n" + "="*40); print(f"Testing Triton/PyTorch ANN Search (D={Dim})..."); print("="*40)
        k1_run = min(K1_probe, actual_k_clusters)
        k2_run = min(K2_final, k1_run)
        print(f"(Using {actual_k_clusters} centroids, K1={k1_run}, K2={k2_run}, L2 Distance, Triton Search)")
        print("="*40)
        ann_indices_centroids = None
        ann_dists_sq_centroids = None
        centroids_used = None
        search_t = 0
        try:
            # Call the Triton/PyTorch L2 version of the function
            ann_indices_centroids, ann_dists_sq_centroids, centroids_used, build_t_ignored, search_t = our_ann_user_pseudocode_impl_l2_triton(
                N_A=N_data, D=Dim, A=A_data,             # Pass A (only needed if building inside)
                X=X_queries_new,                       # Use GPU queries
                k_clusters=actual_k_clusters,          # Pass actual count
                K1=k1_run,                             # Use adjusted K1
                K2=k2_run,                             # Use adjusted K2
                centroids=initial_centroids            # Pass precomputed GPU centroids
            )
            print("Triton ANN results shape (Centroid Indices):", ann_indices_centroids.shape)
            print("Triton ANN results shape (**SQUARED** L2 Distances to Centroids):", ann_dists_sq_centroids.shape)
            print(f"Triton ANN Search Time (D={Dim}): {search_t:.4f}s")
            if search_t > 0: print(f"-> Throughput: {N_queries_new / search_t:.2f} queries/sec (GPU)")

        except RuntimeError as e:
            if "CUDA out of memory" in str(e): print(f"OOM Error during ANN execution (D={Dim}): {e}")
            else: print(f"Runtime Error during ANN execution (D={Dim}): {e}")
            traceback.print_exc(); ann_indices_centroids = None
        except Exception as e:
            print(f"Error during ANN execution (D={Dim}): {e}"); traceback.print_exc(); ann_indices_centroids = None


        # --- Optional: Calculate Recall (GPU) for the current dimension ---
        if ann_indices_centroids is not None and centroids_used is not None and centroids_used.shape[0] > 0:
            print("\n" + "="*40); print(f"Calculating Recall (D={Dim}, Triton/PyTorch L2)..."); print("="*40)
            K_recall = k2_run
            try:
                # Ground truth calculation on GPU using Triton/PyTorch distance
                # print("Calculating ground truth (Triton/PyTorch brute-force nearest centroids using SQUARED L2)...")
                start_gt = time.time()
                # Use the same queries/centroids as the search function used
                all_query_centroid_dists_sq_gt = distance_l2(X_queries_new, centroids_used, prep=False) # Use the L2 function

                actual_num_centroids = centroids_used.shape[0]
                k_recall_adjusted = min(K_recall, actual_num_centroids)

                if k_recall_adjusted > 0:
                    # Find true nearest centroids using SQUARED L2 distance on GPU
                    # Use torch.topk (returns indices in [1])
                    true_knn_centroid_indices = torch.topk(all_query_centroid_dists_sq_gt, k=k_recall_adjusted, dim=1, largest=False, sorted=True)[1]
                else:
                    true_knn_centroid_indices = torch.empty((N_queries_new, 0), dtype=torch.int64, device=device)
                torch.cuda.synchronize(device=device) # Wait for GPU topk
                # print(f"Ground truth calculation time: {time.time() - start_gt:.4f}s")

                # Compare results (transfer minimal data to CPU)
                total_intersect_centroids = 0
                # Transfer only the necessary slices for comparison
                ann_indices_np = ann_indices_centroids[:, :k_recall_adjusted].cpu().numpy()
                true_indices_np = true_knn_centroid_indices.cpu().numpy() # Ground truth indices were already K2

                # Comparison loop on CPU
                for i in range(N_queries_new):
                     approx_centroid_ids = set(idx for idx in ann_indices_np[i] if idx >= 0) # Filter potential -1 if any
                     true_centroid_ids = set(true_indices_np[i])
                     total_intersect_centroids += len(approx_centroid_ids.intersection(true_centroid_ids))

                if N_queries_new > 0 and k_recall_adjusted > 0:
                    avg_recall_centroids = total_intersect_centroids / (N_queries_new * k_recall_adjusted)
                    print(f"\nAverage Recall @ {k_recall_adjusted} (vs Triton/Torch brute-force CENTROIDS w/ L2, D={Dim}): {avg_recall_centroids:.4f} ({avg_recall_centroids:.2%})")
                    # Use corrected commentary logic
                    epsilon = 1e-9
                    if abs(avg_recall_centroids - 1.0) < epsilon: print("Result: 100% recall indicates K1 was large enough...")
                    else: print(f"Result: Recall ({avg_recall_centroids:.4f}) < 100%...")
                else: print("\nCannot calculate recall (N_queries=0 or K2=0).")

            except RuntimeError as e:
                 if "CUDA out of memory" in str(e): print(f"OOM Error during recall calculation (D={Dim}): {e}")
                 else: print(f"Runtime Error during recall calculation (D={Dim}): {e}")
                 traceback.print_exc()
            except Exception as e: print(f"Error during recall calculation (D={Dim}): {e}"); traceback.print_exc()
        else:
             print("\nSkipping recall calculation as ANN results or centroids are unavailable.")

        print(f"\n--- Finished Test for Dimension D = {Dim} ---")
        # Clean up GPU memory before next potentially larger dimension run
        del A_data, X_queries_new, initial_centroids, centroids_used
        if 'ann_indices_centroids' in locals(): del ann_indices_centroids
        if 'ann_dists_sq_centroids' in locals(): del ann_dists_sq_centroids
        if 'all_query_centroid_dists_sq_gt' in locals(): del all_query_centroid_dists_sq_gt
        if 'true_knn_centroid_indices' in locals(): del true_knn_centroid_indices
        torch.cuda.empty_cache() # Clear GPU memory cache


    print("\n--- ALL Triton/PyTorch DIMENSION TESTS FINISHED ---")