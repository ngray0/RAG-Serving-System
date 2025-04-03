import torch
import triton
import triton.language as tl
import math
import heapq # For HNSW priority queues
import random
import time

# --- Ensure GPU is available ---
if not torch.cuda.is_available():
    print("CUDA not available, exiting.")
    exit()
device = torch.device("cuda:0")
print(f"Using device: {device}")

# ============================================================================
# Triton Distance Kernels (Pairwise: Q queries vs N database points)
# ============================================================================

@triton.jit
def l2_dist_kernel_pairwise(
    X_ptr,      # Pointer to Query vectors (Q, D)
    A_ptr,      # Pointer to Database vectors (N, D)
    Out_ptr,    # Pointer to output distances (Q, N)
    # --- Dimensions ---
    Q, N, D,
    # --- Strides ---
    stride_xq, stride_xd,
    stride_an, stride_ad,
    stride_outq, stride_outn,
    # --- Block Size ---
    BLOCK_SIZE_D: tl.constexpr,
):
    """Calculates pairwise squared L2 distance: dist(X[q], A[n])"""
    pid_q = tl.program_id(axis=0) # Query index
    pid_n = tl.program_id(axis=1) # Database index

    dist_sq = tl.zeros((), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_SIZE_D):
        d_end = tl.minimum(d_start + BLOCK_SIZE_D, D)
        offs_d = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offs_d < d_end

        # Load X[pid_q, d_start:d_end]
        x_ptrs = X_ptr + pid_q * stride_xq + offs_d * stride_xd
        x_vals = tl.load(x_ptrs, mask=mask_d, other=0.0)

        # Load A[pid_n, d_start:d_end]
        a_ptrs = A_ptr + pid_n * stride_an + offs_d * stride_ad
        a_vals = tl.load(a_ptrs, mask=mask_d, other=0.0)

        diff = x_vals - a_vals
        dist_sq += tl.sum(diff * diff, axis=0)

    # Store result
    out_offset = pid_q * stride_outq + pid_n * stride_outn
    tl.store(Out_ptr + out_offset, dist_sq)

@triton.jit
def dot_kernel_pairwise(
    X_ptr, A_ptr, Out_ptr,
    Q, N, D,
    stride_xq, stride_xd,
    stride_an, stride_ad,
    stride_outq, stride_outn,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Calculates pairwise dot product: dot(X[q], A[n])"""
    pid_q = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    dot_prod = tl.zeros((), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_SIZE_D):
        d_end = tl.minimum(d_start + BLOCK_SIZE_D, D)
        offs_d = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offs_d < d_end

        x_ptrs = X_ptr + pid_q * stride_xq + offs_d * stride_xd
        x_vals = tl.load(x_ptrs, mask=mask_d, other=0.0)

        a_ptrs = A_ptr + pid_n * stride_an + offs_d * stride_ad
        a_vals = tl.load(a_ptrs, mask=mask_d, other=0.0)

        dot_prod += tl.sum(x_vals * a_vals, axis=0)

    out_offset = pid_q * stride_outq + pid_n * stride_outn
    tl.store(Out_ptr + out_offset, dot_prod)

@triton.jit
def norm_kernel(
    V_ptr, Norms_ptr, # Input vectors (Count, D), Output norms (Count,)
    Count, D,
    stride_vn, stride_vd,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Calculates L2 norm for each vector in V."""
    pid_n = tl.program_id(axis=0) # Vector index

    norm_sq = tl.zeros((), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_SIZE_D):
        d_end = tl.minimum(d_start + BLOCK_SIZE_D, D)
        offs_d = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offs_d < d_end

        v_ptrs = V_ptr + pid_n * stride_vn + offs_d * stride_vd
        v_vals = tl.load(v_ptrs, mask=mask_d, other=0.0)

        norm_sq += tl.sum(v_vals * v_vals, axis=0)

    norm = tl.sqrt(norm_sq)
    tl.store(Norms_ptr + pid_n, norm)


@triton.jit
def manhattan_dist_kernel_pairwise(
    X_ptr, A_ptr, Out_ptr,
    Q, N, D,
    stride_xq, stride_xd,
    stride_an, stride_ad,
    stride_outq, stride_outn,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Calculates pairwise Manhattan (L1) distance: sum(abs(X[q] - A[n]))"""
    pid_q = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    dist_l1 = tl.zeros((), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_SIZE_D):
        d_end = tl.minimum(d_start + BLOCK_SIZE_D, D)
        offs_d = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offs_d < d_end

        x_ptrs = X_ptr + pid_q * stride_xq + offs_d * stride_xd
        x_vals = tl.load(x_ptrs, mask=mask_d, other=0.0)

        a_ptrs = A_ptr + pid_n * stride_an + offs_d * stride_ad
        a_vals = tl.load(a_ptrs, mask=mask_d, other=0.0)

        diff = x_vals - a_vals
        dist_l1 += tl.sum(tl.abs(diff), axis=0)

    out_offset = pid_q * stride_outq + pid_n * stride_outn
    tl.store(Out_ptr + out_offset, dist_l1)


# ============================================================================
# Python Distance Function Wrappers
# ============================================================================
# Default block size for dimension splitting in kernels (can be tuned)
DEFAULT_BLOCK_D = 128

def _prepare_tensors(*tensors):
    """Ensure tensors are float32, contiguous, and on the correct device."""
    prepared = []
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=device)
        if t.device != device:
            t = t.to(device)
        if t.dtype != torch.float32:
            t = t.to(dtype=torch.float32)
        # Kernels often benefit from contiguous memory
        prepared.append(t.contiguous())
    return prepared

def distance_l2(X, A):
    """Computes pairwise squared L2 distance using Triton kernel."""
    X_prep, A_prep = _prepare_tensors(X, A)
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    assert D == D_A, f"Dimension mismatch: X({D}) vs A({D_A})"

    Out = torch.empty((Q, N), dtype=torch.float32, device=device)
    grid = (Q, N)
    l2_dist_kernel_pairwise[grid](
        X_prep, A_prep, Out,
        Q, N, D,
        X_prep.stride(0), X_prep.stride(1),
        A_prep.stride(0), A_prep.stride(1),
        Out.stride(0), Out.stride(1),
        BLOCK_SIZE_D=DEFAULT_BLOCK_D
    )
    return Out

def distance_dot(X, A):
    """Computes pairwise dot product using Triton kernel."""
    X_prep, A_prep = _prepare_tensors(X, A)
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    assert D == D_A, f"Dimension mismatch: X({D}) vs A({D_A})"

    Out = torch.empty((Q, N), dtype=torch.float32, device=device)
    grid = (Q, N)
    dot_kernel_pairwise[grid](
        X_prep, A_prep, Out,
        Q, N, D,
        X_prep.stride(0), X_prep.stride(1),
        A_prep.stride(0), A_prep.stride(1),
        Out.stride(0), Out.stride(1),
        BLOCK_SIZE_D=DEFAULT_BLOCK_D
    )
    # Return negative dot product if used for minimization (finding 'nearest')
    # return -Out
    # Or return raw dot product if similarity maximization is intended
    return Out


def distance_cosine(X, A):
    """Computes pairwise cosine distance using Triton kernels."""
    X_prep, A_prep = _prepare_tensors(X, A)
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    assert D == D_A, f"Dimension mismatch: X({D}) vs A({D_A})"

    # Calculate norms
    X_norms = torch.empty(Q, dtype=torch.float32, device=device)
    A_norms = torch.empty(N, dtype=torch.float32, device=device)

    grid_q = (Q,)
    norm_kernel[grid_q](X_prep, X_norms, Q, D, X_prep.stride(0), X_prep.stride(1), BLOCK_SIZE_D=DEFAULT_BLOCK_D)
    grid_n = (N,)
    norm_kernel[grid_n](A_prep, A_norms, N, D, A_prep.stride(0), A_prep.stride(1), BLOCK_SIZE_D=DEFAULT_BLOCK_D)

    # Calculate dot products
    dot_products = distance_dot(X_prep, A_prep) # Uses the dot kernel internally

    # Calculate cosine similarity
    # Add epsilon to avoid division by zero for zero vectors
    epsilon = 1e-8
    norm_product = X_norms[:, None] * A_norms[None, :]
    similarities = dot_products / norm_product.clamp(min=epsilon)
    # Clamp similarities to [-1, 1] due to potential floating point inaccuracies
    similarities.clamp_(min=-1.0, max=1.0)

    # Cosine distance = 1 - similarity
    distances = 1.0 - similarities
    return distances

def distance_manhattan(X, A):
    """Computes pairwise Manhattan (L1) distance using Triton kernel."""
    X_prep, A_prep = _prepare_tensors(X, A)
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    assert D == D_A, f"Dimension mismatch: X({D}) vs A({D_A})"

    Out = torch.empty((Q, N), dtype=torch.float32, device=device)
    grid = (Q, N)
    manhattan_dist_kernel_pairwise[grid](
        X_prep, A_prep, Out,
        Q, N, D,
        X_prep.stride(0), X_prep.stride(1),
        A_prep.stride(0), A_prep.stride(1),
        Out.stride(0), Out.stride(1),
        BLOCK_SIZE_D=DEFAULT_BLOCK_D
    )
    return Out

# ============================================================================
# Task 1.2: k-Nearest Neighbors (Brute Force)
# ============================================================================

def our_knn(N_A, D, A, X, K):
    """
    Finds the K nearest neighbors in A for each query vector in X using
    brute-force pairwise L2 distance calculation.

    Args:
        N_A (int): Number of database points (should match A.shape[0]).
        D (int): Dimensionality (should match A.shape[1] and X.shape[1]).
        A (torch.Tensor): Database vectors (N_A, D) on GPU.
        X (torch.Tensor): Query vectors (Q, D) on GPU.
        K (int): Number of neighbors to find.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - topk_indices (torch.Tensor): Indices of the K nearest neighbors (Q, K).
            - topk_distances (torch.Tensor): Squared L2 distances of the K nearest neighbors (Q, K).
    """
    A_prep, X_prep = _prepare_tensors(A, X)
    Q = X_prep.shape[0]
    assert A_prep.shape[0] == N_A, "N_A doesn't match A.shape[0]"
    assert A_prep.shape[1] == D, "D doesn't match A.shape[1]"
    assert X_prep.shape[1] == D, "D doesn't match X.shape[1]"
    assert K > 0, "K must be positive"
    assert K <= N_A, "K cannot be larger than the number of database points"


    print(f"Running k-NN: Q={Q}, N={N_A}, D={D}, K={K}")
    start_time = time.time()

    # 1. Calculate all pairwise squared L2 distances
    #    distance_l2 returns squared L2 distances
    all_distances = distance_l2(X_prep, A_prep) # Shape (Q, N_A)

    # 2. Find the top K smallest distances for each query
    #    largest=False gives smallest distances (nearest neighbors)
    topk_distances, topk_indices = torch.topk(all_distances, k=K, dim=1, largest=False)

    end_time = time.time()
    print(f"k-NN computation time: {end_time - start_time:.4f} seconds")

    return topk_indices, topk_distances


# ============================================================================
# Task 2.1: K-Means Clustering
# ============================================================================

# --- Triton Kernels specific to K-Means (adapted from previous examples) ---
# --- Uses 1-vs-M distance logic internally for assignment ---
@triton.jit
def kmeans_assign_kernel(
    A_ptr,           # Pointer to data points (N, D)
    centroids_ptr,   # Pointer to centroids (K, D)
    assignments_ptr, # Pointer to output assignments (N,)
    # --- Dimensions ---
    N, D, K,
    # --- Strides ---
    stride_an, stride_ad,
    stride_ck, stride_cd,
    # --- Block Sizes ---
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K_CHUNK: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr, # ADDED BLOCK_SIZE_D for dimension iteration
):
    """Assigns each point in A to the nearest centroid (Squared L2) by iterating dimensions."""
    pid_n_block = tl.program_id(axis=0) # ID for the N dimension block
    offs_n = pid_n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) # Range for points in block
    mask_n = offs_n < N # Mask for points in this block

    # Initialize minimum distance and best assignment for points in this block
    min_dist_sq = tl.full((BLOCK_SIZE_N,), float('inf'), dtype=tl.float32)
    best_assignment = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)

    # Iterate through centroids in chunks for better cache (?) locality
    for k_start in range(0, K, BLOCK_SIZE_K_CHUNK):
        k_end = tl.minimum(k_start + BLOCK_SIZE_K_CHUNK, K)
        # Note: Using tl.arange here for k_idx loop is fine as BLOCK_SIZE_K_CHUNK is constexpr
        # but a simple python range is often clearer for the outer loops.

        # Iterate through each centroid within the current chunk
        for k_idx in range(BLOCK_SIZE_K_CHUNK):
            actual_k = k_start + k_idx
            if actual_k < k_end: # Check if this centroid index is valid for the chunk/K
                # Accumulate distance for this centroid (actual_k) across all dimension blocks
                current_dist_sq = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

                # Iterate over dimensions in blocks using BLOCK_SIZE_D
                for d_start in range(0, D, BLOCK_SIZE_D):
                    # Create offsets for the current dimension block
                    offs_d = d_start + tl.arange(0, BLOCK_SIZE_D) # Uses constexpr BLOCK_SIZE_D
                    mask_d = offs_d < D # Mask for valid dimensions in this block

                    # Load centroid block for dimension d
                    # Pointer: base + centroid_row * stride_k + dim_offset * stride_d
                    centroid_d_ptr = centroids_ptr + actual_k * stride_ck + offs_d * stride_cd
                    # Load safely using mask; shape (BLOCK_SIZE_D,)
                    centroid_vals = tl.load(centroid_d_ptr, mask=mask_d, other=0.0)

                    # Load points block for dimension d
                    # Base pointer for point row `n`: A_ptr + n * stride_an
                    # Pointer for point `n`, dim `d`: base + d * stride_ad
                    # Need shape (BLOCK_SIZE_N, BLOCK_SIZE_D)
                    # Offset for points: offs_n[:, None] * stride_an
                    # Offset for dims: offs_d[None, :] * stride_ad
                    points_d_ptr = A_ptr + offs_n[:, None] * stride_an + offs_d[None, :] * stride_ad
                    # Load safely using masks; shape (BLOCK_SIZE_N, BLOCK_SIZE_D)
                    points_vals = tl.load(points_d_ptr, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

                    # Calculate difference squared for this dimension block
                    # Broadcasting happens: (BLOCK_N, BLOCK_D) - (BLOCK_D,) -> (BLOCK_N, BLOCK_D)
                    diff = points_vals - centroid_vals[None, :]
                    # Sum squared diffs over the dimension block axis=1 -> (BLOCK_N,)
                    current_dist_sq += tl.sum(diff * diff, axis=1)

                # --- Update Minimum Distance and Assignment ---
                # Compare accumulated distance for centroid 'actual_k' with the current minimum
                is_closer = current_dist_sq < min_dist_sq
                min_dist_sq = tl.where(is_closer, current_dist_sq, min_dist_sq)
                best_assignment = tl.where(is_closer, actual_k, best_assignment)

    # --- Store Results ---
    # Store the best assignment found for each point in this block
    assignments_out_ptrs = assignments_ptr + offs_n
    tl.store(assignments_out_ptrs, best_assignment, mask=mask_n)
"""@triton.jit
def kmeans_update_kernel_part1(
    A_ptr,              # Pointer to data points (N, D)
    assignments_ptr,    # Pointer to assignments (N,)
    partial_sums_ptr,   # Pointer to output partial sums (num_blocks_n, K, D)
    partial_counts_ptr, # Pointer to output partial counts (num_blocks_n, K)
    # --- Dimensions ---
    N, D, K,
    # --- Strides ---
    stride_an, stride_ad,
    stride_ps_block, stride_ps_k, stride_ps_d,
    stride_pc_block, stride_pc_k,
    # --- Block Sizes ---
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
   
    pid_n_block = tl.program_id(axis=0) # Block ID for points
    offs_n = pid_n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N

    assignments_ptrs = assignments_ptr + offs_n
    assignments = tl.load(assignments_ptrs, mask=mask_n, other=-1)
    valid_assignment_mask = mask_n & (assignments >= 0)

    # Iterate over dimensions in blocks for sums
    for d_start in range(0, D, BLOCK_SIZE_D):
        offs_d = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offs_d < D

        # Load data block A
        a_ptrs = A_ptr + offs_n[:, None] * stride_an + offs_d[None, :] * stride_ad
        a_vals = tl.load(a_ptrs, mask=valid_assignment_mask[:, None] & mask_d[None, :], other=0.0)

        # Calculate target pointers for partial sums
        partial_sums_target_ptrs = (partial_sums_ptr +
                                    pid_n_block * stride_ps_block +
                                    assignments[:, None] * stride_ps_k +
                                    offs_d[None, :] * stride_ps_d)

        # Perform atomic add for the sums block (without mem_odr)
        tl.atomic_add(partial_sums_target_ptrs, a_vals,
                      mask=valid_assignment_mask[:, None] & mask_d[None, :]) # Removed mem_odr

    # Atomic Add for Partial Counts (only needs to be done once per point)
    partial_counts_target_ptrs = (partial_counts_ptr +
                                  pid_n_block * stride_pc_block +
                                  assignments * stride_pc_k)

    # Add 1.0 for each valid point/assignment (without mem_odr)
    tl.atomic_add(partial_counts_target_ptrs, 1.0, mask=valid_assignment_mask) # Removed mem_odr
    """
def our_kmeans(N_A, D, A, K, max_iters=100, tol=1e-4):
    """
    Performs K-means clustering on data A using Triton kernel for assignment
    and PyTorch scatter_add_ for the update step.

    Args:
        N_A (int): Number of data points.
        D (int): Dimensionality.
        A (torch.Tensor): Data points (N_A, D) on GPU.
        K (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance for centroid movement convergence check.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - centroids (torch.Tensor): Final centroids (K, D).
            - assignments (torch.Tensor): Final cluster assignment (N_A,).
    """
    A_prep, = _prepare_tensors(A)
    assert A_prep.shape[0] == N_A, "N_A doesn't match A.shape[0]"
    assert A_prep.shape[1] == D, "D doesn't match A.shape[1]"
    assert K > 0, "K must be positive"
    assert K <= N_A, "K cannot be larger than the number of data points"

    print(f"Running K-Means (Update with PyTorch): N={N_A}, D={D}, K={K}")
    start_time_total = time.time()

    # --- Initialization ---
    initial_indices = torch.randperm(N_A, device=device)[:K]
    centroids = A_prep[initial_indices].clone()
    assignments = torch.empty(N_A, dtype=torch.int64, device=device) # Use int64 for scatter_add_ index

    # --- Triton Kernel Launch Configuration (Only for Assignment) ---
    BLOCK_SIZE_N_ASSIGN = 128
    BLOCK_SIZE_K_CHUNK_ASSIGN = 64
    BLOCK_SIZE_D_ASSIGN = DEFAULT_BLOCK_D

    grid_assign = lambda meta: (triton.cdiv(N_A, meta['BLOCK_SIZE_N']),)

    for i in range(max_iters):
        iter_start_time = time.time()

        # --- 1. Assignment Step (Uses Triton Kernel) ---
        # Ensure assignments output tensor is int32 for the kernel if needed, then cast
        assignments_int32 = torch.empty(N_A, dtype=torch.int32, device=device)
        kmeans_assign_kernel[grid_assign](
            A_prep, centroids, assignments_int32, # Use int32 tensor here
            N_A, D, K,
            A_prep.stride(0), A_prep.stride(1),
            centroids.stride(0), centroids.stride(1),
            BLOCK_SIZE_N=BLOCK_SIZE_N_ASSIGN,
            BLOCK_SIZE_K_CHUNK=BLOCK_SIZE_K_CHUNK_ASSIGN,
            BLOCK_SIZE_D=BLOCK_SIZE_D_ASSIGN
        )
        # Cast assignments to int64 for scatter_add_ index
        assignments = assignments_int32.to(torch.int64)
        # Synchronize needed? PyTorch usually handles it, but maybe add for safety before PyTorch ops
        # torch.cuda.synchronize()

        # --- 2. Update Step (Uses PyTorch scatter_add_) ---
        update_start_time = time.time()

        new_sums = torch.zeros_like(centroids) # Shape (K, D)
        cluster_counts = torch.zeros(K, dtype=torch.float32, device=device) # Shape (K,)

        # Expand assignments to match the dimensions of A_prep for scatter_add_ on sums
        # index tensor needs same number of dimensions as src tensor (A_prep)
        idx_expand = assignments.unsqueeze(1).expand(N_A, D)

        # Add data points to corresponding centroid sums
        new_sums.scatter_add_(dim=0, index=idx_expand, src=A_prep)

        # Add 1 to counts for each data point's assigned cluster
        cluster_counts.scatter_add_(dim=0, index=assignments, src=torch.ones_like(assignments, dtype=torch.float32))

        # Calculate new centroids, handle empty clusters
        # Clamp counts to minimum 1 to avoid division by zero
        final_counts_safe = cluster_counts.clamp(min=1.0)
        new_centroids = new_sums / final_counts_safe.unsqueeze(1) # Use broadcasting

        # Keep old centroid if cluster becomes empty
        empty_cluster_mask = (cluster_counts == 0)
        # Use torch.where for efficient conditional assignment if needed, direct masking often works
        new_centroids[empty_cluster_mask] = centroids[empty_cluster_mask]

        update_time = time.time() - update_start_time

        # --- Check Convergence ---
        centroid_diff = torch.norm(new_centroids - centroids)
        centroids = new_centroids # Update centroids

        iter_end_time = time.time()
        print(f"  Iter {i+1}/{max_iters} | Centroid Diff: {centroid_diff:.4f} | Assign Time: {update_start_time - iter_start_time:.4f}s | Update Time: {update_time:.4f}s")

        if centroid_diff < tol:
            print(f"Converged after {i+1} iterations.")
            break

    if i == max_iters - 1:
        print(f"Reached max iterations ({max_iters}).")

    total_time = time.time() - start_time_total
    print(f"Total K-Means time: {total_time:.4f}s")

    # Return int64 assignments consistent with internal use
    return centroids, assignments

# ============================================================================
# Task 2.2: Approximate Nearest Neighbors (Simplified HNSW)
# ============================================================================

# --- Triton Kernel for 1 Query vs M Candidates (used by HNSW) ---
@triton.jit
def l2_dist_kernel_1_vs_M( # Renamed from previous example
    query_ptr,      # Pointer to query vector (D,)
    candidates_ptr, # Pointer to candidate vectors (M, D)
    output_ptr,     # Pointer to output distances (M,)
    M, D,           # Dimensions
    stride_cand_m, stride_cand_d,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Calculates squared L2 distance: 1 query vs M candidates."""
    pid_m = tl.program_id(axis=0) # Candidate index
    dist_sq = tl.zeros((), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_SIZE_D):
        offs_d = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offs_d < D
        # Load query block
        query_d_ptr = query_ptr + offs_d
        query_vals = tl.load(query_d_ptr, mask=mask_d, other=0.0)
        # Load candidate block
        cand_d_ptr = candidates_ptr + pid_m * stride_cand_m + offs_d * stride_cand_d
        cand_vals = tl.load(cand_d_ptr, mask=mask_d, other=0.0)
        # Accumulate difference squared
        diff = query_vals - cand_vals
        dist_sq += tl.sum(diff * diff, axis=0)
    # Store result
    tl.store(output_ptr + pid_m, dist_sq)

class SimpleHNSW_for_ANN: # Using previous class structure
    def __init__(self, dim, M=16, ef_construction=200, ef_search=50, mL=0.5):
        # --- Existing __init__ content ---
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
        self.BLOCK_SIZE_D_DIST = 128
        # --- End of existing __init__ content ---

    def _get_level_for_new_node(self):
        level = int(-math.log(random.uniform(0, 1)) * self.mL)
        return level

    def _distance(self, query_vec, candidate_indices):
        """Internal distance calc using the 1-vs-M Triton kernel (Squared L2)."""
        # --- Uses l2_dist_kernel_1_vs_M as defined before ---
        # (Code identical to previous working version)
        if not isinstance(candidate_indices, list): candidate_indices = list(candidate_indices)
        if not candidate_indices: return torch.empty(0, device=device), [] # Return empty indices too

        num_candidates = len(candidate_indices)
        # Ensure query is prepared (or assume it is)
        query_vec_prep, = _prepare_tensors(query_vec.flatten())
        # Gather candidate vectors from the main storage
        # Ensure indices are valid before indexing
        valid_indices = [idx for idx in candidate_indices if idx < self.node_count and idx >= 0]
        if not valid_indices:
             return torch.empty(0, device=device), []

        num_valid_candidates = len(valid_indices)
        candidate_vectors = self.vectors[valid_indices] # Index with valid indices only

        distances_out = torch.empty(num_valid_candidates, dtype=torch.float32, device=device)
        grid = (num_valid_candidates,)
        l2_dist_kernel_1_vs_M[grid](
            query_vec_prep, candidate_vectors, distances_out,
            num_valid_candidates, self.dim,
            candidate_vectors.stride(0), candidate_vectors.stride(1),
            BLOCK_SIZE_D=self.BLOCK_SIZE_D_DIST
        )
        # Return distances corresponding to the valid_indices used
        return distances_out, valid_indices

    def _distance_batch(self, query_indices, candidate_indices):
        """
        Calculates pairwise distances between batches using pairwise kernel.
        Needed for some heuristics (candidate-to-candidate distances).
        Returns a (len(query_indices), len(candidate_indices)) tensor.
        """
        # NOTE: This uses the *pairwise* L2 kernel, ensure it's defined correctly above.
        # If l2_dist_kernel_pairwise is not defined, this needs modification.
        # Assuming l2_dist_kernel_pairwise exists similar to the one for our_knn
        if not query_indices or not candidate_indices:
            return torch.empty((0,0), device=device)

        query_vectors = self.vectors[query_indices]
        candidate_vectors = self.vectors[candidate_indices]

        Q = len(query_indices)
        N = len(candidate_indices)
        D = self.dim

        Out = torch.empty((Q, N), dtype=torch.float32, device=device)
        grid = (Q, N)

        # Ensure vectors are prepared if necessary (e.g., contiguous)
        query_vectors, candidate_vectors = _prepare_tensors(query_vectors, candidate_vectors)

        # Assuming l2_dist_kernel_pairwise is defined and imported/available
        l2_dist_kernel_pairwise[grid](
            query_vectors, candidate_vectors, Out,
            Q, N, D,
            query_vectors.stride(0), query_vectors.stride(1),
            candidate_vectors.stride(0), candidate_vectors.stride(1),
            Out.stride(0), Out.stride(1),
            BLOCK_SIZE_D=DEFAULT_BLOCK_D # Use appropriate block size
        )
        return Out


    # --- REVISED _select_neighbors_heuristic METHOD ---
    def _select_neighbors_heuristic(self, query_vec, candidates, M_target):
        """
        Selects M_target neighbors from candidates using a heuristic.
        Variant: Keep closest overall, prune candidates overshadowed by selected ones.

        Args:
            query_vec (torch.Tensor): The vector being inserted/queried.
            candidates (list): List of (distance, node_id) tuples, sorted by distance.
                               Size is typically ef_construction.
            M_target (int): The number of neighbors to select (e.g., self.M).

        Returns:
            list: List of selected node IDs (up to M_target).
        """
        selected_neighbors = []
        # Use a min-heap for working candidates for efficient retrieval of closest
        working_candidates_heap = [(dist, nid) for dist, nid in candidates]
        heapq.heapify(working_candidates_heap)

        # Keep track of candidates discarded by the heuristic
        discarded_candidates = set()

        while working_candidates_heap and len(selected_neighbors) < M_target:
            # Get the best candidate (closest to query)
            dist_best, best_nid = heapq.heappop(working_candidates_heap)

            # Skip if already discarded by heuristic in a previous step
            if best_nid in discarded_candidates:
                continue

            # Add the best candidate to results
            selected_neighbors.append(best_nid)

            # --- Heuristic Pruning ---
            # Prune remaining candidates in the heap that are "overshadowed" by the newly added best_nid.
            # Heuristic: If dist(candidate, best_nid) < dist(candidate, query), discard candidate.
            # This requires candidate-to-candidate distances - potentially expensive!

            # Extract remaining candidates (IDs and distances to query) from heap for check
            # Rebuild heap later if needed, or use a different structure
            remaining_candidates_info = {} # {nid: dist_to_query}
            temp_heap = []
            while working_candidates_heap:
                 dist_r, nid_r = heapq.heappop(working_candidates_heap)
                 if nid_r not in discarded_candidates:
                      remaining_candidates_info[nid_r] = dist_r
                      temp_heap.append((dist_r, nid_r)) # Keep for potential rebuild
            working_candidates_heap = temp_heap # Restore for next main loop iteration
            heapq.heapify(working_candidates_heap) # Re-heapify

            remaining_nids = list(remaining_candidates_info.keys())

            if remaining_nids:
                # Calculate distances from the newly selected neighbor ('best_nid')
                # to all remaining candidates ('remaining_nids').
                # Need pairwise distance calculation capability.
                try:
                    # Assuming self._distance_batch calculates pairwise squared L2
                    dists_best_to_remaining = self._distance_batch([best_nid], remaining_nids) # Shape (1, len(remaining_nids))
                    dists_best_to_remaining = dists_best_to_remaining.squeeze(0) # Shape (len(remaining_nids),)

                    # Check heuristic
                    for i, r_nid in enumerate(remaining_nids):
                        dist_r_query = remaining_candidates_info[r_nid] # Distance from candidate r to query
                        dist_r_best = dists_best_to_remaining[i].item() # Distance from candidate r to best_nid

                        # Apply heuristic check (using squared distances is fine for comparison)
                        if dist_r_best < dist_r_query:
                            discarded_candidates.add(r_nid) # Mark to discard later

                except NameError:
                     # Fallback if _distance_batch or pairwise kernel isn't defined:
                     # Skip heuristic pruning if distance function is missing.
                     # This will revert to basically just taking the top M closest.
                     print("Warning: Skipping neighbor selection heuristic pruning due to missing distance function.")
                     pass # Or handle differently


        return selected_neighbors

    # --- REVISED add_point METHOD ---
    def add_point(self, point_vec):
        """Adds a single point to the graph using heuristic neighbor selection."""
        point_vec, = _prepare_tensors(point_vec.flatten()) # Prepare input
        new_node_id = self.node_count
        # Note: In a real implementation, dynamically resizing GPU tensors like this
        # can be inefficient. Pre-allocation or different memory strategies are often used.
        self.vectors = torch.cat((self.vectors, point_vec.unsqueeze(0)), dim=0)
        self.node_count += 1

        node_level = self._get_level_for_new_node()
        self.level_assignments.append(node_level)

        # Expand graph structure (CPU lists)
        while node_level >= len(self.graph): self.graph.append([])
        for lvl in range(len(self.graph)):
             while len(self.graph[lvl]) <= new_node_id: self.graph[lvl].append([])

        current_entry_point = self.entry_point
        current_max_level = self.max_level

        if current_entry_point == -1: # First node
            self.entry_point = new_node_id
            self.max_level = node_level
            return new_node_id

        # Search from top layer down to insertion level + 1 to find EPs
        ep = [current_entry_point]
        for level in range(current_max_level, node_level, -1):
             # Ensure graph structure is valid for this level and EP
             if level >= len(self.graph) or ep[0] >= len(self.graph[level]):
                  continue # Skip if level or node doesn't exist in graph structure yet
             search_results = self._search_layer(point_vec, ep, level, ef=1)
             if not search_results: break
             ep = [search_results[0][1]]

        # Connect node at each level from min(node_level, current_max_level) down to 0
        for level in range(min(node_level, current_max_level), -1, -1):
             # Ensure graph structure is valid before search
             if level >= len(self.graph) or not ep or any(idx >= len(self.graph[level]) for idx in ep):
                 # Handle cases where entry points might be invalid for the level
                 # This might happen if levels were added but nodes weren't fully connected yet.
                 # A safer bet might be to use the global entry point if ep is invalid here.
                 if current_entry_point < len(self.graph[level]):
                     ep = [current_entry_point]
                 else:
                     continue # Cannot proceed at this level

             # Search layer to find candidate neighbors (size ef_construction)
             neighbors_found_with_dist = self._search_layer(point_vec, ep, level, self.ef_construction)

             if not neighbors_found_with_dist:
                 # If search yields nothing, maybe use EPs from upper layer?
                 # Or handle appropriately. For now, skip connecting if no neighbors found.
                 continue

             # --- Use Heuristic Selection ---
             # Select M neighbors using the heuristic method
             selected_neighbor_ids = self._select_neighbors_heuristic(point_vec, neighbors_found_with_dist, self.M)

             # --- Add Connections ---
             # Add forward connections (new_node -> selected_neighbors)
             self.graph[level][new_node_id] = selected_neighbor_ids

             # Add backward connections (selected_neighbors -> new_node) with pruning
             for neighbor_id in selected_neighbor_ids:
                 # Ensure neighbor_id is valid for this level's graph structure
                 if neighbor_id >= len(self.graph[level]):
                      # This indicates a potential issue if search returned an invalid ID
                      print(f"Warning: Neighbor ID {neighbor_id} out of bounds for level {level}. Skipping connection.")
                      continue

                 neighbor_connections = self.graph[level][neighbor_id]
                 if new_node_id not in neighbor_connections: # Avoid duplicate links
                     if len(neighbor_connections) < self.M:
                         neighbor_connections.append(new_node_id)
                     else:
                         # --- Pruning Logic ---
                         # Calculate distance from neighbor_id to new_node_id
                         # Calculate distances from neighbor_id to its current neighbors
                         # If new_node is closer than the furthest current neighbor, replace it.

                         dist_new, _ = self._distance(self.vectors[neighbor_id], [new_node_id])
                         if not dist_new: continue # Handle error if distance calc fails
                         dist_new = dist_new[0].item()

                         current_neighbor_ids = list(neighbor_connections) # Copy list
                         dists_to_current, valid_curr_ids = self._distance(self.vectors[neighbor_id], current_neighbor_ids)

                         if dists_to_current.numel() > 0:
                              furthest_dist = -1.0
                              furthest_idx_in_list = -1
                              # Map distances back to original list indices if valid_curr_ids differs
                              dist_map = {nid: d.item() for nid, d in zip(valid_curr_ids, dists_to_current)}

                              for list_idx, current_nid in enumerate(current_neighbor_ids):
                                   d = dist_map.get(current_nid, float('inf'))
                                   if d > furthest_dist:
                                        furthest_dist = d
                                        furthest_idx_in_list = list_idx

                              # If new node is better than the worst current one, replace
                              if furthest_idx_in_list != -1 and dist_new < furthest_dist:
                                   neighbor_connections[furthest_idx_in_list] = new_node_id
                         elif len(neighbor_connections) < self.M: # Should have been caught earlier, but safe fallback
                              neighbor_connections.append(new_node_id)


             # Update entry points for the next level down (use selected neighbors)
             # Using only selected neighbors might be better than all found
             ep = selected_neighbor_ids # Or use neighbors_found_with_dist? HNSW paper detail. Let's try selected.
             if not ep: # If selection yielded nothing, fall back?
                  ep = [nid for _, nid in neighbors_found_with_dist[:1]] # Fallback to closest found

        # Update global entry point if this node is the highest level
        if node_level > self.max_level:
            self.max_level = node_level
            self.entry_point = new_node_id
        return new_node_id


    # --- _search_layer METHOD ---
    # (Assumed to be identical to previous working version - uses self._distance)
    def _search_layer(self, query_vec, entry_points, target_level, ef):
        """Performs greedy search on a single layer (identical to previous example)."""
        if self.entry_point == -1: return []
        # Ensure entry points are valid indices
        valid_entry_points = [ep for ep in entry_points if ep < self.node_count and ep >=0]
        if not valid_entry_points:
             if self.entry_point != -1: valid_entry_points = [self.entry_point] # Fallback
             else: return []

        candidates = set(valid_entry_points)
        visited = set(valid_entry_points)
        initial_distances, valid_indices_init = self._distance(query_vec, valid_entry_points)
        if initial_distances.numel() == 0: return [] # Cannot start search

        # Map distances back to original requested entry points if some were invalid
        dist_map_init = {nid: d.item() for nid, d in zip(valid_indices_init, initial_distances)}

        candidate_heap = [(dist_map_init.get(ep, float('inf')), ep) for ep in valid_entry_points]
        heapq.heapify(candidate_heap)
        results_heap = [(-dist, node_id) for dist, node_id in candidate_heap if dist != float('inf')]
        heapq.heapify(results_heap)

        while candidate_heap:
            dist_candidate, current_node_id = heapq.heappop(candidate_heap)
            if dist_candidate == float('inf'): continue # Skip if invalid distance somehow got in

            if results_heap:
                 dist_furthest_result = -results_heap[0][0]
                 if dist_candidate > dist_furthest_result and len(results_heap) >= ef: break

            try: neighbors = self.graph[target_level][current_node_id]
            except IndexError: neighbors = []

            unvisited_neighbors = [n for n in neighbors if n not in visited]
            if unvisited_neighbors:
                 visited.update(unvisited_neighbors)
                 neighbor_distances, valid_neighbor_indices = self._distance(query_vec, unvisited_neighbors)
                 if neighbor_distances.numel() == 0: continue

                 # Map distances back
                 dist_map_neighbors = {nid: d.item() for nid, d in zip(valid_neighbor_indices, neighbor_distances)}

                 for neighbor_id in valid_neighbor_indices: # Iterate only valid neighbors
                      neighbor_dist_val = dist_map_neighbors[neighbor_id]
                      if len(results_heap) < ef or neighbor_dist_val < -results_heap[0][0]:
                           if len(results_heap) >= ef: heapq.heappop(results_heap)
                           heapq.heappush(results_heap, (-neighbor_dist_val, neighbor_id))

                      # Only add to candidate heap if potentially better than furthest result
                      # (or if heap isn't full) - minor optimization
                      if len(results_heap) < ef or neighbor_dist_val < -results_heap[0][0]:
                           heapq.heappush(candidate_heap, (neighbor_dist_val, neighbor_id))

        final_results = sorted([(abs(neg_dist), node_id) for neg_dist, node_id in results_heap])
        return final_results


    # --- search_knn METHOD ---
    # (Assumed to be identical to previous working version - calls _search_layer)
    def search_knn(self, query_vec, k):
        """Searches for k nearest neighbors (identical to previous example)."""
        if self.entry_point == -1: return []
        query_vec, = _prepare_tensors(query_vec.flatten())

        ep = [self.entry_point]
        current_max_level = self.max_level
        for level in range(current_max_level, 0, -1):
             if level >= len(self.graph) or ep[0] >= len(self.graph[level]): continue # Safety check
             search_results = self._search_layer(query_vec, ep, level, ef=1)
             if not search_results: break
             ep = [search_results[0][1]]
        # Final search at level 0
        if 0 >= len(self.graph) or not ep or ep[0] >= len(self.graph[0]): # Check level 0 validity
             if self.entry_point != -1 and 0 < len(self.graph) and self.entry_point < len(self.graph[0]):
                  ep = [self.entry_point] # Fallback to global EP for level 0 if needed
             else:
                  return [] # Cannot search level 0

        neighbors_found = self._search_layer(query_vec, ep, 0, self.ef_search)
        return neighbors_found[:k]


# --- Ensure the pairwise kernel is defined or imported ---
# Make sure l2_dist_kernel_pairwise from the kNN section is available here

# --- Ensure DEFAULT_BLOCK_D is defined ---
DEFAULT_BLOCK_D = 128 # Or your preferred default

# --- our_ann function remains the same, calling this class ---
def our_ann(N_A, D, A, X, K, M=16, ef_construction=100, ef_search=50):
     # ... (identical to previous version, just instantiates the updated class) ...
    A_prep, X_prep = _prepare_tensors(A, X)
    Q = X_prep.shape[0]
    assert A_prep.shape[0] == N_A, "N_A doesn't match A.shape[0]"
    assert A_prep.shape[1] == D, "D doesn't match A.shape[1]"
    assert X_prep.shape[1] == D, "D doesn't match X.shape[1]"
    assert K > 0, "K must be positive"

    print(f"Running ANN (HNSW w/ Heuristics): Q={Q}, N={N_A}, D={D}, K={K}")
    print(f"HNSW Params: M={M}, efC={ef_construction}, efS={ef_search}")

    # 1. Build the HNSW Index
    start_build = time.time()
    hnsw_index = SimpleHNSW_for_ANN(dim=D, M=M, ef_construction=ef_construction, ef_search=ef_search)

    print("Building index...")
    # Incremental build using the revised add_point
    for i in range(N_A):
        hnsw_index.add_point(A_prep[i])
        if (i + 1) % (N_A // 10 + 1) == 0:
             print(f"  Added {i+1}/{N_A} points...")

    end_build = time.time()
    print(f"Index build time: {end_build - start_build:.2f} seconds")
    # Check if graph was actually built
    if hnsw_index.node_count == 0 or hnsw_index.entry_point == -1 :
        print("Error: Index build resulted in an empty graph.")
        return torch.full((Q, K), -1, dtype=torch.int64), torch.full((Q, K), float('inf'), dtype=torch.float32)


    # 2. Perform Search for each query
    start_search = time.time()
    all_indices = torch.full((Q, K), -1, dtype=torch.int64, device=device)
    all_distances = torch.full((Q, K), float('inf'), dtype=torch.float32, device=device)

    print("Searching queries...")
    for q_idx in range(Q):
        results = hnsw_index.search_knn(X_prep[q_idx], K)
        num_results = len(results)
        if num_results > 0:
            q_dists = torch.tensor([res[0] for res in results], dtype=torch.float32, device=device)
            q_indices = torch.tensor([res[1] for res in results], dtype=torch.int64, device=device)
            # Ensure slicing doesn't go out of bounds if num_results < K
            k_actual = min(num_results, K)
            all_distances[q_idx, :k_actual] = q_dists[:k_actual]
            all_indices[q_idx, :k_actual] = q_indices[:k_actual]
        if (q_idx + 1) % (Q // 10 + 1) == 0:
             print(f"  Searched {q_idx+1}/{Q} queries...")

    end_search = time.time()
    print(f"ANN search time: {end_search - start_search:.4f} seconds")

    return all_indices, all_distances
# ============================================================================
# Example Usage (Illustrative)
# ============================================================================
if __name__ == "__main__":
    N_data = 5000
    N_queries = 100
    Dim = 128
    K_val = 10

    print("="*40)
    print("Generating Data...")
    print("="*40)
    # Database vectors
    A_data = torch.randn(N_data, Dim, dtype=torch.float32, device=device)
    # Query vectors
    X_queries = torch.randn(N_queries, Dim, dtype=torch.float32, device=device)


    print("\n" + "="*40)
    print("Testing distance_l2...")
    print("="*40)
    l2_dists = distance_l2(X_queries[:2], A_data[:5])
    print("Sample L2 distances (squared) shape:", l2_dists.shape)
    print(l2_dists)

    print("\n" + "="*40)
    print("Testing distance_cosine...")
    print("="*40)
    cos_dists = distance_cosine(X_queries[:2], A_data[:5])
    print("Sample Cosine distances shape:", cos_dists.shape)
    print(cos_dists)

    print("\n" + "="*40)
    print("Testing distance_manhattan...")
    print("="*40)
    man_dists = distance_manhattan(X_queries[:2], A_data[:5])
    print("Sample Manhattan distances shape:", man_dists.shape)
    print(man_dists)

    print("\n" + "="*40)
    print(f"Testing our_knn (K={K_val})...")
    print("="*40)
    knn_indices, knn_dists = our_knn(N_data, Dim, A_data, X_queries, K_val)
    print("KNN results shape (Indices):", knn_indices.shape)
    print("KNN results shape (Distances):", knn_dists.shape)
    print("Sample KNN Indices (Query 0):\n", knn_indices[0])
    print("Sample KNN Dists (Query 0):\n", knn_dists[0])


    print("\n" + "="*40)
    print(f"Testing our_kmeans (K=5)...")
    print("="*40)
    K_clusters = 5
    kmeans_centroids, kmeans_assignments = our_kmeans(N_data, Dim, A_data, K_clusters)
    print("KMeans centroids shape:", kmeans_centroids.shape)
    print("KMeans assignments shape:", kmeans_assignments.shape)
    print("Sample KMeans Assignments:\n", kmeans_assignments[:20])
    # Verify counts
    ids, counts = torch.unique(kmeans_assignments, return_counts=True)
    print("Cluster counts:")
    for id_val, count_val in zip(ids.tolist(), counts.tolist()):
        print(f"  Cluster {id_val}: {count_val}")


    print("\n" + "="*40)
    print(f"Testing our_ann (HNSW, K={K_val})...")
    print("="*40)
    # Reduce HNSW params for quicker example run
    ann_indices, ann_dists = our_ann(N_data, Dim, A_data, X_queries, K_val,
                                 M=32,              # Increased connections
                                 ef_construction=200, # Increased construction beam
                                 ef_search=100) 
    print("ANN results shape (Indices):", ann_indices.shape)
    print("ANN results shape (Distances):", ann_dists.shape)
    print("Sample ANN Indices (Query 0):\n", ann_indices[0])
    print("Sample ANN Dists (Query 0):\n", ann_dists[0])

    # Optional: Compare ANN vs KNN recall for query 0
    if N_queries > 0 and K_val > 0:
        true_knn_ids = set(knn_indices[0].tolist())
        approx_ann_ids = set(ann_indices[0].tolist())
        # Remove potential -1 placeholders if K > actual neighbors found
        approx_ann_ids.discard(-1)

        recall = len(true_knn_ids.intersection(approx_ann_ids)) / K_val
        print(f"\nANN Recall @ {K_val} for Query 0 (vs brute-force KNN): {recall:.2%}")
'''
def test_kmeans():
    N, D, A, K = testdata_kmeans("test_file.json")
    kmeans_result = our_kmeans(N, D, A, K)
    print(kmeans_result)

def test_knn():
    N, D, A, X, K = testdata_knn("test_file.json")
    knn_result = our_knn(N, D, A, X, K)
    print(knn_result)
    
def test_ann():
    N, D, A, X, K = testdata_ann("test_file.json")
    ann_result = our_ann(N, D, A, X, K)
    print(ann_result)
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

def main():
    np.random.seed(0)
    #size = 1 << 20  # 1 million elements
    n = 1000
    d = 100
    # Transfer to GPU.
    A_cp = cp.random.rand(n, d).astype(cp.float32)
    B_cp = cp.random.rand(n, d).astype(cp.float32)
    
    start_time = time.time()
    l2 = distance_l2(A_cp, B_cp)
    end_time = time.time()
    print(f"L2 distance computation time: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    dot = distance_dot(A_cp, B_cp)
    end_time = time.time()
    print(f"Dot Product distance computation time: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    manhattan = distance_manhattan(A_cp, B_cp)
    end_time = time.time()
    print(f"Manhattan distance computation time: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    cosine = distance_cosine(A_cp, B_cp)
    end_time = time.time()
    print(f"Cosine distance computation time: {end_time - start_time:.4f} seconds")

    
    print("L2 (Euclidean) distance:", l2)
    print("Dot product:", dot)
    print("Manhattan (L1) distance:", manhattan)
    print("Cosine distance:", cosine)

    knn_idx, knn_dists = our_knn(A_cp.shape[0], A_cp.shape[1], A_cp, B_cp, 5, distance_l2)
    print(knn_idx, knn_dists)
    knn_idx, knn_dists = our_knn(A_cp.shape[0], A_cp.shape[1], A_cp, B_cp, 5, distance_cosine)
    print(knn_idx, knn_dists)
    knn_idx, knn_dists = our_knn(A_cp.shape[0], A_cp.shape[1], A_cp, B_cp, 5, distance_manhattan)
    print(knn_idx, knn_dists)
    knn_idx, knn_dists = our_knn(A_cp.shape[0], A_cp.shape[1], A_cp, B_cp, 5, distance_dot)
    print(knn_idx, knn_dists)


if __name__ == "__main__":
    main()
'''