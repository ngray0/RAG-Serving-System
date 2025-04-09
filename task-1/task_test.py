# task-1/task.py
import torch
print("load")
import triton
print("loading")
import triton.language as tl
print("Loading.")
import math
print("Loading..")
import heapq # For HNSW priority queues
print("Loading...")
import random
print("Loading....")
import time
print("Loading......")
# Removed: import cupy as cp - CuPy is no longer used

# --- Ensure GPU is available ---
if not torch.cuda.is_available():
    print("CUDA not available, exiting.")
    exit()
device = torch.device("cuda:0")
print(f"Using device: {device}")

# ============================================================================
# Triton Distance Kernels (Pairwise: Q queries vs N database points)
# ============================================================================
DEFAULT_BLOCK_Q = 32
DEFAULT_BLOCK_N = 64
DEFAULT_BLOCK_K = 64 # Block size for the reduction dimension D (used by all tiled kernels)

def ceil_div(a, b):
    return (a + b - 1) // b

# --- Optimized Tiled Dot Product Kernel ---
@triton.autotune(
       configs=[
        # Min block sizes are 16 due to tl.dot constraints
        triton.Config({'BLOCK_Q': 16, 'BLOCK_N': 16, 'BLOCK_K': 16, 'num_stages': 1, 'num_warps': 2}),
        triton.Config({'BLOCK_Q': 16, 'BLOCK_N': 16, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 16, 'BLOCK_N': 16, 'BLOCK_K': 64, 'num_stages': 1, 'num_warps': 4}),

        triton.Config({'BLOCK_Q': 16, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 32, 'BLOCK_N': 16, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 16, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 16, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),

        triton.Config({'BLOCK_Q': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 32, 'BLOCK_N': 32, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}), # Added stage variation
        triton.Config({'BLOCK_Q': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}), # Added stage variation
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}), # Added stage variation

        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),

        # Keep a few larger options if hardware potentially supports it
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 1, 'num_warps': 4}),
    ],
    key=['Q', 'N', 'D'],
)
@triton.jit
def dot_kernel_pairwise_tiled(
    X_ptr, A_ptr, Out_ptr,           # Data pointers
    Q, N, D,                         # Matrix dimensions
    stride_xq, stride_xd,            # Strides for X (row, col)
    stride_an, stride_ad,            # Strides for A (row, col)
    stride_outq, stride_outn,        # Strides for Out (row, col)
    BLOCK_Q: tl.constexpr,           # Tile size for Q dimension
    BLOCK_N: tl.constexpr,           # Tile size for N dimension
    BLOCK_K: tl.constexpr,           # Tile size for D dimension (often called K)
):
    """
    Calculates pairwise dot product using tiling: Out[q, n] = dot(X[q, :], A[n, :])
    Equivalent to MatMul: Out = X @ A.T
    """
    pid_q_block = tl.program_id(axis=0)
    pid_n_block = tl.program_id(axis=1)
    offs_q = pid_q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_n = pid_n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_Q, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, D, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Load X tile
        x_ptrs = X_ptr + (offs_q[:, None] * stride_xq + offs_k[None, :] * stride_xd)
        q_mask = offs_q[:, None] < Q
        k_mask_x = offs_k[None, :] < D
        x_mask = q_mask & k_mask_x
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        # Load A tile
        a_ptrs = A_ptr + (offs_n[:, None] * stride_an + offs_k[None, :] * stride_ad)
        n_mask = offs_n[:, None] < N
        k_mask_a = offs_k[None, :] < D
        a_mask = n_mask & k_mask_a
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        # Compute dot product for tiles
        accumulator += tl.dot(x_tile, tl.trans(a_tile))

    # Store result tile
    out_ptrs = Out_ptr + (offs_q[:, None] * stride_outq + offs_n[None, :] * stride_outn)
    out_mask = (offs_q[:, None] < Q) & (offs_n[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=out_mask)

@triton.autotune(
        configs=[
        # --- Blocks including 8x? or ?x8 ---
        triton.Config({'BLOCK_Q': 8,  'BLOCK_N': 8,  'BLOCK_K': 16, 'num_stages': 1, 'num_warps': 2}),
        triton.Config({'BLOCK_Q': 8,  'BLOCK_N': 8,  'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 8,  'BLOCK_N': 16, 'BLOCK_K': 16, 'num_stages': 1, 'num_warps': 2}),
        triton.Config({'BLOCK_Q': 16, 'BLOCK_N': 8,  'BLOCK_K': 16, 'num_stages': 1, 'num_warps': 2}),
        triton.Config({'BLOCK_Q': 8,  'BLOCK_N': 16, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 16, 'BLOCK_N': 8,  'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 8,  'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 32, 'BLOCK_N': 8,  'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),

        # --- Blocks including 16x? or ?x16 ---
        triton.Config({'BLOCK_Q': 16, 'BLOCK_N': 16, 'BLOCK_K': 16, 'num_stages': 1, 'num_warps': 2}),
        triton.Config({'BLOCK_Q': 16, 'BLOCK_N': 16, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 16, 'BLOCK_N': 16, 'BLOCK_K': 64, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 16, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 32, 'BLOCK_N': 16, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 16, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 16, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),

        # --- Blocks including 32x? or ?x32 ---
        triton.Config({'BLOCK_Q': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 32, 'BLOCK_N': 32, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),

        # --- Blocks including 64x? or ?x64 (Keep a few, maybe less optimal for GTX 1060) ---
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
    ],
    key=['Q', 'N', 'D'],
)
@triton.jit
def manhattan_kernel_pairwise_tiled(
    X_ptr, A_ptr, Out_ptr,           # Data pointers
    Q, N, D,                         # Matrix dimensions
    stride_xq, stride_xd,            # Strides for X (row, col)
    stride_an, stride_ad,            # Strides for A (row, col)
    stride_outq, stride_outn,        # Strides for Out (row, col)
    BLOCK_Q: tl.constexpr,           # Tile size for Q dimension
    BLOCK_N: tl.constexpr,           # Tile size for N dimension
    BLOCK_K: tl.constexpr,           # Tile size for D dimension (often called K)
):
    """
    Calculates pairwise Manhattan (L1) distance using tiling:
    Out[q, n] = sum(abs(X[q, :] - A[n, :]))
    Each program instance computes a BLOCK_Q x BLOCK_N tile of the output.
    """
    # 1. Program ID and Offsets
    pid_q_block = tl.program_id(axis=0)
    pid_n_block = tl.program_id(axis=1)
    offs_q = pid_q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_n = pid_n_block * BLOCK_N + tl.arange(0, BLOCK_N)

    # 2. Initialize Accumulator Tile for L1 distances
    accumulator = tl.zeros((BLOCK_Q, BLOCK_N), dtype=tl.float32)

    # 3. Loop over the Dimension D (K dimension) in blocks of BLOCK_K
    for k_start in range(0, D, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # --- Load X tile ---
        x_ptrs = X_ptr + (offs_q[:, None] * stride_xq + offs_k[None, :] * stride_xd)
        q_mask = offs_q[:, None] < Q
        k_mask_x = offs_k[None, :] < D
        x_mask = q_mask & k_mask_x
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0) # Shape (BLOCK_Q, BLOCK_K)

        # --- Load A tile ---
        a_ptrs = A_ptr + (offs_n[:, None] * stride_an + offs_k[None, :] * stride_ad)
        n_mask = offs_n[:, None] < N
        k_mask_a = offs_k[None, :] < D
        a_mask = n_mask & k_mask_a
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0) # Shape (BLOCK_N, BLOCK_K)

        # --- Compute Absolute Differences and Sum ---
        # Use broadcasting to calculate pairwise differences within tiles
        # x_tile: (BLOCK_Q, BLOCK_K) -> (BLOCK_Q, 1, BLOCK_K)
        # a_tile: (BLOCK_N, BLOCK_K) -> Transpose needed for broadcasting: (1, BLOCK_N, BLOCK_K) ?
        # Let's load A transposed conceptually. Load a_tile as (BLOCK_K, BLOCK_N)
        # No, the previous load loads A as (BLOCK_N, BLOCK_K). Correct.
        # Broadcast: x_tile[:, None, :] - a_tile[None, :, :]
        #   -> (BLOCK_Q, 1, BLOCK_K) - (1, BLOCK_N, BLOCK_K) -> (BLOCK_Q, BLOCK_N, BLOCK_K)
        diff = x_tile[:, None, :] - a_tile[None, :, :]
        abs_diff = tl.abs(diff)

        # Sum absolute differences over the K dimension block (axis=2)
        # Result shape: (BLOCK_Q, BLOCK_N)
        accumulator += tl.sum(abs_diff, axis=2)

    # 4. Store the Resulting Tile
    out_ptrs = Out_ptr + (offs_q[:, None] * stride_outq + offs_n[None, :] * stride_outn)
    out_mask = (offs_q[:, None] < Q) & (offs_n[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=out_mask)


# --- Kernels needed for K-Means and HNSW ---
# (These remain unchanged as they use different distance logic or simple kernels)
@triton.jit
def kmeans_assign_kernel(
    A_ptr, centroids_ptr, assignments_ptr,
    N, D, K,
    stride_an, stride_ad, stride_ck, stride_cd,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K_CHUNK: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,
):
    """Assigns each point in A to the nearest centroid (Squared L2) by iterating dimensions."""
    pid_n_block = tl.program_id(axis=0)
    offs_n = pid_n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N

    min_dist_sq = tl.full((BLOCK_SIZE_N,), float('inf'), dtype=tl.float32)
    best_assignment = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)

    for k_start in range(0, K, BLOCK_SIZE_K_CHUNK):
        k_end = tl.minimum(k_start + BLOCK_SIZE_K_CHUNK, K)
        for k_idx in range(BLOCK_SIZE_K_CHUNK):
            actual_k = k_start + k_idx
            if actual_k < k_end:
                current_dist_sq = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
                for d_start in range(0, D, BLOCK_SIZE_D):
                    offs_d = d_start + tl.arange(0, BLOCK_SIZE_D)
                    mask_d = offs_d < D
                    centroid_d_ptr = centroids_ptr + actual_k * stride_ck + offs_d * stride_cd
                    centroid_vals = tl.load(centroid_d_ptr, mask=mask_d, other=0.0)
                    points_d_ptr = A_ptr + offs_n[:, None] * stride_an + offs_d[None, :] * stride_ad
                    points_vals = tl.load(points_d_ptr, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
                    diff = points_vals - centroid_vals[None, :]
                    current_dist_sq += tl.sum(diff * diff, axis=1)

                is_closer = current_dist_sq < min_dist_sq
                min_dist_sq = tl.where(is_closer, current_dist_sq, min_dist_sq)
                best_assignment = tl.where(is_closer, actual_k, best_assignment)

    assignments_out_ptrs = assignments_ptr + offs_n
    tl.store(assignments_out_ptrs, best_assignment, mask=mask_n)

@triton.jit
def l2_dist_kernel_1_vs_M(
    query_ptr, candidates_ptr, output_ptr,
    M, D, stride_cand_m, stride_cand_d,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Calculates squared L2 distance: 1 query vs M candidates."""
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

# ============================================================================
# Helper Functions
# ============================================================================
def _prepare_tensors(*tensors, target_device =device):
    """Ensure tensors are float32, contiguous, and on the correct device."""
    prepared = []
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=target_device)
        if t.device != target_device:
            t = t.to(target_device)
        if t.dtype != torch.float32:
            t = t.to(dtype=torch.float32)
        prepared.append(t.contiguous())
    return prepared

# ============================================================================
# Python Distance Function Wrappers using Triton / PyTorch
# ============================================================================

def distance_dot_triton(X, A, **kwargs):
    """Computes pairwise dot product (X @ A.T) using the tiled Triton kernel."""
    target_device = X.device
    X_prep, A_prep = _prepare_tensors(X, A, target_device=target_device)
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    assert D == D_A, f"Dimension mismatch: X({D}) vs A({D_A})"
    Out = torch.empty((Q, N), dtype=torch.float32, device=target_device)
    BLOCK_Q = kwargs.get('BLOCK_Q', DEFAULT_BLOCK_Q)
    BLOCK_N = kwargs.get('BLOCK_N', DEFAULT_BLOCK_N)
    grid = (ceil_div(Q, BLOCK_Q), ceil_div(N, BLOCK_N))
    # print(f"Launching Triton Kernel dot_kernel_pairwise_tiled with grid={grid}") # Optional verbose
    dot_kernel_pairwise_tiled[grid](
        X_prep, A_prep, Out,
        Q, N, D,
        X_prep.stride(0), X_prep.stride(1), A_prep.stride(0), A_prep.stride(1),
        Out.stride(0), Out.stride(1),
        **kwargs
    )
    return Out

def distance_l2_triton(X, A, **kwargs):
    """
    Computes pairwise L2 (Euclidean) distances using the tiled dot product kernel
    and PyTorch operations for norms.
    """
    target_device = X.device
    X_prep, A_prep = _prepare_tensors(X, A, target_device=target_device)
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    assert D == D_A, f"Dimension mismatch: X({D}) vs A({D_A})"
    # print(f"Calculating pairwise L2 (Triton Dot + PyTorch Norm) for shapes: {X_prep.shape} and {A_prep.shape}") # Optional verbose

    dot_products = distance_dot_triton(X_prep, A_prep, **kwargs) # (Q, N)
    X_norm_sq = torch.sum(X_prep**2, axis=1, keepdims=True)  # (Q, 1)
    A_norm_sq = torch.sum(A_prep**2, axis=1, keepdims=True)  # (N, 1)
    dist_sq = X_norm_sq - 2 * dot_products + A_norm_sq.T # (Q, N)
    dist_sq.clamp_(min=0.0)
    dist = torch.sqrt(dist_sq)
    return dist

def distance_cosine_triton(X, A, epsilon=1e-8, **kwargs):
    """
    Computes pairwise Cosine distances using the tiled dot product kernel
    and PyTorch operations for norms.
    """
    target_device = X.device
    X_prep, A_prep = _prepare_tensors(X, A, target_device=target_device)
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    assert D == D_A, f"Dimension mismatch: X({D}) vs A({D_A})"
    # print(f"Calculating pairwise Cosine (Triton Dot + PyTorch Norm) for shapes: {X_prep.shape} and {A_prep.shape}") # Optional verbose

    dot_products = distance_dot_triton(X_prep, A_prep, **kwargs) # (Q, N)
    X_norm = torch.linalg.norm(X_prep, axis=1, keepdims=True) # (Q, 1)
    A_norm = torch.linalg.norm(A_prep, axis=1, keepdims=True) # (N, 1)
    norm_product = X_norm * A_norm.T # (Q, N)
    cosine_similarity = dot_products / (norm_product + epsilon)
    cosine_similarity.clamp_(min=-1.0, max=1.0)
    cosine_distance = 1.0 - cosine_similarity
    return cosine_distance

def distance_manhattan_triton(X, A, **kwargs):
    """Computes pairwise Manhattan (L1) distance using the tiled Triton kernel."""
    target_device = X.device
    X_prep, A_prep = _prepare_tensors(X, A, target_device=target_device)
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    assert D == D_A, f"Dimension mismatch: X({D}) vs A({D_A})"
    # print(f"Calculating pairwise Manhattan (Tiled Triton Kernel) for shapes: {X_prep.shape} and {A_prep.shape}") # Optional verbose

    Out = torch.empty((Q, N), dtype=torch.float32, device=target_device)
    BLOCK_Q = kwargs.get('BLOCK_Q', DEFAULT_BLOCK_Q)
    BLOCK_N = kwargs.get('BLOCK_N', DEFAULT_BLOCK_N)
    grid = (ceil_div(Q, BLOCK_Q), ceil_div(N, BLOCK_N))
    # print(f"Launching Triton Kernel manhattan_kernel_pairwise_tiled with grid={grid}") # Optional verbose

    # Add synchronization for debugging hangs
    manhattan_kernel_pairwise_tiled[grid](
        X_prep, A_prep, Out,
        Q, N, D,
        X_prep.stride(0), X_prep.stride(1), A_prep.stride(0), A_prep.stride(1),
        Out.stride(0), Out.stride(1),
        # Pass fixed block sizes for debugging, comment out autotune on kernel
        # BLOCK_Q=16, BLOCK_N=16, BLOCK_K=16
        # Or pass from autotuner via kwargs if autotune is enabled
         **kwargs
    )
    # Try uncommenting synchronize if debugging hangs persists
    # torch.cuda.synchronize()
    return Out

# ============================================================================
# Task 1.2: k-Nearest Neighbors (Brute Force)
# ============================================================================
def our_knn2(N_A, D, A, X, K):
    """
    Finds the K nearest neighbors in A for each query vector in X using
    brute-force pairwise L2 distance calculation (via distance_l2_triton).
    """
    target_device = X.device
    A_prep, X_prep = _prepare_tensors(A, X, target_device=target_device)
    Q = X_prep.shape[0]
    assert A_prep.shape[0] == N_A and A_prep.shape[1] == D and X_prep.shape[1] == D and K > 0 and K <= N_A

    print(f"Running k-NN: Q={Q}, N={N_A}, D={D}, K={K}")
    start_time = time.time()
    all_distances = distance_l2_triton(X_prep, A_prep) # Shape (Q, N_A) -> Returns actual L2
    topk_distances, topk_indices = torch.topk(all_distances, k=K, dim=1, largest=False)
    end_time = time.time()
    print(f"k-NN computation time: {end_time - start_time:.4f} seconds")
    return topk_indices, topk_distances
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
    all_distances = distance_l2_triton(X_prep, A_prep) # Shape (Q, N_A)

    # 2. Find the top K smallest distances for each query
    #    largest=False gives smallest distances (nearest neighbors)
    topk_distances, topk_indices = torch.topk(all_distances, k=K, dim=1, largest=False)

    end_time = time.time()
    print(f"k-NN computation time: {end_time - start_time:.4f} seconds")

    return topk_indices, topk_distances

# ============================================================================
# Task 2.1: K-Means Clustering
# ============================================================================
# (K-Means implementation remains unchanged, using its specific kernel)
def our_kmeans(N_A, D, A, K, max_iters=100, tol=1e-4):
    target_device = A.device
    A_prep, = _prepare_tensors(A, target_device=target_device)
    assert A_prep.shape[0] == N_A and A_prep.shape[1] == D and K > 0 and K <= N_A

    print(f"Running K-Means (Update with PyTorch): N={N_A}, D={D}, K={K}")
    start_time_total = time.time()
    initial_indices = torch.randperm(N_A, device=device)[:K]
    centroids = A_prep[initial_indices].clone()
    assignments = torch.empty(N_A, dtype=torch.int64, device=device)
    BLOCK_SIZE_N_ASSIGN = 128
    BLOCK_SIZE_K_CHUNK_ASSIGN = 64
    BLOCK_SIZE_D_ASSIGN = DEFAULT_BLOCK_K # Adjusted default
    grid_assign = lambda meta: (triton.cdiv(N_A, meta['BLOCK_SIZE_N']),)

    for i in range(max_iters):
        #iter_start_time = time.time() # Optional timing
        assignments_int32 = torch.empty(N_A, dtype=torch.int32, device=device)
        kmeans_assign_kernel[grid_assign](
            A_prep, centroids, assignments_int32,
            N_A, D, K,
            A_prep.stride(0), A_prep.stride(1), centroids.stride(0), centroids.stride(1),
            BLOCK_SIZE_N=BLOCK_SIZE_N_ASSIGN, BLOCK_SIZE_K_CHUNK=BLOCK_SIZE_K_CHUNK_ASSIGN, BLOCK_SIZE_D=BLOCK_SIZE_D_ASSIGN
        )
        assignments = assignments_int32.to(torch.int64)
        #update_start_time = time.time() # Optional timing
        new_sums = torch.zeros_like(centroids)
        cluster_counts = torch.zeros(K, dtype=torch.float32, device=device)
        idx_expand = assignments.unsqueeze(1).expand(N_A, D)
        new_sums.scatter_add_(dim=0, index=idx_expand, src=A_prep)
        cluster_counts.scatter_add_(dim=0, index=assignments, src=torch.ones_like(assignments, dtype=torch.float32))
        final_counts_safe = cluster_counts.clamp(min=1.0)
        new_centroids = new_sums / final_counts_safe.unsqueeze(1)
        empty_cluster_mask = (cluster_counts == 0)
        new_centroids[empty_cluster_mask] = centroids[empty_cluster_mask]
        #update_time = time.time() - update_start_time # Optional timing
        centroid_diff = torch.norm(new_centroids - centroids)
        centroids = new_centroids
        # print(f"  Iter {i+1}/{max_iters} | Centroid Diff: {centroid_diff:.4f} | Assign Time: {update_start_time - iter_start_time:.4f}s | Update Time: {update_time:.4f}s") # Optional verbose
        if centroid_diff < tol: break
    if i == max_iters - 1: print(f"KMeans reached max iterations ({max_iters}).")
    total_time = time.time() - start_time_total
    print(f"Total K-Means time: {total_time:.4f}s")
    return centroids, assignments
def distance_l2_squared_triton(X, A, **kwargs):
    """
    Computes pairwise SQUARED L2 distances using the tiled dot product kernel
    and PyTorch operations for norms. (No final sqrt)
    """
    target_device = X.device
    X_prep, A_prep = _prepare_tensors(X, A, target_device=target_device)
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    assert D == D_A, f"Dimension mismatch: X({D}) vs A({D_A})"

    dot_products = distance_dot_triton(X_prep, A_prep, **kwargs) # (Q, N)
    X_norm_sq = torch.sum(X_prep**2, axis=1, keepdims=True)  # (Q, 1)
    A_norm_sq = torch.sum(A_prep**2, axis=1, keepdims=True)  # (N, 1)
    # ||x - a||^2 = ||x||^2 - 2<x, a> + ||a||^2
    dist_sq = X_norm_sq - 2 * dot_products + A_norm_sq.T # (Q, N)
    dist_sq.clamp_(min=0.0) # Ensure non-negative due to potential float errors
    # NO SQRT HERE
    return dist_sq
def _prepare_tensors2(*tensors):
    """Ensure tensors are float32, contiguous, and on the correct device."""
    prepared = []
    # Assume 'device' is globally available or passed appropriately
    global device
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
DEFAULT_BLOCK_D = 128
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


# ============================================================================
# Task 2.2: Approximate Nearest Neighbors (Simplified HNSW)
# ============================================================================
# (Using the HNSW class provided in the previous version)
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


# --- our_ann function remains the same, calling this class ---
'''
def our_ann(N_A, D, A, X, K, M=16, ef_construction=100, ef_search=50):
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
     print(f"Index build time: {end_build - start_build:.2f} seconds")
     if hnsw_index.node_count == 0 or hnsw_index.entry_point == -1 : print("Error: Index build failed."); return torch.full((Q, K), -1), torch.full((Q, K), float('inf'))
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
     build_time = end_build - start_build
     search_time = end_search - start_search
     print(f"ANN search time: {end_search - start_search:.4f} seconds")
     return all_indices, all_distances, build_time, search_time # Returns indices and SQUARED L2 distances
'''
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
# Example Usage (Modified)
# ============================================================================
if __name__ == "__main__":
    N_data = 5000
    N_queries = 100
    Dim = 128
    K_val = 10

    print("="*40)
    print("Generating Data...")
    print("="*40)
    A_data = torch.randn(N_data, Dim, dtype=torch.float32, device=device)
    X_queries = torch.randn(N_queries, Dim, dtype=torch.float32, device=device)

    # --- Test Triton Dot Product ---
    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_dot_triton...")
    print("="*40)
    _ = distance_dot_triton(X_queries[:2], A_data[:5]) # Warm-up/autotune trigger small
    torch.cuda.synchronize()
    dot_dists = distance_dot_triton(X_queries, A_data) # Run on full data
    print("Dot distances shape:", dot_dists.shape)
    print("Sample (first 2x5):\n", dot_dists[:2,:5])
    end_time = time.time()
    print(f"Dot distance (Triton) computation time: {end_time - start_time:.4f} seconds")
    torch.cuda.synchronize()
    # Benchmark
    num_runs = 10
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for _ in range(num_runs): _ = distance_dot_triton(X_queries, A_data)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event) # Time in milliseconds
    avg_time_ms = elapsed_time_ms / num_runs
    avg_time_s = avg_time_ms / 1000.0
    print(f"Average Dot execution time {avg_time_s:.6f} seconds ({avg_time_ms:.4f} ms) seconds")
    print(dot_kernel_pairwise_tiled.best_config)
    
    # --- Test Triton L2 ---
    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_l2_triton...")
    print("="*40)
    l2_dists = distance_l2_triton(X_queries, A_data) # Run on full data
    print("L2 distances shape:", l2_dists.shape)
    print("Sample (first 2x5):\n", l2_dists[:2,:5])
    end_time = time.time()
    print(f"L2 distance (Triton) computation time: {end_time - start_time:.4f} seconds")
    # Benchmark L2
    torch.cuda.synchronize()
    num_runs = 10
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for _ in range(num_runs): _ = distance_l2_triton(X_queries, A_data)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event) # Time in milliseconds
    avg_time_ms = elapsed_time_ms / num_runs
    avg_time_s = avg_time_ms / 1000.0
    print(f"Average L2 execution time {avg_time_s:.6f} seconds ({avg_time_ms:.4f} ms) seconds")


    # --- Test Triton Cosine ---
    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_cosine_triton...")
    print("="*40)
    cos_dists = distance_cosine_triton(X_queries, A_data) # Run on full data
    print("Cosine distances shape:", cos_dists.shape)
    print("Sample (first 2x5):\n", cos_dists[:2,:5])
    end_time = time.time()
    torch.cuda.synchronize()
    print(f"Cosine distance (Triton) computation time: {end_time - start_time:.4f} seconds")
    # Benchmark Cosine
    num_runs = 10
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for _ in range(num_runs): _ = distance_cosine_triton(X_queries, A_data)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event) # Time in milliseconds
    avg_time_ms = elapsed_time_ms / num_runs
    avg_time_s = avg_time_ms / 1000.0
    print(f"Average Cosine execution time {avg_time_s:.6f} seconds ({avg_time_ms:.4f} ms) seconds")
    print("Best:", dot_kernel_pairwise_tiled.best_config)
    torch.cuda.synchronize()
    # --- Test NEW Tiled Triton Manhattan ---
    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_manhattan_triton...")
    print("="*40)
    # Run with fixed small blocks first to ensure it works without autotune potentially breaking it
    # Comment out the @triton.autotune above manhattan_kernel_pairwise_tiled when using fixed blocks
  #  print("Running Manhattan with fixed small blocks (16x16x16) for debugging...")
   # man_dists = distance_manhattan_triton(X_queries, A_data, BLOCK_Q=16, BLOCK_N=16, BLOCK_K=16)
    # If the above works, re-enable autotune on the kernel and run normally:
    print("Running Manhattan with autotune...")
    _ = distance_manhattan_triton(X_queries[:2], A_data[:5]) # Warm-up/autotune trigger small
    man_dists = distance_manhattan_triton(X_queries, A_data) # Run on full data
    torch.cuda.synchronize()
    print("Manhattan distances shape:", man_dists.shape)
    print("Sample (first 2x5):\n", man_dists[:2,:5])
    end_time = time.time()
    print(f"Manhattan distance (Tiled Triton) computation time: {end_time - start_time:.4f} seconds")
    # Benchmark
    print("Best",manhattan_kernel_pairwise_tiled.best_config)
    num_runs = 10
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    for _ in range(num_runs): _ = distance_manhattan_triton(X_queries, A_data)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event) # Time in milliseconds
    avg_time_ms = elapsed_time_ms / num_runs
    avg_time_s = avg_time_ms / 1000.0
    print(f"Average Manhattan execution time {avg_time_s:.6f} seconds ({avg_time_ms:.4f} ms) seconds")

    # --- Test k-NN ---
    print("\n" + "="*40)
    print(f"Testing our_knn (K={K_val}) using distance_l2_triton...")
    print("="*40)
    knn_indices, knn_dists = our_knn(N_data, Dim, A_data, X_queries, K_val)
    print("KNN results shape (Indices):", knn_indices.shape)
    print("KNN results shape (Distances - L2):", knn_dists.shape)
    print("Sample KNN Indices (Query 0):\n", knn_indices[0])
    print("Sample KNN Dists (Query 0):\n", knn_dists[0])

    # --- Test K-Means ---
    print("\n" + "="*40)
    print(f"Testing our_kmeans (K=5)...")
    print("="*40)
    K_clusters = 5
    kmeans_centroids, kmeans_assignments = our_kmeans(N_data, Dim, A_data, K_clusters)
    print("KMeans centroids shape:", kmeans_centroids.shape)
    print("KMeans assignments shape:", kmeans_assignments.shape)
    # print("Sample KMeans Assignments:\n", kmeans_assignments[:20]) # Optional print
    ids, counts = torch.unique(kmeans_assignments, return_counts=True)
    print("Cluster counts:")
    for id_val, count_val in zip(ids.tolist(), counts.tolist()): print(f"  Cluster {id_val}: {count_val}")
    print("\n" + "="*40)
    print(f"Starting ANN Hyperparameter Tuning (K={K_val})...")
    print("="*40)

    # Define parameter ranges to test
    # Adjust these lists based on how wide you want to search
    m_options = [32, 16, 64]
    efc_options = [100, 200, 400] # efConstruction
    efs_options = [50, 100, 200, 400, 800] # efSearch

    results_log = [] # To store results

    # --- Get KNN results for Query 0 once ---
    # Assumes knn_indices is already calculated from the KNN test block
    if N_queries > 0 and K_val > 0 and 'knn_indices' in locals():
        true_knn_ids_q0 = set(knn_indices[0].tolist())
    else:
        true_knn_ids_q0 = set() # Cannot calculate recall
    '''
    # --- Loop through hyperparameters ---
    for m_val in m_options:
        for efc_val in efc_options:
            for efs_val in efs_options:
                print(f"\n--- Testing M={m_val}, efC={efc_val}, efS={efs_val} ---")

                # Call the modified our_ann function
                ann_indices, ann_dists, build_time, search_time = our_ann(
                    N_data, Dim, A_data, X_queries, K_val,
                    M=m_val, ef_construction=efc_val, ef_search=efs_val
                )

                # Calculate recall for Query 0
                recall_q0 = float('nan') # Default if cannot calculate
                if N_queries > 0 and K_val > 0 and true_knn_ids_q0:
                    approx_ann_ids_q0 = set(ann_indices[0].tolist())
                    approx_ann_ids_q0.discard(-1) # Remove placeholders
                    recall_q0 = len(true_knn_ids_q0.intersection(approx_ann_ids_q0)) / K_val

                # Log results
                results_log.append({
                    'M': m_val, 'efC': efc_val, 'efS': efs_val,
                    'build_time': build_time, 'search_time': search_time, 'recall_q0': recall_q0
                })
                print(f"    Result: Build={build_time:.2f}s, Search={search_time:.4f}s, Recall@10(Q0)={recall_q0:.2%}")
                # Optional: Add a small delay if needed, e.g. time.sleep(1)
    
    # --- Print Summary ---
    print("\n" + "="*40)
    print("ANN Hyperparameter Test Summary:")
    print("="*40)
    # Sort results perhaps by recall or search time
    results_log.sort(key=lambda r: (r.get('recall_q0', 0), -r.get('search_time', float('inf'))), reverse=True) # Sort by recall (desc), then search time (asc)

    for res in results_log:
         print(f"M={res['M']:<3d}, efC={res['efC']:<4d}, efS={res['efS']:<4d} -> "
               f"Build={res['build_time']:.2f}s, Search={res['search_time']:.4f}s, "
               f"Recall(Q0)={res['recall_q0']:.2%}")
    '''
    # --- Test ANN (HNSW) ---
    print("\n" + "="*40)
    print(f"Testing our_ann (HNSW, K={K_val})...")
    print("="*40)
    ann_indices, ann_dists = our_ann(N_data, Dim, A_data, X_queries, K_val,
                             M=32,              # Keep M for now
                             ef_construction=200, # Keep efC for now
                             ef_search=100) 
    print("ANN results shape (Indices):", ann_indices.shape)
    print("ANN results shape (Distances - Squared L2):", ann_dists.shape) # HNSW returns Squared L2
    # print("Sample ANN Indices (Query 0):\n", ann_indices[0]) # Optional print
    # print("Sample ANN Dists (Query 0):\n", ann_dists[0]) # Optional print

    # --- Recall Calculation ---
    if N_queries > 0 and K_val > 0 and 'knn_indices' in locals() and 'ann_indices' in locals():
        total_intersect = 0
    # Loop through all queries
        for i in range(N_queries):
        # Get true KNN neighbors for query i
            true_knn_ids = set(knn_indices[i].cpu().tolist())
        # Get approximate ANN neighbors for query i
            approx_ann_ids = set(ann_indices[i].cpu().tolist())
        # Remove potential -1 placeholders from ANN results
            approx_ann_ids.discard(-1)
        # Add the number of overlapping neighbors to the total
            total_intersect += len(true_knn_ids.intersection(approx_ann_ids))

    # Calculate average recall across all queries
        if N_queries * K_val > 0:
            avg_recall = total_intersect / (N_queries * K_val)
            print(f"\nAverage ANN Recall @ {K_val} (vs brute-force KNN): {avg_recall:.2%}")
        else:
            print("\nCannot calculate average recall (N_queries or K_val is zero).")

    elif 'knn_indices' not in locals() or 'ann_indices' not in locals():
        print("\nCannot calculate recall: KNN or ANN results not available.")