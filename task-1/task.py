import torch
import triton
import triton.language as tl
import math
import heapq # For HNSW priority queues
import random
import time
import cupy as cp
# task-1/task.py
import torch
import triton
import triton.language as tl
import math
import heapq # For HNSW priority queues
import random
import time
# Removed: import cupy as cp

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
DEFAULT_BLOCK_K = 64
# Default block size for non-tiled kernels (Manhattan, older L2/Dot if kept)
DEFAULT_BLOCK_D = 128

def ceil_div(a, b):
    return (a + b - 1) // b

# --- Optimized Tiled Dot Product Kernel ---
@triton.autotune(
    configs=[
        # Basic configs with varying block sizes
        triton.Config({'BLOCK_Q': 16, 'BLOCK_N': 16, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 2}),
        triton.Config({'BLOCK_Q': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 32, 'BLOCK_N': 32, 'BLOCK_K': 64, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 1, 'num_warps': 4}),
        # Add num_stages variations
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 4}),
    ],
    key=['Q', 'N', 'D'], # Dimensions influencing performance
)
@triton.jit
def dot_kernel_pairwise_tiled(
    X_ptr, A_ptr, Out_ptr,           # Data pointers
    Q, N, D,                         # Matrix dimensions
    stride_xq, stride_xd,            # Strides for X (row, col)
    stride_an, stride_ad,            # Strides for A (row, col) - Note: A is (N, D)
    stride_outq, stride_outn,        # Strides for Out (row, col)
    BLOCK_Q: tl.constexpr,           # Tile size for Q dimension
    BLOCK_N: tl.constexpr,           # Tile size for N dimension
    BLOCK_K: tl.constexpr,           # Tile size for D dimension (often called K)
):
    """
    Calculates pairwise dot product using tiling: Out[q, n] = dot(X[q, :], A[n, :])
    This is equivalent to MatMul: Out = X @ A.T
    Each program instance computes a BLOCK_Q x BLOCK_N tile of the output.
    """
    # 1. Program ID and Offsets
    pid_q_block = tl.program_id(axis=0)  # Block index along Q dimension
    pid_n_block = tl.program_id(axis=1)  # Block index along N dimension

    # Calculate offsets for the current block this program will compute
    offs_q = pid_q_block * BLOCK_Q + tl.arange(0, BLOCK_Q) # Shape (BLOCK_Q,)
    offs_n = pid_n_block * BLOCK_N + tl.arange(0, BLOCK_N) # Shape (BLOCK_N,)

    # 2. Initialize Accumulator Tile
    accumulator = tl.zeros((BLOCK_Q, BLOCK_N), dtype=tl.float32)

    # 3. Loop over the Dimension D (K dimension) in blocks of BLOCK_K
    for k_start in range(0, D, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K) # Shape (BLOCK_K,)

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

        # --- Compute Matrix Multiplication for the tiles ---
        accumulator += tl.dot(x_tile, tl.trans(a_tile))

    # 4. Store the Resulting Tile
    out_ptrs = Out_ptr + (offs_q[:, None] * stride_outq + offs_n[None, :] * stride_outn)
    out_mask = (offs_q[:, None] < Q) & (offs_n[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=out_mask)
# --- NEW Optimized Tiled Manhattan (L1) Distance Kernel ---
@triton.autotune(
    configs=[
        # Similar configs to dot product, adjust if needed based on L1 characteristics
        triton.Config({'BLOCK_Q': 16, 'BLOCK_N': 16, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 2}),
        triton.Config({'BLOCK_Q': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 32, 'BLOCK_N': 32, 'BLOCK_K': 64, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 1, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 4}),
    ],
    key=['Q', 'N', 'D'], # Dimensions influencing performance
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
        # a_tile: (BLOCK_N, BLOCK_K) -> (1, BLOCK_N, BLOCK_K)
        # Difference: (BLOCK_Q, BLOCK_N, BLOCK_K)
        diff = x_tile[:, None, :] - a_tile[None, :, :]
        abs_diff = tl.abs(diff)

        # Sum absolute differences over the K dimension block (axis=2)
        # Result shape: (BLOCK_Q, BLOCK_N)
        accumulator += tl.sum(abs_diff, axis=2)

    # 4. Store the Resulting Tile
    out_ptrs = Out_ptr + (offs_q[:, None] * stride_outq + offs_n[None, :] * stride_outn)
    out_mask = (offs_q[:, None] < Q) & (offs_n[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=out_mask)



# --- Manhattan Distance Kernel (Simple, Non-Tiled) ---
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

# --- Kernels needed for K-Means and HNSW ---
@triton.jit
def kmeans_assign_kernel(
    A_ptr,           # Pointer to data points (N, D)
    centroids_ptr,   # Pointer to centroids (K, D)
    assignments_ptr, # Pointer to output assignments (N,)
    N, D, K,
    stride_an, stride_ad,
    stride_ck, stride_cd,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K_CHUNK: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Assigns each point in A to the nearest centroid (Squared L2) by iterating dimensions."""
    pid_n_block = tl.program_id(axis=0) # ID for the N dimension block
    offs_n = pid_n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) # Range for points in block
    mask_n = offs_n < N # Mask for points in this block

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
                    # Load centroid block
                    centroid_d_ptr = centroids_ptr + actual_k * stride_ck + offs_d * stride_cd
                    centroid_vals = tl.load(centroid_d_ptr, mask=mask_d, other=0.0)
                    # Load points block
                    points_d_ptr = A_ptr + offs_n[:, None] * stride_an + offs_d[None, :] * stride_ad
                    points_vals = tl.load(points_d_ptr, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
                    # Accumulate squared diff
                    diff = points_vals - centroid_vals[None, :]
                    current_dist_sq += tl.sum(diff * diff, axis=1)

                is_closer = current_dist_sq < min_dist_sq
                min_dist_sq = tl.where(is_closer, current_dist_sq, min_dist_sq)
                best_assignment = tl.where(is_closer, actual_k, best_assignment)

    assignments_out_ptrs = assignments_ptr + offs_n
    tl.store(assignments_out_ptrs, best_assignment, mask=mask_n)


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


# ============================================================================
# Helper Functions
# ============================================================================

def _prepare_tensors(*tensors, target_device):
    """Ensure tensors are float32, contiguous, and on the correct device."""
    prepared = []
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32, device=target_device)
        if t.device != target_device:
            t = t.to(target_device)
        if t.dtype != torch.float32:
            t = t.to(dtype=torch.float32)
        # Kernels often benefit from contiguous memory
        prepared.append(t.contiguous())
    return prepared
# --- NEW Launcher for Tiled Manhattan Distance ---
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

    manhattan_kernel_pairwise_tiled[grid](
        X_prep, A_prep, Out,
        Q, N, D,
        X_prep.stride(0), X_prep.stride(1), A_prep.stride(0), A_prep.stride(1),
        Out.stride(0), Out.stride(1),
        **kwargs # Pass autotuner kwargs like BLOCK_Q, BLOCK_N, BLOCK_K etc.
    )
    return Out

# ============================================================================
# Python Distance Function Wrappers using Triton / PyTorch
# ============================================================================

def distance_dot_triton(X, A, **kwargs):
    """Computes pairwise dot product (X @ A.T) using the tiled Triton kernel."""
    target_device = X.device # Assume X dictates the device
    X_prep, A_prep = _prepare_tensors(X, A, target_device=target_device)
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    assert D == D_A, f"Dimension mismatch: X({D}) vs A({D_A})"

    Out = torch.empty((Q, N), dtype=torch.float32, device=target_device)

    BLOCK_Q = kwargs.get('BLOCK_Q', DEFAULT_BLOCK_Q)
    BLOCK_N = kwargs.get('BLOCK_N', DEFAULT_BLOCK_N)
    grid = (ceil_div(Q, BLOCK_Q), ceil_div(N, BLOCK_N))
    print(f"Launching Triton Kernel dot_kernel_pairwise_tiled with grid={grid}")

    dot_kernel_pairwise_tiled[grid](
        X_prep, A_prep, Out,
        Q, N, D,
        X_prep.stride(0), X_prep.stride(1),
        A_prep.stride(0), A_prep.stride(1),
        Out.stride(0), Out.stride(1),
        **kwargs # Pass autotuner kwargs like BLOCK_Q, BLOCK_N, BLOCK_K etc.
    )
    return Out


def distance_l2_triton(X, A, **kwargs):
    """
    Computes pairwise L2 (Euclidean) distances using the tiled dot product kernel
    and PyTorch operations for norms.

    Args:
        X: Query vectors (Q, D) on GPU.
        A: Database vectors (N, D) on GPU.
        **kwargs: Forwarded to distance_dot_triton (for autotuning).

    Returns:
        torch.Tensor: Pairwise L2 distances (Q, N).
    """
    target_device = X.device
    X_prep, A_prep = _prepare_tensors(X, A, target_device=target_device)
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    assert D == D_A, f"Dimension mismatch: X({D}) vs A({D_A})"
    print(f"Calculating pairwise L2 (Triton Dot + PyTorch Norm) for shapes: {X_prep.shape} and {A_prep.shape}")

    # 1. Calculate pairwise dot products using the optimized Triton kernel
    dot_products = distance_dot_triton(X_prep, A_prep, **kwargs) # Shape (Q, N)

    # 2. Calculate squared L2 norms using efficient PyTorch operations
    X_norm_sq = torch.sum(X_prep**2, axis=1, keepdims=True)  # Shape (Q, 1)
    A_norm_sq = torch.sum(A_prep**2, axis=1, keepdims=True)  # Shape (N, 1)

    # 3. Calculate squared L2 distances using broadcasting:
    # ||X||^2 - 2*dot(X,A) + ||A||^2.T
    dist_sq = X_norm_sq - 2 * dot_products + A_norm_sq.T # Shapes (Q,1), (Q,N), (1,N) -> (Q,N)

    # 4. Clamp to non-negative and take square root for L2 distance
    dist_sq.clamp_(min=0.0) # In-place clamp
    dist = torch.sqrt(dist_sq)

    return dist # Shape (Q, N)


def distance_cosine_triton(X, A, epsilon=1e-8, **kwargs):
    """
    Computes pairwise Cosine distances using the tiled dot product kernel
    and PyTorch operations for norms.

    Args:
        X: Query vectors (Q, D) on GPU.
        A: Database vectors (N, D) on GPU.
        epsilon: Small value for numerical stability.
        **kwargs: Forwarded to distance_dot_triton (for autotuning).

    Returns:
        torch.Tensor: Pairwise Cosine distances (Q, N).
    """
    target_device = X.device
    X_prep, A_prep = _prepare_tensors(X, A, target_device=target_device)
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    assert D == D_A, f"Dimension mismatch: X({D}) vs A({D_A})"
    print(f"Calculating pairwise Cosine (Triton Dot + PyTorch Norm) for shapes: {X_prep.shape} and {A_prep.shape}")

    # 1. Calculate pairwise dot products using the optimized Triton kernel
    dot_products = distance_dot_triton(X_prep, A_prep, **kwargs) # Shape (Q, N)

    # 2. Calculate L2 norms using efficient PyTorch operations
    # keepdims=True is important for broadcasting
    X_norm = torch.linalg.norm(X_prep, axis=1, keepdims=True) # Shape (Q, 1)
    A_norm = torch.linalg.norm(A_prep, axis=1, keepdims=True) # Shape (N, 1)

    # 3. Calculate product of norms for denominator: ||X|| * ||A||.T
    # Using broadcasting: (Q, 1) * (1, N) -> (Q, N)
    norm_product = X_norm * A_norm.T

    # 4. Calculate cosine similarity
    cosine_similarity = dot_products / (norm_product + epsilon)

    # 5. Clamp similarity to valid range [-1, 1]
    cosine_similarity.clamp_(min=-1.0, max=1.0)

    # 6. Calculate Cosine distance = 1 - similarity
    cosine_distance = 1.0 - cosine_similarity

    return cosine_distance # Shape (Q, N)


def distance_manhattan(X, A):
    """Computes pairwise Manhattan (L1) distance using the simple Triton kernel."""
    target_device = X.device
    X_prep, A_prep = _prepare_tensors(X, A, target_device=target_device)
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    assert D == D_A, f"Dimension mismatch: X({D}) vs A({D_A})"
    print(f"Calculating pairwise Manhattan (Simple Triton Kernel) for shapes: {X_prep.shape} and {A_prep.shape}")

    Out = torch.empty((Q, N), dtype=torch.float32, device=device)
    grid = (Q, N) # Simple grid, one distance per program
    manhattan_dist_kernel_pairwise[grid](
        X_prep, A_prep, Out,
        Q, N, D,
        X_prep.stride(0), X_prep.stride(1),
        A_prep.stride(0), A_prep.stride(1),
        Out.stride(0), Out.stride(1),
        BLOCK_SIZE_D=DEFAULT_BLOCK_D # Use default block for inner loop
    )
    return Out

def distance_cosine2(X, Y, epsilon=1e-8):
   
    X_cp = cp.asarray(X)
    Y_cp = cp.asarray(Y)
    print(f"Calculating pairwise Cosine for shapes: {X_cp.shape} and {Y_cp.shape}")

    dot_products = X_cp @ Y_cp.T

    norm_X = cp.linalg.norm(X_cp, axis=1, keepdims=True)
    norm_Y = cp.linalg.norm(Y_cp, axis=1, keepdims=True)

    norm_product = norm_X @ norm_Y.T

    cosine_similarity = dot_products / (norm_product + epsilon)

    cosine_similarity = cp.clip(cosine_similarity, -1.0, 1.0)

    cosine_distance = 1.0 - cosine_similarity

    return cosine_distance
def distance_manhattan2(X, Y):
  
    X_cp = cp.asarray(X)
    Y_cp = cp.asarray(Y)
    print(f"Calculating pairwise Manhattan for shapes: {X_cp.shape} and {Y_cp.shape}")

    X_reshaped = X_cp[:, None, :]
    Y_reshaped = Y_cp[None, :, :]

    abs_diff = cp.abs(X_reshaped - Y_reshaped)

    manhattan_dists = cp.sum(abs_diff, axis=2)
    return manhattan_dists

def distance_l22(X, Y):
    X_cp = cp.asarray(X)
    Y_cp = cp.asarray(Y)
    print(f"Calculating pairwise L2 for shapes: {X_cp.shape} and {Y_cp.shape}")

    # Calculate squared L2 norms for each row. keepdims=True is important for broadcasting.
    X_norm_sq = cp.sum(X_cp**2, axis=1, keepdims=True)  # Shape (N, 1)
    Y_norm_sq = cp.sum(Y_cp**2, axis=1, keepdims=True)  # Shape (M, 1)

    # Calculate all pairwise dot products: X @ Y.T
    # Y_cp.T has shape (D, M)
    dot_products = X_cp @ Y_cp.T  # Shape (N, M)

    # Calculate squared L2 distances using broadcasting:
    # ||X||^2              (N, 1) broadcasts to (N, M)
    # - 2 * (X @ Y.T)     (N, M)
    # + ||Y||^2.T         (M, 1).T -> (1, M) broadcasts to (N, M)
    dist_sq = X_norm_sq - 2 * dot_products + Y_norm_sq.T

    # Handle potential small negative values due to floating point inaccuracies
    dist_sq = cp.maximum(0, dist_sq)

    # Return L2 distance (square root of squared distance)
    return cp.sqrt(dist_sq) # Shape (N, M)

def distance_dot2(X, Y):
    X_cp = cp.asarray(X)
    Y_cp = cp.asarray(Y)
    # X_cp shape: (N, D), Y_cp shape: (M, D)
    # Output shape: (N, M)
    print(f"Calculating pairwise dot products for shapes: {X_cp.shape} and {Y_cp.shape}")
    return X_cp @ Y_cp.T
# ============================================================================
# Task 1.2: k-Nearest Neighbors (Brute Force)
# ============================================================================

def our_knn(N_A, D, A, X, K):
    """
    Finds the K nearest neighbors in A for each query vector in X using
    brute-force pairwise L2 distance calculation (via distance_l2_triton).

    Args:
        N_A (int): Number of database points (should match A.shape[0]).
        D (int): Dimensionality (should match A.shape[1] and X.shape[1]).
        A (torch.Tensor): Database vectors (N_A, D) on GPU.
        X (torch.Tensor): Query vectors (Q, D) on GPU.
        K (int): Number of neighbors to find.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - topk_indices (torch.Tensor): Indices of the K nearest neighbors (Q, K).
            - topk_distances (torch.Tensor): L2 distances of the K nearest neighbors (Q, K).
    """
    target_device = X.device # Assume X dictates device
    A_prep, X_prep = _prepare_tensors(A, X, target_device=target_device)
    Q = X_prep.shape[0]
    assert A_prep.shape[0] == N_A, "N_A doesn't match A.shape[0]"
    assert A_prep.shape[1] == D, "D doesn't match A.shape[1]"
    assert X_prep.shape[1] == D, "D doesn't match X.shape[1]"
    assert K > 0, "K must be positive"
    assert K <= N_A, "K cannot be larger than the number of database points"

    print(f"Running k-NN: Q={Q}, N={N_A}, D={D}, K={K}")
    start_time = time.time()

    # 1. Calculate all pairwise L2 distances using the new Triton+PyTorch function
    all_distances = distance_l2_triton(X_prep, A_prep) # Shape (Q, N_A) -> Returns actual L2

    # 2. Find the top K smallest distances for each query
    topk_distances, topk_indices = torch.topk(all_distances, k=K, dim=1, largest=False)

    end_time = time.time()
    print(f"k-NN computation time: {end_time - start_time:.4f} seconds")

    # Returns actual L2 distances now
    return topk_indices, topk_distances


# ============================================================================
# Task 2.1: K-Means Clustering
# ============================================================================
# (Keeping the existing K-Means implementation using its specific Triton kernel)
def our_kmeans(N_A, D, A, K, max_iters=100, tol=1e-4):
    """
    Performs K-means clustering on data A using Triton kernel for assignment
    and PyTorch scatter_add_ for the update step.
    (Implementation unchanged from provided code)
    """
    target_device = A.device
    A_prep, = _prepare_tensors(A, target_device=target_device)
    assert A_prep.shape[0] == N_A, "N_A doesn't match A.shape[0]"
    assert A_prep.shape[1] == D, "D doesn't match A.shape[1]"
    assert K > 0, "K must be positive"
    assert K <= N_A, "K cannot be larger than the number of data points"

    print(f"Running K-Means (Update with PyTorch): N={N_A}, D={D}, K={K}")
    start_time_total = time.time()

    # --- Initialization ---
    initial_indices = torch.randperm(N_A, device=device)[:K]
    centroids = A_prep[initial_indices].clone()
    # assignments must be int64 for scatter_add_ index later
    assignments = torch.empty(N_A, dtype=torch.int64, device=device)

    # --- Triton Kernel Launch Configuration (Only for Assignment) ---
    BLOCK_SIZE_N_ASSIGN = 128
    BLOCK_SIZE_K_CHUNK_ASSIGN = 64
    BLOCK_SIZE_D_ASSIGN = DEFAULT_BLOCK_D # Use the default block D

    grid_assign = lambda meta: (triton.cdiv(N_A, meta['BLOCK_SIZE_N']),)

    for i in range(max_iters):
        iter_start_time = time.time()

        # --- 1. Assignment Step (Uses Triton Kernel) ---
        # Kernel expects int32 output, so create temp tensor
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
        # torch.cuda.synchronize() # Optional synchronization

        # --- 2. Update Step (Uses PyTorch scatter_add_) ---
        update_start_time = time.time()

        new_sums = torch.zeros_like(centroids) # Shape (K, D)
        cluster_counts = torch.zeros(K, dtype=torch.float32, device=device) # Shape (K,)
        idx_expand = assignments.unsqueeze(1).expand(N_A, D)
        new_sums.scatter_add_(dim=0, index=idx_expand, src=A_prep)
        cluster_counts.scatter_add_(dim=0, index=assignments, src=torch.ones_like(assignments, dtype=torch.float32))

        final_counts_safe = cluster_counts.clamp(min=1.0)
        new_centroids = new_sums / final_counts_safe.unsqueeze(1)
        empty_cluster_mask = (cluster_counts == 0)
        new_centroids[empty_cluster_mask] = centroids[empty_cluster_mask]
        update_time = time.time() - update_start_time

        # --- Check Convergence ---
        centroid_diff = torch.norm(new_centroids - centroids)
        centroids = new_centroids # Update centroids

        iter_end_time = time.time()
        # Reduce print frequency maybe?
        #if (i+1) % 10 == 0 or centroid_diff < tol or i == max_iters -1:
        print(f"  Iter {i+1}/{max_iters} | Centroid Diff: {centroid_diff:.4f} | Assign Time: {update_start_time - iter_start_time:.4f}s | Update Time: {update_time:.4f}s")

        if centroid_diff < tol:
            print(f"Converged after {i+1} iterations.")
            break

    if i == max_iters - 1:
        print(f"Reached max iterations ({max_iters}).")

    total_time = time.time() - start_time_total
    print(f"Total K-Means time: {total_time:.4f}s")

    return centroids, assignments

# ============================================================================
# Task 2.2: Approximate Nearest Neighbors (Simplified HNSW)
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
        self.BLOCK_SIZE_D_DIST = DEFAULT_BLOCK_D # Use default block D

    def _get_level_for_new_node(self):
        level = int(-math.log(random.uniform(0, 1)) * self.mL)
        return level

    def _distance(self, query_vec, candidate_indices):
        """Internal distance calc using the 1-vs-M Triton kernel (Squared L2)."""
        if not isinstance(candidate_indices, list): candidate_indices = list(candidate_indices)
        if not candidate_indices: return torch.empty(0, device=device), []

        num_candidates = len(candidate_indices)
        # Prepare query ONCE before loop/call
        query_vec_prep, = _prepare_tensors(query_vec.flatten(), target_device=query_vec.device)

        # Filter valid indices ONCE
        valid_indices = [idx for idx in candidate_indices if idx < self.node_count and idx >= 0]
        if not valid_indices:
             return torch.empty(0, device=device), []

        num_valid_candidates = len(valid_indices)
        # Indexing can be slow if done repeatedly. Ideally, pre-fetch or ensure locality if possible.
        # For now, standard indexing is used. Ensure self.vectors is contiguous if possible.
        candidate_vectors, = _prepare_tensors(self.vectors[valid_indices], target_device=query_vec.device)

        distances_out = torch.empty(num_valid_candidates, dtype=torch.float32, device=device)
        grid = (num_valid_candidates,)
        l2_dist_kernel_1_vs_M[grid]( # Uses simple 1-vs-M L2 kernel
            query_vec_prep, candidate_vectors, distances_out,
            num_valid_candidates, self.dim,
            candidate_vectors.stride(0), candidate_vectors.stride(1),
            BLOCK_SIZE_D=self.BLOCK_SIZE_D_DIST
        )
        return distances_out, valid_indices # Returns squared L2

    def _distance_batch(self, query_indices, candidate_indices):
        """
        Calculates pairwise **L2** distances between batches using distance_l2_triton.
        Returns a (len(query_indices), len(candidate_indices)) tensor of L2 distances.
        """
        if not query_indices or not candidate_indices:
            return torch.empty((0,0), device=device)

        # Ensure indices are valid and get vectors
        valid_query_indices = [idx for idx in query_indices if idx < self.node_count and idx >= 0]
        valid_candidate_indices = [idx for idx in candidate_indices if idx < self.node_count and idx >= 0]

        if not valid_query_indices or not valid_candidate_indices:
             return torch.empty((len(query_indices), len(candidate_indices)), device=device) # Return shape expected? Or (0,0)?

        query_vectors = self.vectors[valid_query_indices]
        candidate_vectors = self.vectors[valid_candidate_indices]

        # Use the optimized L2 distance function
        # Note: distance_l2_triton returns actual L2, not squared L2
        pairwise_l2_distances = distance_l2_triton(query_vectors, candidate_vectors)

        # If the original lists had invalid indices, we need to map results back or handle it.
        # For simplicity here, assuming the caller handles potential shape mismatches if indices were invalid.
        # A more robust implementation might return distances corresponding to original indices with inf for invalid ones.
        return pairwise_l2_distances # Shape (len(valid_query), len(valid_candidate))


    def _select_neighbors_heuristic(self, query_vec, candidates, M_target):
        """
        Selects M_target neighbors from candidates using a heuristic.
        Variant: Keep closest overall, prune candidates overshadowed by selected ones.
        Uses L2 distance (not squared).
        """
        selected_neighbors = []
        # Candidates are (squared_dist, node_id) from _search_layer which uses _distance
        # Convert to (L2_dist, node_id) for consistency if needed, or work with squared dists
        # Let's work with squared distances for selection comparisons, but store IDs.
        working_candidates_heap = [(dist, nid) for dist, nid in candidates] # Assume dist is squared L2
        heapq.heapify(working_candidates_heap)
        discarded_candidates = set()

        while working_candidates_heap and len(selected_neighbors) < M_target:
            dist_best_sq, best_nid = heapq.heappop(working_candidates_heap)
            if best_nid in discarded_candidates: continue

            selected_neighbors.append(best_nid)

            # --- Heuristic Pruning ---
            # Heuristic: If dist(candidate, best_nid) < dist(candidate, query), discard candidate.
            # Use L2 distances (or consistently squared L2)
            remaining_candidates_info = {} # {nid: dist_sq_to_query}
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
                # Calculate pairwise L2 distances between best_nid and remaining_nids
                # _distance_batch returns actual L2 distances
                dists_best_to_remaining = self._distance_batch([best_nid], remaining_nids) # Shape (1, len(remaining))
                if dists_best_to_remaining.numel() > 0:
                    dists_best_to_remaining_sq = dists_best_to_remaining.squeeze(0)**2 # Convert back to squared L2

                    for i, r_nid in enumerate(remaining_nids):
                        dist_r_query_sq = remaining_candidates_info[r_nid] # Already squared L2
                        dist_r_best_sq = dists_best_to_remaining_sq[i].item()

                        # Apply heuristic check (using squared distances is fine)
                        if dist_r_best_sq < dist_r_query_sq:
                            discarded_candidates.add(r_nid)
                # else: Handle case where _distance_batch returned empty

        return selected_neighbors


    # --- REVISED add_point METHOD ---
    def add_point(self, point_vec):
        """Adds a single point to the graph using heuristic neighbor selection."""
        target_device = self.vectors.device if self.node_count > 0 else device # Use existing device or default
        point_vec_prep, = _prepare_tensors(point_vec.flatten(), target_device=target_device)
        new_node_id = self.node_count

        # Append vector
        if self.node_count == 0:
             self.vectors = point_vec_prep.unsqueeze(0)
        else:
             self.vectors = torch.cat((self.vectors, point_vec_prep.unsqueeze(0)), dim=0)
        self.node_count += 1

        node_level = self._get_level_for_new_node()
        self.level_assignments.append(node_level)

        # Expand graph structure (CPU lists) - this remains a bottleneck
        while node_level >= len(self.graph): self.graph.append([])
        for lvl in range(len(self.graph)):
             while len(self.graph[lvl]) <= new_node_id: self.graph[lvl].append([])

        current_entry_point = self.entry_point
        current_max_level = self.max_level

        if current_entry_point == -1:
            self.entry_point = new_node_id
            self.max_level = node_level
            return new_node_id

        ep = [current_entry_point]
        for level in range(current_max_level, node_level, -1):
             if level >= len(self.graph) or ep[0] >= len(self.graph[level]): continue
             search_results = self._search_layer(point_vec_prep, ep, level, ef=1) # Use prepared vec
             if not search_results: break
             ep = [search_results[0][1]] # EP is node ID

        for level in range(min(node_level, current_max_level), -1, -1):
             if level >= len(self.graph) or not ep or any(idx >= len(self.graph[level]) for idx in ep):
                 if current_entry_point < len(self.graph[level]): ep = [current_entry_point]
                 else: continue

             # Search layer returns (squared_dist, node_id)
             neighbors_found_with_dist_sq = self._search_layer(point_vec_prep, ep, level, self.ef_construction) # Use prepared vec
             if not neighbors_found_with_dist_sq: continue

             # --- Use Heuristic Selection ---
             selected_neighbor_ids = self._select_neighbors_heuristic(point_vec_prep, neighbors_found_with_dist_sq, self.M) # Use prepared vec

             # --- Add Connections ---
             self.graph[level][new_node_id] = selected_neighbor_ids
             for neighbor_id in selected_neighbor_ids:
                 if neighbor_id >= len(self.graph[level]): continue
                 neighbor_connections = self.graph[level][neighbor_id]
                 if new_node_id not in neighbor_connections:
                     if len(neighbor_connections) < self.M:
                         neighbor_connections.append(new_node_id)
                     else:
                         # --- Pruning Logic --- (Uses squared L2 from _distance)
                         dist_new_sq, _ = self._distance(self.vectors[neighbor_id], [new_node_id])
                         if not dist_new_sq.numel(): continue
                         dist_new_sq = dist_new_sq[0].item()

                         current_neighbor_ids = list(neighbor_connections)
                         dists_to_current_sq, valid_curr_ids = self._distance(self.vectors[neighbor_id], current_neighbor_ids)

                         if dists_to_current_sq.numel() > 0:
                              furthest_dist_sq = -1.0
                              furthest_idx_in_list = -1
                              dist_map = {nid: d.item() for nid, d in zip(valid_curr_ids, dists_to_current_sq)}
                              for list_idx, current_nid in enumerate(current_neighbor_ids):
                                   d_sq = dist_map.get(current_nid, float('inf'))
                                   if d_sq > furthest_dist_sq:
                                        furthest_dist_sq = d_sq
                                        furthest_idx_in_list = list_idx
                              if furthest_idx_in_list != -1 and dist_new_sq < furthest_dist_sq:
                                   neighbor_connections[furthest_idx_in_list] = new_node_id
                         # Fallback not strictly needed if M constraint is hard

             # Update entry points for the next level down
             ep = selected_neighbor_ids
             if not ep: ep = [nid for _, nid in neighbors_found_with_dist_sq[:1]]

        if node_level > self.max_level:
            self.max_level = node_level
            self.entry_point = new_node_id
        return new_node_id

    def _search_layer(self, query_vec, entry_points, target_level, ef):
        """Performs greedy search on a single layer. Returns list of (squared_dist, node_id)."""
        # (Implementation remains largely the same as before, using self._distance which returns squared L2)
        if self.entry_point == -1: return []
        valid_entry_points = [ep for ep in entry_points if ep < self.node_count and ep >=0]
        if not valid_entry_points:
             if self.entry_point != -1: valid_entry_points = [self.entry_point]
             else: return []

        # _distance returns (squared_distances, valid_indices)
        initial_distances_sq, valid_indices_init = self._distance(query_vec, valid_entry_points)
        if initial_distances_sq.numel() == 0: return []

        dist_map_init = {nid: d.item() for nid, d in zip(valid_indices_init, initial_distances_sq)}
        candidate_heap = [(dist_map_init.get(ep, float('inf')), ep) for ep in valid_entry_points]
        heapq.heapify(candidate_heap)
        # Results heap stores (-squared_dist, node_id) for max-heap behavior on distance
        results_heap = [(-dist_sq, node_id) for dist_sq, node_id in candidate_heap if dist_sq != float('inf')]
        heapq.heapify(results_heap)
        visited = set(valid_entry_points)

        while candidate_heap:
            dist_candidate_sq, current_node_id = heapq.heappop(candidate_heap)
            if dist_candidate_sq == float('inf'): continue

            furthest_dist_sq = -results_heap[0][0] if results_heap else float('inf')
            if dist_candidate_sq > furthest_dist_sq and len(results_heap) >= ef: break

            try: neighbors = self.graph[target_level][current_node_id]
            except IndexError: neighbors = []

            unvisited_neighbors = [n for n in neighbors if n not in visited]
            if unvisited_neighbors:
                 visited.update(unvisited_neighbors)
                 # _distance returns (squared_distances, valid_indices)
                 neighbor_distances_sq, valid_neighbor_indices = self._distance(query_vec, unvisited_neighbors)
                 if neighbor_distances_sq.numel() == 0: continue

                 dist_map_neighbors = {nid: d.item() for nid, d in zip(valid_neighbor_indices, neighbor_distances_sq)}

                 for neighbor_id in valid_neighbor_indices:
                      neighbor_dist_sq_val = dist_map_neighbors[neighbor_id]
                      furthest_dist_sq = -results_heap[0][0] if results_heap else float('inf')

                      if len(results_heap) < ef or neighbor_dist_sq_val < furthest_dist_sq:
                           # Check if neighbor is already in results with a potentially worse distance (shouldn't happen with heap)
                           # Add to results heap
                           heapq.heappush(results_heap, (-neighbor_dist_sq_val, neighbor_id))
                           if len(results_heap) > ef: heapq.heappop(results_heap) # Keep heap size <= ef

                           # Add to candidate heap for exploration
                           # Optimization: only add if it could possibly lead to a better result?
                           # Basic approach: add if it's better than current furthest or heap not full
                           if len(results_heap) < ef or neighbor_dist_sq_val < (-results_heap[0][0] if results_heap else float('inf')):
                               heapq.heappush(candidate_heap, (neighbor_dist_sq_val, neighbor_id))

        # Return sorted list: (squared_dist, node_id)
        final_results = sorted([(abs(neg_dist_sq), node_id) for neg_dist_sq, node_id in results_heap])
        return final_results


    def search_knn(self, query_vec, k):
        """Searches for k nearest neighbors. Returns list of (squared_dist, node_id)."""
        # (Implementation remains largely the same, calls _search_layer)
        if self.entry_point == -1: return []
        query_vec_prep, = _prepare_tensors(query_vec.flatten(), target_device=query_vec.device)

        ep = [self.entry_point]
        current_max_level = self.max_level
        for level in range(current_max_level, 0, -1):
             if level >= len(self.graph) or ep[0] >= len(self.graph[level]): continue
             # search_results is list of (squared_dist, node_id)
             search_results = self._search_layer(query_vec_prep, ep, level, ef=1)
             if not search_results: break
             ep = [search_results[0][1]] # Use ID as next EP

        if 0 >= len(self.graph) or not ep or ep[0] >= len(self.graph[0]):
             if self.entry_point != -1 and 0 < len(self.graph) and self.entry_point < len(self.graph[0]):
                  ep = [self.entry_point]
             else:
                  return []

        # Final search at level 0 returns list of (squared_dist, node_id)
        neighbors_found = self._search_layer(query_vec_prep, ep, 0, self.ef_search)
        return neighbors_found[:k]


# --- our_ann function remains the same, calling this class ---
def our_ann(N_A, D, A, X, K, M=16, ef_construction=100, ef_search=50):
    target_device = X.device
    A_prep, X_prep = _prepare_tensors(A, X, target_device=target_device)
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
    build_batch_size = 1000 # Process in batches if N_A is huge? For now, sequential.
    for i in range(N_A):
        hnsw_index.add_point(A_prep[i])
        if (i + 1) % (N_A // 10 + 1) == 0:
             print(f"  Added {i+1}/{N_A} points...")
    end_build = time.time()
    print(f"Index build time: {end_build - start_build:.2f} seconds")

    if hnsw_index.node_count == 0 or hnsw_index.entry_point == -1 :
        print("Error: Index build resulted in an empty graph.")
        return torch.full((Q, K), -1, dtype=torch.int64), torch.full((Q, K), float('inf'), dtype=torch.float32)

    # 2. Perform Search for each query
    start_search = time.time()
    all_indices = torch.full((Q, K), -1, dtype=torch.int64, device=device)
    all_distances = torch.full((Q, K), float('inf'), dtype=torch.float32, device=device) # Store Squared L2

    print("Searching queries...")
    for q_idx in range(Q):
        # search_knn returns list of (squared_dist, node_id)
        results = hnsw_index.search_knn(X_prep[q_idx], K)
        num_results = len(results)
        if num_results > 0:
            # Distances are squared L2 from HNSW search
            q_dists_sq = torch.tensor([res[0] for res in results], dtype=torch.float32, device=device)
            q_indices = torch.tensor([res[1] for res in results], dtype=torch.int64, device=device)
            k_actual = min(num_results, K)
            all_distances[q_idx, :k_actual] = q_dists_sq[:k_actual] # Store squared L2
            all_indices[q_idx, :k_actual] = q_indices[:k_actual]
        if (q_idx + 1) % (Q // 10 + 1) == 0:
             print(f"  Searched {q_idx+1}/{Q} queries...")

    end_search = time.time()
    print(f"ANN search time: {end_search - start_search:.4f} seconds")

    # Return squared L2 distances consistent with HNSW internal calculations
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
    A_data = torch.randn(N_data, Dim, dtype=torch.float32, device=device)
    X_queries = torch.randn(N_queries, Dim, dtype=torch.float32, device=device)

    # --- Test Triton Dot Product ---
    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_dot_triton...")
    print("="*40)
    # Run once for autotuning cache if needed
    _ = distance_dot_triton(X_queries[:2], A_data[:5]) # Small run for syntax check / warm-up
    dot_dists = distance_dot_triton(X_queries, A_data)
    print("Dot distances shape:", dot_dists.shape)
    print("Sample (first 2x5):\n", dot_dists[:2,:5])
    end_time = time.time()
    print(f"Dot distance (Triton) computation time: {end_time - start_time:.4f} seconds")
    # Benchmark
    num_runs = 10
    start_bench_time = time.time()
    for _ in range(num_runs):
        _ = distance_dot_triton(X_queries, A_data)
    end_bench_time = time.time()
    avg_time = (end_bench_time - start_bench_time) / num_runs
    print(f"Average execution time ({num_runs} runs): {avg_time:.4f} seconds")
    print(dot_kernel_pairwise_tiled.best_config)

    # --- Test Triton L2 ---
    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_l2_triton...")
    print("="*40)
    l2_dists = distance_l2_triton(X_queries, A_data)
    print("L2 distances shape:", l2_dists.shape)
    print("Sample (first 2x5):\n", l2_dists[:2,:5])
    end_time = time.time()
    print(f"L2 distance (Triton) computation time: {end_time - start_time:.4f} seconds")
    num_runs = 10
    start_bench_time = time.time()
    for _ in range(num_runs):
        _ = distance_l2_triton(X_queries, A_data)
    end_bench_time = time.time()
    avg_time = (end_bench_time - start_bench_time) / num_runs
    print(f"Average execution time ({num_runs} runs): {avg_time:.4f} seconds")

    # --- Test Triton Cosine ---
    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_cosine_triton...")
    print("="*40)
    cos_dists = distance_cosine_triton(X_queries, A_data)
    print("Cosine distances shape:", cos_dists.shape)
    print("Sample (first 2x5):\n", cos_dists[:2,:5])
    end_time = time.time()
    print(f"Cosine distance (Triton) computation time: {end_time - start_time:.4f} seconds")
    num_runs = 10
    start_bench_time = time.time()
    for _ in range(num_runs):
        _ = distance_cosine_triton(X_queries, A_data)
    end_bench_time = time.time()
    avg_time = (end_bench_time - start_bench_time) / num_runs
    print(f"Average execution time ({num_runs} runs): {avg_time:.4f} seconds")

    # --- Test Triton Manhattan ---
    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_manhattan (Triton)...")
    print("="*40)
    man_dists = distance_manhattan_triton(X_queries, A_data)
    print("Manhattan distances shape:", man_dists.shape)
    print("Sample (first 2x5):\n", man_dists[:2,:5])
    end_time = time.time()
    print(f"Manhattan distance (Triton) computation time: {end_time - start_time:.4f} seconds")
    num_runs = 10
    start_bench_time = time.time()
    for _ in range(num_runs):
        _ = distance_manhattan_triton(X_queries, A_data)
    end_bench_time = time.time()
    avg_time = (end_bench_time - start_bench_time) / num_runs
    print(f"Average execution time ({num_runs} runs): {avg_time:.4f} seconds")

    dlpack_A = torch.to_dlpack(A_data)
    dlpack_X = torch.to_dlpack(X_queries)

# 2. Import DLPack capsule into CuPy array
    A_data_cp = cp.from_dlpack(dlpack_A)
    X_queries_cp = cp.from_dlpack(dlpack_X)
    print("CuPy testing....")
  
    start_time = time.time()

    print("\n" + "="*40)
    print("Testing distance_dot...")
    print("="*40)
    dot2_dists = distance_dot2(X_queries_cp, A_data_cp)
    print("Sample dot distances (squared) shape:", dot2_dists.shape)
    print(dot2_dists)
    end_time = time.time()
    print(f"Dot distance computation time: {end_time - start_time:.4f} seconds")


    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_l2...")
    print("="*40)
    l22_dists = distance_l22(X_queries_cp, A_data_cp)
    print("Sample L2 distances (squared) shape:", l22_dists.shape)
    print(l22_dists)
    end_time = time.time()
    print(f"L2 distance computation time: {end_time - start_time:.4f} seconds")


    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_cosine...")
    print("="*40)
    cos_dists2 = distance_cosine2(X_queries_cp, A_data_cp)
    print("Sample Cosine distances shape:", cos_dists2.shape)
    print(cos_dists2)
    end_time = time.time()
    print(f"Cosine distance computation time: {end_time - start_time:.4f} seconds")


    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_manhattan...")
    print("="*40)
    man_dists2 = distance_manhattan2(X_queries_cp, A_data_cp)
    print("Sample Manhattan distances shape:", man_dists2.shape)
    print(man_dists2)
    end_time = time.time()
    print(f"Manhattan distance computation time: {end_time - start_time:.4f} seconds")

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
    print("Sample KMeans Assignments:\n", kmeans_assignments[:20])
    ids, counts = torch.unique(kmeans_assignments, return_counts=True)
    print("Cluster counts:")
    for id_val, count_val in zip(ids.tolist(), counts.tolist()):
        print(f"  Cluster {id_val}: {count_val}")

    # --- Test ANN (HNSW) ---
    print("\n" + "="*40)
    print(f"Testing our_ann (HNSW, K={K_val})...")
    print("="*40)
    ann_indices, ann_dists = our_ann(N_data, Dim, A_data, X_queries, K_val,
                                 M=32, ef_construction=200, ef_search=100)
    print("ANN results shape (Indices):", ann_indices.shape)
    print("ANN results shape (Distances - Squared L2):", ann_dists.shape) # HNSW returns Squared L2
    print("Sample ANN Indices (Query 0):\n", ann_indices[0])
    print("Sample ANN Dists (Query 0):\n", ann_dists[0])

    # --- Recall Calculation ---
    if N_queries > 0 and K_val > 0:
        true_knn_ids = set(knn_indices[0].tolist())
        approx_ann_ids = set(ann_indices[0].tolist())
        approx_ann_ids.discard(-1)
        recall = len(true_knn_ids.intersection(approx_ann_ids)) / K_val
        print(f"\nANN Recall @ {K_val} for Query 0 (vs brute-force KNN): {recall:.2%}")
'''
def distance_cosine2(X, Y):
    norm_X = cp.linalg.norm(X, axis=1) 
    norm_Y = cp.linalg.norm(Y, axis=1)
    cosine_similarity = cp.einsum('ik,ik->ij', X, Y) / (norm_X * norm_Y)
    return 1 - cosine_similarity
'''


'''
def distance_manhattan2(X, Y):
    return cp.sum(cp.abs(X - Y), axis=1)
'''

'''
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
    dlpack_A = torch.to_dlpack(A_data)
    dlpack_X = torch.to_dlpack(X_queries)

# 2. Import DLPack capsule into CuPy array
    A_data_cp = cp.from_dlpack(dlpack_A)
    X_queries_cp = cp.from_dlpack(dlpack_X)

    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_dot...")
    print("="*40)
    dot_dists = distance_dot_triton(X_queries, A_data)
    print("Sample dot distances (squared) shape:", dot_dists.shape)
    print(dot_dists)
    end_time = time.time()
    print(f"Dot distance computation time: {end_time - start_time:.4f} seconds")
    num_runs = 10
    start_bench_time = time.time()
    for _ in range(num_runs):
        _ = distance_dot_triton(X_queries, A_data)
    end_bench_time = time.time()
    avg_time = (end_bench_time - start_bench_time) / num_runs
    print(f"Average execution time ({num_runs} runs): {avg_time:.4f} seconds")



    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_l2...")
    print("="*40)
    l2_dists = distance_l2(X_queries[:2], A_data[:5])
    print("Sample L2 distances (squared) shape:", l2_dists.shape)
    print(l2_dists)
    end_time = time.time()
    print(f"L2 distance computation time: {end_time - start_time:.4f} seconds")


    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_cosine...")
    print("="*40)
    cos_dists = distance_cosine(X_queries[:2], A_data[:5])
    print("Sample Cosine distances shape:", cos_dists.shape)
    print(cos_dists)
    end_time = time.time()
    print(f"Cosine distance computation time: {end_time - start_time:.4f} seconds")


    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_manhattan...")
    print("="*40)
    man_dists = distance_manhattan(X_queries[:2], A_data[:5])
    print("Sample Manhattan distances shape:", man_dists.shape)
    print(man_dists)
    end_time = time.time()
    print(f"Manhattan distance computation time: {end_time - start_time:.4f} seconds")


    start_time = time.time()

    print("\n" + "="*40)
    print("Testing distance_dot...")
    print("="*40)
    dot2_dists = distance_dot2(X_queries_cp[:2], A_data_cp[:5])
    print("Sample dot distances (squared) shape:", dot2_dists.shape)
    print(dot2_dists)
    end_time = time.time()
    print(f"Dot distance computation time: {end_time - start_time:.4f} seconds")


    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_l2...")
    print("="*40)
    l22_dists = distance_l22(X_queries_cp[:2], A_data_cp[:5])
    print("Sample L2 distances (squared) shape:", l22_dists.shape)
    print(l22_dists)
    end_time = time.time()
    print(f"L2 distance computation time: {end_time - start_time:.4f} seconds")


    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_cosine...")
    print("="*40)
    cos_dists2 = distance_cosine2(X_queries_cp[:2], A_data_cp[:5])
    print("Sample Cosine distances shape:", cos_dists2.shape)
    print(cos_dists2)
    end_time = time.time()
    print(f"Cosine distance computation time: {end_time - start_time:.4f} seconds")


    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_manhattan...")
    print("="*40)
    man_dists2 = distance_manhattan2(X_queries_cp[:2], A_data_cp[:5])
    print("Sample Manhattan distances shape:", man_dists2.shape)
    print(man_dists2)
    end_time = time.time()
    print(f"Manhattan distance computation time: {end_time - start_time:.4f} seconds")
'''

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