import torch
import cupy as cp
import triton
import triton.language as tl
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
import csv
import os
import math

if not torch.cuda.is_available():
    print("CUDA not available, exiting.")
    exit()
device = torch.device("cuda:0")
print(f"Using device: {device}")

# -------------------------------
# Distance Functions Implementation
# -------------------------------

DEFAULT_BLOCK_Q = 32
DEFAULT_BLOCK_N = 64
DEFAULT_BLOCK_D = 512
DEFAULT_BLOCK_K = 16 # Block size for the reduction dimension D (used by all tiled kernels)

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
    accumulator = tl.zeros((BLOCK_Q, BLOCK_N), dtype=tl.float64)

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

    dot_prod = tl.zeros((), dtype=tl.float64)
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
def l2_dist_kernel_1_vs_M2(
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


def distance_dot(X, A):
    """Computes pairwise dot product using Triton kernel."""
    X_prep, A_prep = _prepare_tensors(X, A)
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    assert D == D_A, f"Dimension mismatch: X({D}) vs A({D_A})"

    Out = torch.empty((Q, N), dtype=torch.float64, device=device)
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
    return -Out

def distance_l2(X, A, **kwargs):
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

    dot_products = distance_dot(X_prep, A_prep, **kwargs) # (Q, N)
    X_norm_sq = torch.sum(X_prep**2, axis=1, keepdims=True)  # (Q, 1)
    A_norm_sq = torch.sum(A_prep**2, axis=1, keepdims=True)  # (N, 1)
    dist_sq = X_norm_sq - 2 * dot_products + A_norm_sq.T # (Q, N)
    dist_sq.clamp_(min=0.0)
    #dist = torch.sqrt(dist_sq)
    return dist_sq
'''
def distance_l2_triton2(X, A):
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
'''

def distance_cosine(X, A, epsilon=1e-8, **kwargs):
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

    dot_products = distance_dot(X_prep, A_prep, **kwargs) # (Q, N)
    X_norm = torch.linalg.norm(X_prep, axis=1, keepdims=True) # (Q, 1)
    A_norm = torch.linalg.norm(A_prep, axis=1, keepdims=True) # (N, 1)
    norm_product = X_norm * A_norm.T # (Q, N)
    cosine_similarity = dot_products / (norm_product + epsilon)
    cosine_similarity.clamp_(min=-1.0, max=1.0)
    cosine_distance = 1.0 - cosine_similarity
    return cosine_distance

def distance_manhattan(X, A, **kwargs):
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

def distance_dot3(X, Y):
    """
    Calculates pairwise dot products between rows of X and rows of Y.

    Args:
        X: cupy array of shape (Q, D) - Query points.
        Y: cupy array of shape (N, D) - Reference data points.

    Returns:
        cupy array of shape (Q, N) where element (i, j) is dot(X[i], Y[j]).
    """
    # Ensure inputs are CuPy arrays (good practice, though matmul might handle some cases)
    X_cp = cp.asarray(X)
    Y_cp = cp.asarray(Y)
    # X_cp shape: (Q, D), Y_cp shape: (N, D) -> Y_cp.T shape: (D, N)
    # Output shape: (Q, N)
    print(f"Calculating pairwise dot products via matmul for shapes: {X_cp.shape} and {Y_cp.T.shape}")
    return X_cp @ Y_cp.T

def distance_cosine2(X, Y):
    norm_X = cp.linalg.norm(X, axis=1) 
    norm_Y = cp.linalg.norm(Y, axis=1)
    cosine_similarity = cp.einsum('ij,ij->i', X, Y) / (norm_X * norm_Y)
    return 1 - cosine_similarity

def distance_l22(X, Y):
    return cp.linalg.norm(X - Y, axis=1)

def distance_dot2(X, Y):
    return cp.einsum('ij,ij->i', X, Y)

def distance_manhattan2(X, Y):
    return cp.sum(cp.abs(X - Y), axis=1)

def compute_all_distances(A, X):
    """
    Compute all four distances and return as a dictionary.
    """
    if X.ndim == 1:
        X = X[None, :]
    X =  cp.broadcast_to(X, A.shape)
    return {
        "Cosine": distance_cosine2(A, X),
        "L2": distance_l22(A, X),
        "Dot": distance_dot2(A, X),
        "Manhattan": distance_manhattan2(A, X)
    }
def distance_dot_tiled(X, A, N_TILE=50000, prep = True): # Tile size, adjust if needed
    """
    Computes pairwise dot product using Triton kernel, tiled over A
    to avoid exceeding GPU grid dimension limits.

    Args:
        X (torch.Tensor): Query vectors (Q, D) on GPU.
        A (torch.Tensor): Database vectors (N, D) on GPU.
        N_TILE (int): The maximum number of rows of A to process in one kernel launch.

    Returns:
        torch.Tensor: Output tensor of dot products (Q, N) on GPU.
    """
    if prep == True:
        X_prep, A_prep = _prepare_tensors(X, A) # Ensure tensors are ready
    else:
        X_prep, A_prep = X, A
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    assert D == D_A, f"Dimension mismatch: X({D}) vs A({D_A})"

    # Output tensor remains the full size
    Out = torch.empty((Q, N), dtype=torch.float32, device=device)

    print(f"Tiling dot product calculation with N_TILE={N_TILE}")

    for n_start in range(0, N, N_TILE):
        n_end = min(n_start + N_TILE, N)
        N_chunk = n_end - n_start # Size of the current chunk of A
        A_chunk = A_prep[n_start:n_end, :] # Shape (N_chunk, D)
        # Slice the relevant part of Out for this tile
        Out_chunk = Out[:, n_start:n_end]   # Shape (Q, N_chunk)

        grid = (Q, N_chunk)
        if grid[0] == 0 or grid[1] == 0: continue 

        dot_kernel_pairwise[grid](
            X_prep, A_chunk, Out_chunk,       # Data pointers for the chunks
            Q, N_chunk, D,                    # Dimensions for the chunk
            X_prep.stride(0), X_prep.stride(1),
            A_chunk.stride(0), A_chunk.stride(1), # Strides of A_chunk
            Out_chunk.stride(0), Out_chunk.stride(1),# Strides of Out_chunk
            BLOCK_SIZE_D=DEFAULT_BLOCK_D          # Kernel block size constant
        )
        # Potentially add torch.cuda.synchronize() here if debugging tile-by-tile issues

    return Out

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
    all_distances = distance_dot_tiled(X_prep, A_prep, prep = False) # Shape (Q, N_A)

    # 2. Find the top K smallest distances for each query
    #    largest=False gives smallest distances (nearest neighbors)
    topk_distances, topk_indices = torch.topk(all_distances, k=K, dim=1, largest=False)

    end_time = time.time()
    print(f"k-NN computation time: {end_time - start_time:.6f} seconds")

    return topk_indices, topk_distances

CURRENT_DISTANCE = "Dot"
def our_knn_hierachy(N, D, A, X, K):
    """
    KNN using hierarchical memory: Pinned memory enables fast CPU–GPU transfers.
    """
    if not isinstance(A, cp.ndarray):
        A = cp.asarray(A)
    if not isinstance(X, cp.ndarray):
        X = cp.asarray(X)
    
    distances = compute_all_distances(A, X)[CURRENT_DISTANCE]
    k_indices = cp.argpartition(distances, K)[:K]
    k_indices = k_indices[cp.argsort(distances[k_indices])]
    return k_indices

def our_knn_stream2(N, D, A, X, K):
    """
    KNN using CUDA Streams in CuPy for concurrent query processing.
    """
    B = X.shape[0] if X.ndim > 1 else 1  # Determine batch size.
    streams = [cp.cuda.Stream() for _ in range(B)]
    results = [None] * B

    for i in range(B):
        with streams[i]:
            query = X[i] if X.ndim > 1 else X
            distances = distance_dot2(A, query)[CURRENT_DISTANCE]
            k_indices = cp.argpartition(distances, K)[:K]
            results[i] = k_indices[cp.argsort(distances[k_indices])]
    for s in streams:
        s.synchronize()

    return results if B > 1 else results[0]


def our_knn_stream(N, D, A, X, K):
    """
    KNN using CUDA Streams in CuPy for concurrent query processing.
    MODIFIED: Uses distance_dot for similarity (finds K most similar neighbors).
    """
    # Ensure A and X are CuPy arrays
    if not isinstance(A, cp.ndarray):
        A = cp.asarray(A)
    if not isinstance(X, cp.ndarray):
        X = cp.asarray(X)

    # --- Input Validation (Optional but Recommended) ---
    if A.shape[0] != N or A.shape[1] != D:
         print(f"Warning: Dataset A shape {A.shape} does not match N={N}, D={D}")
         # Or raise ValueError("Dataset A shape mismatch")
    if X.ndim > 0 and X.shape[-1] != D:
         raise ValueError(f"Query X feature dimension {X.shape[-1]} does not match D={D}")
    # --- End Validation ---

    B = X.shape[0] if X.ndim > 1 else 1  # Determine batch size.
    if B == 0: return [] # Handle empty query batch

    streams = [cp.cuda.Stream() for _ in range(B)]
    results = [None] * B # To store indices for each query

    # Determine the actual number of neighbors to find (cannot exceed dataset size)
    actual_k = min(K, N)
    if actual_k <= 0: # Handles K<=0 or N=0
        empty_result = cp.array([], dtype=cp.int64)
        return [empty_result] * B if B > 1 else empty_result

    for i in range(B):
        # Assign work to the stream
        with streams[i]:
            # Extract the query, ensure it's 2D (1, D) for distance_dot
            query = X[i] if X.ndim > 1 else X
            # Reshape even if X was 1D, so query_2d is always (1, D)
            query_2d = query.reshape(1, D)

            # --- MODIFIED DISTANCE CALCULATION ---
            # Calculate raw dot products (similarities) using the specified function
            # Input shapes: query_2d (1, D), A (N, D)
            # Output shape: raw_dot_products (1, N)
            raw_dot_products = distance_dot3(query_2d, A)

            # We have scores for one query vs all dataset points, shape (1, N).
            # Get the 1D array of scores (shape N,)
            scores_1d = raw_dot_products[0]

            # --- MODIFIED K-NN LOGIC FOR SIMILARITY (Find K LARGEST scores) ---
            # Find the indices of the K largest scores (most similar neighbors)

            # Method 1: Partitioning (often faster for large N)
            # Partition to find the indices of the K largest (unsorted among themselves)
            # We need to partition around the (N - actual_k)-th element index
            k_indices_unsorted = cp.argpartition(scores_1d, N - actual_k)[-actual_k:]

            # Get the actual scores for these K indices if needed (e.g., for sorting)
            k_scores_unsorted = scores_1d[k_indices_unsorted]

            # Sort these K scores in descending order to get the final ranking
            # argsort on negative scores gives descending order indices relative to the K subset
            sorted_order_in_k = cp.argsort(-k_scores_unsorted)

            # Apply the sort order to the indices
            top_k_indices = k_indices_unsorted[sorted_order_in_k]

            # # Method 2: Sorting (simpler, maybe faster for small N)
            # sorted_indices_descending = cp.argsort(-scores_1d) # Sort all N scores descending
            # top_k_indices = sorted_indices_descending[:actual_k]

            # Store the final indices for this query
            results[i] = top_k_indices
            # --------------------------------------------------------------------

    # Wait for all streams to finish their work
    for s in streams:
        s.synchronize()

    # Return the list of results (or single result if B=1)
    return results if B > 1 else results[0]

if __name__ == "__main__":
    # --- Fixed Parameters ---
    N_data = 100000     # Reduced N for faster testing across dimensions
    N_queries = 1000    # Number of queries
    K_val = 10          # K for KNN
    NUM_RUNS = 10       # Number of timed runs for averaging
    WARMUP_RUNS = 2     # Number of warm-up runs

    # --- Dimensions to Test ---
    dimensions_to_test = [2, 4, 64, 256, 1024]

    print(f"--- GPU KNN/DISTANCE BENCHMARKING ---")
    print(f"Fixed Params: N={N_data}, Q={N_queries}, K={K_val}, Warmup={WARMUP_RUNS}, Runs={NUM_RUNS}")
    print(f"Testing Dimensions: {dimensions_to_test}")

    # --- Check Devices ---
    try:
        if not torch.cuda.is_available(): raise RuntimeError("Torch CUDA not available.")
        device = torch.device("cuda:0"); print(f"Using PyTorch device: {device}")
    except Exception as e: print(f"PyTorch device error: {e}"); exit()
    try:
        cp.cuda.Device(0).use(); print(f"Using CuPy device: {cp.cuda.Device(0)}")
        cupy_device_ok = True
    except Exception as e: print(f"CuPy device error: {e}"); cupy_device_ok = False
    # Exit if CuPy is needed but unavailable for parts of the benchmark
    # if not cupy_device_ok: print("CuPy device required but not available."); exit()

    # --- Storage for results across dimensions (optional) ---
    results = {}

    # Loop through each dimension
    for Dim in dimensions_to_test:
        print("\n" + "#"*70)
        print(f"# Starting Test for Dimension D = {Dim}")
        print("#"*70)

        results[Dim] = {} # Store results for this dimension

        # --- Generate Base Data (PyTorch and CuPy) ---
        print("\n" + "="*40); print(f"Generating Data (D={Dim})..."); print("="*40)
        try:
            # PyTorch Data
            A_data = torch.randn(N_data, Dim, dtype=torch.float32, device=device)
            X_queries = torch.randn(N_queries, Dim, dtype=torch.float32, device=device)
            torch.cuda.synchronize(device=device)
            print(f"Database shape (Torch): {A_data.shape}")
            print(f"Query shape (Torch): {X_queries.shape}")

            # CuPy Data (using DLPack for zero-copy transfer if possible)
            if cupy_device_ok:
                # Ensure PyTorch tensors are contiguous before DLPack export
                A_data_contig = A_data.contiguous()
                X_queries_contig = X_queries.contiguous()
                dlpack_A = torch.to_dlpack(A_data_contig)
                dlpack_X = torch.to_dlpack(X_queries_contig)
                A_data_cp = cp.from_dlpack(dlpack_A)
                X_queries_cp = cp.from_dlpack(dlpack_X)
                cp.cuda.Stream.null.synchronize()
                print(f"Database shape (CuPy): {A_data_cp.shape}")
                print(f"Query shape (CuPy): {X_queries_cp.shape}")
            else:
                A_data_cp, X_queries_cp = None, None
                print("CuPy data generation skipped.")

        except RuntimeError as e: # Catch CUDA OOM errors etc.
             if "CUDA out of memory" in str(e): print(f"Error: OOM Generating Data (D={Dim}). Skipping.")
             else: print(f"Runtime Error Generating Data (D={Dim}): {e}")
             if 'A_data' in locals(): del A_data
             if 'X_queries' in locals(): del X_queries
             if 'A_data_cp' in locals(): del A_data_cp
             if 'X_queries_cp' in locals(): del X_queries_cp
             torch.cuda.empty_cache(); cp.get_default_memory_pool().free_all_blocks()
             continue
        except Exception as e: print(f"Error generating data (D={Dim}): {e}"); continue

        # ===--------------------------------------------------===
        # ===              WARM-UP RUNS                      ===
        # ===--------------------------------------------------===
        print("\n" + "="*40); print(f"Performing Warm-up Runs (D={Dim})..."); print("="*40)
        try:
            for _ in range(WARMUP_RUNS):
                # --- PyTorch/Triton Warm-up ---
                # Distances (add try-except for each in case not defined)
                try: _ = distance_l2(X_queries, A_data)
                except NameError: pass
                try: _ = distance_cosine(X_queries, A_data)
                except NameError: pass
                try: _ = distance_manhattan(X_queries, A_data)
                except NameError: pass
                # KNN
                try: _ = our_knn(N_data, Dim, A_data, X_queries, K_val)
                except NameError: pass

                # --- CuPy Warm-up ---
                if cupy_device_ok and A_data_cp is not None:
                    # Distances
                    try: _ = distance_l22(X_queries_cp, A_data_cp)
                    except NameError: pass
                    try: _ = distance_cosine2(X_queries_cp, A_data_cp)
                    except NameError: pass
                    try: _ = distance_manhattan2(X_queries_cp, A_data_cp)
                    except NameError: pass
                    # KNNs
                    try: _ = our_knn_stream(N_data, Dim, A_data_cp, X_queries_cp, K_val)
                    except NameError: pass
                    try: _ = our_knn_hierachy(N_data, Dim, A_data_cp, X_queries_cp[0], K_val) # Warm-up one query
                    except NameError: pass

            torch.cuda.synchronize()
            if cupy_device_ok: cp.cuda.Stream.null.synchronize()
            print("Warm-up complete.")
        except Exception as e:
            print(f"Error during warm-up (D={Dim}): {e}")
            print("Skipping benchmarks for this dimension.")
            del A_data, X_queries; torch.cuda.empty_cache()
            if cupy_device_ok: del A_data_cp, X_queries_cp; cp.get_default_memory_pool().free_all_blocks()
            continue

        # ===--------------------------------------------------===
        # ===         DISTANCE FUNCTION BENCHMARKS           ===
        # ===--------------------------------------------------===
        print("\n" + "="*40); print(f"Benchmarking Distance Functions (D={Dim})..."); print("="*40)

        # --- PyTorch/Triton L2 Distance (distance_l2) ---
        try:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize() # Ensure previous work is done
            start_event.record()
            for r in range(NUM_RUNS):
                _ = distance_l2(X_queries, A_data) # Returns squared L2
            end_event.record()
            torch.cuda.synchronize()
            avg_time_l2 = (start_event.elapsed_time(end_event) / 1000.0) / NUM_RUNS # Avg time in seconds
            print(f"PyTorch/Triton distance_l2 Avg Time:   {avg_time_l2:.6f} seconds")
            results[Dim]['dist_l2_torch'] = avg_time_l2
        except NameError: print("distance_l2 not defined, skipping benchmark.")
        except Exception as e: print(f"Error benchmarking distance_l2: {e}")

        # --- PyTorch/Triton Cosine Distance (distance_cosine) ---
        try:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            for r in range(NUM_RUNS):
                _ = distance_cosine(X_queries, A_data) # Assuming exists
            end_event.record()
            torch.cuda.synchronize()
            avg_time_cos = (start_event.elapsed_time(end_event) / 1000.0) / NUM_RUNS
            print(f"PyTorch/Triton distance_cosine Avg Time: {avg_time_cos:.6f} seconds")
            results[Dim]['dist_cos_torch'] = avg_time_cos
        except NameError: print("distance_cosine not defined, skipping benchmark.")
        except Exception as e: print(f"Error benchmarking distance_cosine: {e}")

        # --- PyTorch/Triton Manhattan Distance (distance_manhattan) ---
        try:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            for r in range(NUM_RUNS):
                _ = distance_manhattan(X_queries, A_data) # Assuming exists
            end_event.record()
            torch.cuda.synchronize()
            avg_time_man = (start_event.elapsed_time(end_event) / 1000.0) / NUM_RUNS
            print(f"PyTorch/Triton distance_manhattan Avg Time: {avg_time_man:.6f} seconds")
            results[Dim]['dist_man_torch'] = avg_time_man
        except NameError: print("distance_manhattan not defined, skipping benchmark.")
        except Exception as e: print(f"Error benchmarking distance_manhattan: {e}")

        print("-" * 20) # Separator for CuPy distances

        # --- CuPy L2 Distance (distance_l22) ---
        if cupy_device_ok and A_data_cp is not None:
            try:
                start_event = cp.cuda.Event()
                end_event = cp.cuda.Event()
                cp.cuda.Stream.null.synchronize() # Ensure previous work is done
                start_event.record()
                for r in range(NUM_RUNS):
                    _ = distance_l22(X_queries_cp, A_data_cp) # Assuming exists & pairwise
                end_event.record()
                end_event.synchronize()
                avg_time_l2_cp = (cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0) / NUM_RUNS
                print(f"CuPy distance_l22 Avg Time:           {avg_time_l2_cp:.6f} seconds")
                results[Dim]['dist_l2_cupy'] = avg_time_l2_cp
            except NameError: print("distance_l22 not defined, skipping benchmark.")
            # Add Warning if function exists but likely wrong:
            except ValueError as e: print(f"ValueError benchmarking distance_l22 (check if pairwise): {e}")
            except Exception as e: print(f"Error benchmarking distance_l22: {e}")
        else: print("CuPy distance_l22 benchmark skipped (CuPy unavailable or data missing).")

        # --- CuPy Cosine Distance (distance_cosine2) ---
        if cupy_device_ok and A_data_cp is not None:
            try:
                start_event = cp.cuda.Event()
                end_event = cp.cuda.Event()
                cp.cuda.Stream.null.synchronize()
                start_event.record()
                for r in range(NUM_RUNS):
                    _ = distance_cosine2(X_queries_cp, A_data_cp) # Assuming exists & pairwise
                end_event.record()
                end_event.synchronize()
                avg_time_cos_cp = (cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0) / NUM_RUNS
                print(f"CuPy distance_cosine2 Avg Time:       {avg_time_cos_cp:.6f} seconds")
                results[Dim]['dist_cos_cupy'] = avg_time_cos_cp
            except NameError: print("distance_cosine2 not defined, skipping benchmark.")
            except ValueError as e: print(f"ValueError benchmarking distance_cosine2 (check if pairwise): {e}")
            except Exception as e: print(f"Error benchmarking distance_cosine2: {e}")
        else: print("CuPy distance_cosine2 benchmark skipped.")

        # --- CuPy Manhattan Distance (distance_manhattan2) ---
        if cupy_device_ok and A_data_cp is not None:
            try:
                start_event = cp.cuda.Event()
                end_event = cp.cuda.Event()
                cp.cuda.Stream.null.synchronize()
                start_event.record()
                for r in range(NUM_RUNS):
                    _ = distance_manhattan2(X_queries_cp, A_data_cp) # Assuming exists & pairwise
                end_event.record()
                end_event.synchronize()
                avg_time_man_cp = (cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0) / NUM_RUNS
                print(f"CuPy distance_manhattan2 Avg Time:    {avg_time_man_cp:.6f} seconds")
                results[Dim]['dist_man_cupy'] = avg_time_man_cp
            except NameError: print("distance_manhattan2 not defined, skipping benchmark.")
            except ValueError as e: print(f"ValueError benchmarking distance_manhattan2 (check if pairwise): {e}")
            except Exception as e: print(f"Error benchmarking distance_manhattan2: {e}")
        else: print("CuPy distance_manhattan2 benchmark skipped.")


        # ===--------------------------------------------------===
        # ===            KNN FUNCTION BENCHMARKS             ===
        # ===--------------------------------------------------===
        print("\n" + "="*40); print(f"Benchmarking KNN Functions (D={Dim})..."); print("="*40)

        # --- Triton/PyTorch KNN (our_knn) ---
        try:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            for r in range(NUM_RUNS):
                knn_indices_torch, _ = our_knn(N_data, Dim, A_data, X_queries, K_val)
            end_event.record()
            torch.cuda.synchronize()
            avg_time_knn_torch = (start_event.elapsed_time(end_event) / 1000.0) / NUM_RUNS
            qps_knn_torch = N_queries / avg_time_knn_torch if avg_time_knn_torch > 0 else 0
            print(f"Triton/Torch our_knn Avg Time:           {avg_time_knn_torch:.6f} seconds ({qps_knn_torch:.2f} QPS)")
            results[Dim]['knn_torch'] = avg_time_knn_torch
        except NameError: print("our_knn (Triton/Torch) not defined, skipping benchmark.")
        except Exception as e: print(f"Error benchmarking our_knn (Triton/Torch): {e}")

        # --- CuPy Hierarchy KNN (our_knn_hierachy - Looped) ---
        if cupy_device_ok and A_data_cp is not None:
            try:
                start_event = cp.cuda.Event()
                end_event = cp.cuda.Event()
                times = []
                # Time the whole loop including Python overhead NUM_RUNS times
                for r in range(NUM_RUNS):
                    all_knn_indices_hier = [None] * N_queries # Optional preallocation
                    cp.cuda.Stream.null.synchronize() # Sync before starting timer for this run
                    start_event.record()
                    for i in range(N_queries):
                        query_vector_cp = X_queries_cp[i] # Shape (D,)
                        knn_indices_cp_hier = our_knn_hierachy(N_data, Dim, A_data_cp, query_vector_cp, K_val)
                        # Storing results adds overhead, can comment out for pure timing
                        # all_knn_indices_hier[i] = knn_indices_cp_hier
                    end_event.record()
                    end_event.synchronize() # Wait for all GPU ops in the loop for this run
                    times.append(cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0)
                avg_time_hier = sum(times) / NUM_RUNS
                qps_hier = N_queries / avg_time_hier if avg_time_hier > 0 else 0
                print(f"CuPy our_knn_hierachy (Looped) Avg Time: {avg_time_hier:.6f} seconds ({qps_hier:.2f} QPS)")
                results[Dim]['knn_hier_cupy'] = avg_time_hier
            except NameError: print("our_knn_hierachy not defined, skipping benchmark.")
            except Exception as e: print(f"Error benchmarking our_knn_hierachy: {e}")
        else: print("CuPy our_knn_hierachy benchmark skipped.")

        # --- CuPy Stream KNN (our_knn_stream) ---
        if cupy_device_ok and A_data_cp is not None:
            try:
                start_event = cp.cuda.Event()
                end_event = cp.cuda.Event()
                times = []
                cp.cuda.Stream.null.synchronize() # Sync before starting timer
                start_event.record()
                for r in range(NUM_RUNS):
                    knn_indices_stream = our_knn_stream(N_data, Dim, A_data_cp, X_queries_cp, K_val)
                    # Assuming our_knn_stream synchronizes internally before returning
                end_event.record()
                end_event.synchronize() # Sync after all runs
                avg_time_stream = (cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0) / NUM_RUNS
                qps_stream = N_queries / avg_time_stream if avg_time_stream > 0 else 0
                print(f"CuPy our_knn_stream Avg Time:          {avg_time_stream:.6f} seconds ({qps_stream:.2f} QPS)")
                results[Dim]['knn_stream_cupy'] = avg_time_stream
            except NameError: print("our_knn_stream not defined, skipping benchmark.")
            except Exception as e: print(f"Error benchmarking our_knn_stream: {e}")
        else: print("CuPy our_knn_stream benchmark skipped.")


        print(f"\n--- Finished Benchmarks for Dimension D = {Dim} ---")
        # Clean up GPU memory
        del A_data, X_queries
        if cupy_device_ok and A_data_cp is not None: del A_data_cp, X_queries_cp
        torch.cuda.empty_cache()
        if cupy_device_ok: cp.get_default_memory_pool().free_all_blocks()


    print("\n" + "#"*70)
    print("# ALL DIMENSION BENCHMARKS FINISHED")
    print("#"*70)

    # --- Optional: Print Summary Table ---
    print("\nBenchmark Summary (Average Times in Seconds):")
    print("-" * 150) # Adjust width maybe
    header = f"{'Dim':<6}"
    # Dynamically build header based on actual results collected
    col_order = [ # Define preferred column order
        'dist_l2_torch', 'dist_cos_torch', 'dist_man_torch',
        'dist_l2_cupy', 'dist_cos_cupy', 'dist_man_cupy',
        'knn_torch', 'knn_hier_cupy', 'knn_stream_cupy'
    ]
    present_cols = set()
    for d in results:
        present_cols.update(results[d].keys())

    # Filter and order columns based on presence and preference
    final_cols = [col for col in col_order if col in present_cols]
    for col in sorted(present_cols): # Add any other columns found that weren't in col_order
        if col not in final_cols:
            final_cols.append(col)

    # Create header string
    for col_key in final_cols:
         header += f"{col_key:<25}" # Adjust spacing as needed
    print(header)
    print("-" * 150)

    for Dim in dimensions_to_test:
        if Dim in results:
            r = results[Dim]
            row = f"{Dim:<6}"
            for col_key in final_cols:
                 row += f"{r.get(col_key, -1.0):<25.6f}"
            print(row.replace('-1.000000', '        N/A            ')) # Basic formatting for missing results
        else:
            # Print dimension skipped row
            row = f"{Dim:<6}"
            for _ in final_cols: row += f"{'Skipped':<25}"
            print(row)
    print("-" * 150)