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
DEFAULT_BLOCK_D = 128
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
    return Out

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
    all_distances = distance_dot(X_prep, A_prep) # Shape (Q, N_A)

    # 2. Find the top K smallest distances for each query
    #    largest=False gives smallest distances (nearest neighbors)
    topk_distances, topk_indices = torch.topk(all_distances, k=K, dim=1, largest=False)

    end_time = time.time()
    print(f"k-NN computation time: {end_time - start_time:.4f} seconds")

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

def our_knn_stream(N, D, A, X, K):
    """
    KNN using CUDA Streams in CuPy for concurrent query processing.
    """
    B = X.shape[0] if X.ndim > 1 else 1  # Determine batch size.
    streams = [cp.cuda.Stream() for _ in range(B)]
    results = [None] * B

    for i in range(B):
        with streams[i]:
            query = X[i] if X.ndim > 1 else X
            distances = compute_all_distances(A, query)[CURRENT_DISTANCE]
            k_indices = cp.argpartition(distances, K)[:K]
            results[i] = k_indices[cp.argsort(distances[k_indices])]
    for s in streams:
        s.synchronize()
    
    return results if B > 1 else results[0]


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

    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_dot...")
    print("="*40)
    dot_dists = distance_dot(X_queries[:2], A_data[:5])
    end_time = time.time()
    print(f"Dot distance computation time: {end_time - start_time:.4f} seconds")
    print("Sample L2 distances (squared) shape:", dot_dists.shape)
    print(dot_dists)

    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_l2...")
    print("="*40)
    l2_dists = distance_l2(X_queries[:2], A_data[:5])
    end_time = time.time()
    print(f"L2 distance computation time: {end_time - start_time:.4f} seconds")
    print("Sample L2 distances (squared) shape:", l2_dists.shape)
    print(l2_dists)

    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_cosine...")
    print("="*40)
    cos_dists = distance_cosine(X_queries[:2], A_data[:5])
    end_time = time.time()
    print(f"Cosine distance computation time: {end_time - start_time:.4f} seconds")
    print("Sample Cosine distances shape:", cos_dists.shape)
    print(cos_dists)

    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_manhattan...")
    print("="*40)
    man_dists = distance_manhattan(X_queries[:2], A_data[:5])
    end_time = time.time()
    print(f"Manhattan distance computation time: {end_time - start_time:.4f} seconds")
    print("Sample Manhattan distances shape:", man_dists.shape)
    print(man_dists)

    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_dot...")
    print("="*40)
    dot_dists2 = distance_dot(X_queries, A_data)
    end_time = time.time()
    print(f"Dot distance computation time: {end_time - start_time:.4f} seconds")
    print("Sample L2 distances (squared) shape:", dot_dists2.shape)
    

    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_l2...")
    print("="*40)
    l2_dists2 = distance_l2(X_queries, A_data)
    end_time = time.time()
    print(f"L2 distance computation time: {end_time - start_time:.4f} seconds")
    print("Sample L2 distances (squared) shape:", l2_dists2.shape)
    

    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_cosine...")
    print("="*40)
    cos_dists2 = distance_cosine(X_queries, A_data)
    end_time = time.time()
    print(f"Cosine distance computation time: {end_time - start_time:.4f} seconds")
    print("Sample Cosine distances shape:", cos_dists2.shape)
   

    start_time = time.time()
    print("\n" + "="*40)
    print("Testing distance_manhattan...")
    print("="*40)
    man_dists2 = distance_manhattan(X_queries, A_data)
    end_time = time.time()
    print(f"Manhattan distance computation time: {end_time - start_time:.4f} seconds")
    print("Sample Manhattan distances shape:", man_dists2.shape)
   

    torch.cuda.synchronize()

    dlpack_A = torch.to_dlpack(A_data)
    dlpack_X = torch.to_dlpack(X_queries)

# 2. Import DLPack capsule into CuPy array
    A_data_cp = cp.from_dlpack(dlpack_A)
    X_queries_cp = cp.from_dlpack(dlpack_X)
    print("CuPy testing....")
    N_queries = X_queries_cp.shape[0]
    NUM_RUNS = 10
    all_knn_indices = [] # To store results for each query
    print("Performing warm-up runs...")
    try:
        _ = our_knn(N_data, Dim, A_data, X_queries, K_val)
        torch.cuda.synchronize()
        _ = our_knn_hierachy(N_data, Dim, A_data_cp, X_queries_cp[0], K_val) # Warm-up one query
        cp.cuda.Stream.null.synchronize()
        _ = our_knn_stream(N_data, Dim, A_data_cp, X_queries_cp, K_val)
        # our_knn_stream already synchronizes internally
        print("Warm-up complete.")
    except Exception as e:
        print(f"Error during warm-up: {e}")
        print("Skipping benchmarks.")
        exit()
    print("-" * 40)



    print("Processing queries individually...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for i in range(N_queries):
        query_vector_cp = X_queries_cp[i] # Get the i-th query vector (shape will be (D,))
        try:
        # Call the function with a single query vector
            knn_indices_cp = our_knn_hierachy(N_data, Dim, A_data_cp, query_vector_cp, K_val)
            all_knn_indices.append(cp.asnumpy(knn_indices_cp)) # Store result (optional: convert to numpy)
        # Optional: Add print statement for progress
        # if (i+1) % 10 == 0: print(f"Processed query {i+1}/{N_queries}")
        except Exception as e:
            print(f"Error processing query {i}: {e}")
            all_knn_indices.append(None) # Or handle error differently

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print("Hierarchy", elapsed_time_ms/ 1000.0, (elapsed_time_ms/ 1000.0)/N_queries)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    knn_indices = our_knn_stream(N_data, Dim, A_data_cp, X_queries_cp, K_val)         
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print("Triton",elapsed_time_ms/1000.0, (elapsed_time_ms/ 1000.0)/N_queries)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    knn_indices_cp,_ = our_knn(N_data, Dim, A_data, X_queries, K_val)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print("Triton",elapsed_time_ms/ 1000.0, (elapsed_time_ms/ 1000.0)/N_queries)
    print(f"Timing CuPy Hierarchy (our_knn_hierachy - Looped) over {NUM_RUNS} runs...")
    cupy_hier_times = []
    for r in range(NUM_RUNS):
        all_knn_indices_hier = []
        start_time = time.time()
        for i in range(N_queries):
            query_vector_cp = X_queries_cp[i]
            try:
                knn_indices_cp_hier = our_knn_hierachy(N_data, Dim, A_data_cp, query_vector_cp, K_val)
                # We don't synchronize inside the loop, only after all queries
                all_knn_indices_hier.append(knn_indices_cp_hier) # Keep result if needed
            except Exception as e:
                print(f"Error processing query {i} in hierarchy run {r+1}: {e}")
                all_knn_indices_hier.append(None)
        # --- IMPORTANT: Synchronize after the loop before stopping timer ---
        cp.cuda.Stream.null.synchronize()
        end_time = time.time()
        cupy_hier_times.append(end_time - start_time)
        # print(f" Run {r+1}/{NUM_RUNS} time: {cupy_hier_times[-1]:.4f}s") # Optional: print time per run
    cupy_hier_avg_time = sum(cupy_hier_times) / NUM_RUNS
    print(f"CuPy Hierarchy Avg Time: {cupy_hier_avg_time:.4f} seconds")
    print(f"CuPy Hierarchy Avg Time per Query: {cupy_hier_avg_time / N_queries:.6f} seconds")
    print("-" * 40)


    # ------------------------------------
    # Timed Benchmark: CuPy Streams (our_knn_stream)
    # ------------------------------------
    # Note: This method uses CuPy streams for potentially concurrent execution.
    print(f"Timing CuPy Streams (our_knn_stream) over {NUM_RUNS} runs...")
    cupy_stream_times = []
    for r in range(NUM_RUNS):
        start_time = time.time()
        # our_knn_stream handles internal synchronization
        knn_indices_stream = our_knn_stream(N_data, Dim, A_data_cp, X_queries_cp, K_val)
        # No explicit sync needed here as the function guarantees completion
        end_time = time.time()
        cupy_stream_times.append(end_time - start_time)
        # print(f" Run {r+1}/{NUM_RUNS} time: {cupy_stream_times[-1]:.4f}s") # Optional: print time per run
    cupy_stream_avg_time = sum(cupy_stream_times) / NUM_RUNS
    print(f"CuPy Streams Avg Time: {cupy_stream_avg_time:.4f} seconds")
    print(f"CuPy Streams Avg Time per Query: {cupy_stream_avg_time / N_queries:.6f} seconds")
    print("-" * 40)


    start_time = time.time()
    for i in range(N_queries):
        query_vector = X_queries[i] # Get the i-th query vector (shape will be (D,))
        try:
        # Call the function with a single query vector
            knn_indices,_ = our_knn(N_data, Dim, A_data, query_vector, K_val)
        # Optional: Add print statement for progress
        # if (i+1) % 10 == 0: print(f"Processed query {i+1}/{N_queries}")
        except Exception as e:
            print(f"Error processing query {i}: {e}")
            all_knn_indices.append(None) # Or handle error differently
    end_time = time.time()
    print("Triton", (end_time-start_time)/N_queries)

    print(f"Timing Triton Batched (our_knn) over {NUM_RUNS} runs...")
    triton_times = []
    for r in range(NUM_RUNS):
        start_time = time.time()
        knn_indices_torch, _ = our_knn(N_data, Dim, A_data, X_queries, K_val)
        # --- IMPORTANT: Synchronize before stopping timer ---
        torch.cuda.synchronize()
        end_time = time.time()
        triton_times.append(end_time - start_time)
        # print(f" Run {r+1}/{NUM_RUNS} time: {triton_times[-1]:.4f}s") # Optional: print time per run
    triton_avg_time = sum(triton_times) / NUM_RUNS
    print(f"Triton Batched Avg Time: {triton_avg_time:.4f} seconds")
    print(f"Triton Batched Avg Time per Query: {triton_avg_time / N_queries:.6f} seconds")
    print("-" * 40)

    

    print("Finished processing all queries.")
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

