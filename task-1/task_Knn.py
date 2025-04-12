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
def distance_dot_tiled(X, A, N_TILE=32768, prep=True):
    # ... (definition returning positive dot product) ...
    if prep: X_prep, A_prep = _prepare_tensors(X, A)
    else: X_prep, A_prep = X, A
    Q, D = X_prep.shape; N, D_A = A_prep.shape
    if D != D_A: raise ValueError("Dimension mismatch")
    Out = torch.empty((Q, N), dtype=torch.float32, device=device)
    for n_start in range(0, N, N_TILE):
        n_end = min(n_start + N_TILE, N); N_chunk = n_end - n_start
        if N_chunk <= 0: continue
        A_chunk = A_prep[n_start:n_end, :]; Out_chunk = Out[:, n_start:n_end]
        grid = (Q, N_chunk)
        if grid[0] == 0 or grid[1] == 0: continue
        dot_kernel_pairwise[grid](
            X_prep, A_chunk, Out_chunk, Q, N_chunk, D,
            X_prep.stride(0), 1, A_chunk.stride(0), 1, Out_chunk.stride(0), 1,
            BLOCK_SIZE_D=DEFAULT_BLOCK_D)
    return Out

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
   # print(f"Calculating pairwise dot products via matmul for shapes: {X_cp.shape} and {Y_cp.T.shape}")
    return X_cp @ Y_cp.T

def pairwise_l2_squared_cupy(X_cp, C_cp):
    """ Computes pairwise SQUARED L2 distances using CuPy. """
    if not isinstance(X_cp, cp.ndarray): X_cp = cp.asarray(X_cp, dtype=cp.float32)
    elif X_cp.dtype != cp.float32: X_cp = X_cp.astype(cp.float32)
    if not isinstance(C_cp, cp.ndarray): C_cp = cp.asarray(C_cp, dtype=cp.float32)
    elif C_cp.dtype != cp.float32: C_cp = C_cp.astype(cp.float32)
    if X_cp.ndim == 1: X_cp = X_cp[None, :]
    if C_cp.ndim == 1: C_cp = C_cp[None, :]
    if X_cp.shape[1] != C_cp.shape[1]: raise ValueError("Dimension mismatch")

    X_norm_sq = cp.einsum('ij,ij->i', X_cp, X_cp)[:, cp.newaxis]
    C_norm_sq = cp.einsum('ij,ij->i', C_cp, C_cp)[cp.newaxis, :]
    # Use matmul for dot product: (Q, D) @ (D, N) -> (Q, N)
    dot_products = cp.matmul(X_cp, C_cp.T)
    dist_sq = X_norm_sq - 2 * dot_products + C_norm_sq
    return cp.maximum(0.0, dist_sq)

def distance_dot2(X, Y): # Corrected: Pairwise Dot Product
    """ Calculates pairwise dot products between rows of X and rows of Y using CuPy matmul. """
    X_cp = cp.asarray(X, dtype=cp.float32)
    Y_cp = cp.asarray(Y, dtype=cp.float32)
    if X_cp.ndim == 1: X_cp = X_cp[None, :] # Ensure X is 2D (Q, D)
    if Y_cp.ndim == 1: Y_cp = Y_cp[None, :] # Ensure Y is 2D (N, D)
    if X_cp.shape[1] != Y_cp.shape[1]: raise ValueError("Dimension mismatch for dot product")
    # print(f"CuPy Pairwise Dot: {X_cp.shape} @ {Y_cp.T.shape}")
    return cp.matmul(X_cp, Y_cp.T) # (Q, D) @ (D, N) -> (Q, N)

def distance_l22(X, Y): # Corrected: Pairwise SQUARED L2
    """ Calculates pairwise SQUARED L2 distance using CuPy. """
    # Reuse the optimized squared L2 logic
    return pairwise_l2_squared_cupy(X, Y)

def distance_cosine2(X, Y, epsilon=1e-8): # Corrected: Pairwise Cosine Distance
    """ Calculates pairwise cosine distance (1 - similarity) using CuPy. """
    X_cp = cp.asarray(X, dtype=np.float32)
    Y_cp = cp.asarray(Y, dtype=np.float32)
    if X_cp.ndim == 1: X_cp = X_cp[None, :]
    if Y_cp.ndim == 1: Y_cp = Y_cp[None, :]
    if X_cp.shape[1] != Y_cp.shape[1]: raise ValueError("Dimension mismatch")

    dot_products = distance_dot2(X_cp, Y_cp) # Pairwise dot (Q, N)
    norm_X = cp.linalg.norm(X_cp, axis=1, keepdims=True) # (Q, 1)
    norm_Y = cp.linalg.norm(Y_cp, axis=1, keepdims=True) # (N, 1)
    norm_product = (norm_X + epsilon) @ (norm_Y.T + epsilon) # (Q, N)
    cosine_similarity = dot_products / norm_product
    cosine_similarity = cp.clip(cosine_similarity, -1.0, 1.0)
    distance = cp.maximum(0.0, 1.0 - cosine_similarity)
    return distance

# Replace the old distance_manhattan2 function with this one:

def distance_manhattan2(X, Y, Q_TILE=256, N_TILE=256): # Tile sizes, can be tuned
    """
    Calculates pairwise Manhattan (L1) distance using CuPy with tiling
    to manage memory usage.
    """
    X_cp = cp.asarray(X, dtype=cp.float32)
    Y_cp = cp.asarray(Y, dtype=cp.float32)
    if X_cp.ndim == 1: X_cp = X_cp[None, :]
    if Y_cp.ndim == 1: Y_cp = Y_cp[None, :]

    Q, D = X_cp.shape
    N = Y_cp.shape[0]
    if D != Y_cp.shape[1]: raise ValueError(f"Dimension mismatch: X({D}) vs Y({N},{Y_cp.shape[1]})")
    if Q == 0 or N == 0: return cp.empty((Q,N), dtype=cp.float32) # Handle empty inputs

    # print(f"CuPy Pairwise Manhattan (Tiled): Shapes {X_cp.shape}, {Y_cp.shape}")
    l1_distance = cp.empty((Q, N), dtype=cp.float32)

    # Iterate through tiles of queries (Q) and database points (N)
    for q_start in range(0, Q, Q_TILE):
        q_end = min(q_start + Q_TILE, Q)
        X_chunk_q = X_cp[q_start:q_end] # Shape (curr_Q, D)
        curr_Q = X_chunk_q.shape[0]
        if curr_Q == 0: continue

        for n_start in range(0, N, N_TILE):
            n_end = min(n_start + N_TILE, N)
            Y_chunk_n = Y_cp[n_start:n_end] # Shape (curr_N, D)
            curr_N = Y_chunk_n.shape[0]
            if curr_N == 0: continue

            # Broadcast within the smaller tile: (curr_Q, 1, D) vs (1, curr_N, D)
            # Intermediate shape: (curr_Q, curr_N, D)
            # Peak memory reduced significantly, e.g. 256*256*D*4 bytes per tile
            try:
                abs_diff_tile = cp.abs(X_chunk_q[:, None, :] - Y_chunk_n[None, :, :])
                l1_distance_tile = cp.sum(abs_diff_tile, axis=2) # Shape (curr_Q, curr_N)
            except cp.cuda.memory.OutOfMemoryError:
                 print(f"\n--- OOM Error even within Manhattan tile (D={D}, Tile={curr_Q}x{curr_N}) ---")
                 print(f"--- Try reducing Q_TILE/N_TILE in distance_manhattan2 definition ---")
                 # Fill problematic tile with Inf and continue if possible, or re-raise
                 l1_distance[q_start:q_end, n_start:n_end] = cp.inf
                 cp.get_default_memory_pool().free_all_blocks() # Attempt cleanup
                 continue # Skip this tile, maybe others work

            # Store result in the output matrix slice
            l1_distance[q_start:q_end, n_start:n_end] = l1_distance_tile

            # Clean up intermediate tile explicitly (optional, helps memory management)
            del abs_diff_tile, l1_distance_tile
            # cp.get_default_memory_pool().free_all_blocks() # Can slow things down if called too often

    return l1_distance

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



# Corrected distance_l2 (Removes kwargs, ensures prep=False passed correctly)
def distance_l2(X, A): # Removed **kwargs
    """
    Computes pairwise SQUARED L2 distances using the tiled dot product kernel.
    """
    X_prep, A_prep = _prepare_tensors(X, A) # Prepare tensors internally
    Q, D = X_prep.shape; N, D_A = A_prep.shape
    if D != D_A: raise ValueError(f"Dimension mismatch: X({D}) vs A({D_A})")

    # Call distance_dot_tiled, tensors are prepared so prep=False
    # Do NOT pass **kwargs
    dot_products = distance_dot_tiled(X_prep, A_prep, prep=False) # Shape (Q, N)

    X_norm_sq = torch.sum(X_prep**2, axis=1, keepdims=True)  # Shape (Q, 1)
    A_norm_sq = torch.sum(A_prep**2, axis=1, keepdims=True)  # Shape (N, 1)
    dist_sq = X_norm_sq + A_norm_sq.T - 2 * dot_products # Shape (Q, N)
    dist_sq.clamp_(min=0.0)
    return dist_sq

# Corrected distance_cosine (Removes kwargs, calls corrected distance_dot_tiled)
def distance_cosine(X, A, epsilon=1e-8): # Removed **kwargs
    """
    Computes pairwise Cosine distances using the tiled dot product kernel.
    """
    target_device = X.device if isinstance(X, torch.Tensor) else A.device
    X_prep, A_prep = _prepare_tensors(X, A, target_device=target_device)
    Q, D = X_prep.shape; N, D_A = A_prep.shape
    if D != D_A: raise ValueError(f"Dimension mismatch: X({D}) vs A({D_A})")

    # Call distance_dot_tiled, tensors are prepared so prep=False
    # Do NOT pass **kwargs
    dot_products = distance_dot_tiled(X_prep, A_prep, prep=False) # (Q, N), POSITIVE dot

    X_norm = torch.linalg.norm(X_prep, axis=1, keepdims=True) # (Q, 1)
    A_norm = torch.linalg.norm(A_prep, axis=1, keepdims=True) # (N, 1)
    norm_product = (X_norm + epsilon) * (A_norm.T + epsilon) # Add epsilon before multiplication
    cosine_similarity = dot_products / norm_product
    cosine_similarity.clamp_(min=-1.0, max=1.0)
    cosine_distance = 1.0 - cosine_similarity
    return cosine_distance

# Corrected distance_manhattan (Removes kwargs, uses grid lambda for autotuned kernel)
def distance_manhattan(X, A):
    """
    Computes pairwise Manhattan (L1) distance using the tiled Triton kernel
    with FIXED block sizes (16, 16, 32).
    NOTE: Ensure the @triton.autotune decorator is removed/commented out
          from the 'manhattan_kernel_pairwise_tiled' kernel definition.
    """
    target_device = X.device if isinstance(X, torch.Tensor) else A.device # Get device from input
    X_prep, A_prep = _prepare_tensors(X, A, target_device=target_device) # Prepare tensors
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    if D != D_A: raise ValueError(f"Dimension mismatch: X({D}) vs A({D_A})")

    Out = torch.empty((Q, N), dtype=torch.float32, device=target_device)

    # --- Define Fixed Block Sizes ---
    BLOCK_Q_MAN = 16
    BLOCK_N_MAN = 16
    BLOCK_K_MAN = 32
    # --------------------------------

    # Calculate the launch grid based on fixed block sizes
    grid_man = (ceil_div(Q, BLOCK_Q_MAN), ceil_div(N, BLOCK_N_MAN))
    # print(f"Launching Manhattan kernel with fixed grid={grid_man}, Blocks=({BLOCK_Q_MAN},{BLOCK_N_MAN},{BLOCK_K_MAN})") # Optional debug

    # Launch the kernel, passing the grid and FIXED block sizes explicitly
    # Ensure manhattan_kernel_pairwise_tiled NO LONGER has the @triton.autotune decorator
    manhattan_kernel_pairwise_tiled[grid_man](
        X_prep, A_prep, Out,
        Q, N, D,
        X_prep.stride(0), 1, # Assuming contiguous tensor stride for D = 1
        A_prep.stride(0), 1, # Assuming contiguous tensor stride for D = 1
        Out.stride(0), 1,    # Assuming contiguous tensor stride for D = 1
        # Pass the block sizes explicitly matching the kernel signature
        BLOCK_Q=BLOCK_Q_MAN,
        BLOCK_N=BLOCK_N_MAN,
        BLOCK_K=BLOCK_K_MAN
    )
    torch.cuda.synchronize(device=target_device) # Sync after kernel call
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
def compute_all_distances_single_query(A_cp, X_query_cp):
    """
    Computes distances between all rows of database A_cp and a single query vector X_query_cp.
    Returns a dictionary of distance vectors (shape N,).
    """
    if not isinstance(A_cp, cp.ndarray): A_cp = cp.asarray(A_cp, dtype=cp.float32)
    if not isinstance(X_query_cp, cp.ndarray): X_query_cp = cp.asarray(X_query_cp, dtype=cp.float32)

    if X_query_cp.ndim == 1:
        X_query_2d = X_query_cp[None, :] # Reshape query to (1, D) for pairwise functions
    elif X_query_cp.shape[0] == 1 and X_query_cp.ndim == 2:
        X_query_2d = X_query_cp # Already (1, D)
    else:
        raise ValueError(f"X_query_cp must be a 1D vector or a (1, D) tensor, got shape {X_query_cp.shape}")

    if A_cp.shape[1] != X_query_2d.shape[1]:
        raise ValueError(f"Dimension mismatch between A ({A_cp.shape[1]}) and X ({X_query_2d.shape[1]})")

    N = A_cp.shape[0]
    dist_dict = {}

    # Call the PAIRWISE distance functions. Output shape will be (1, N).
    # Extract the first (and only) row to get the distance vector of shape (N,).
    try: dist_dict["Dot"] = distance_dot2(X_query_2d, A_cp)[0]
    except NameError: print("Warning: distance_dot2 not defined.")
    except Exception as e: print(f"Error in distance_dot2: {e}")

    try: dist_dict["L2"] = distance_l22(X_query_2d, A_cp)[0] # Assuming l22 returns squared L2
    except NameError: print("Warning: distance_l22 not defined.")
    except Exception as e: print(f"Error in distance_l22: {e}")

    try: dist_dict["Cosine"] = distance_cosine2(X_query_2d, A_cp)[0]
    except NameError: print("Warning: distance_cosine2 not defined.")
    except Exception as e: print(f"Error in distance_cosine2: {e}")

    try: dist_dict["Manhattan"] = distance_manhattan2(X_query_2d, A_cp)[0]
    except NameError: print("Warning: distance_manhattan2 not defined.")
    except Exception as e: print(f"Error in distance_manhattan2: {e}")

    return dist_dict

# Corrected KNN Hierarchy function
CURRENT_DISTANCE = "L2" # Default to L2, but can be changed globally if needed
def our_knn_hierachy(N, D, A, X_query, K): # X_query is a single query vector (D,)
    """
    KNN for a single query vector X_query against database A using CuPy.
    Computes distance between X_query and all vectors in A.
    """
    # Ensure CuPy arrays
    A_cp = cp.asarray(A)
    X_query_cp = cp.asarray(X_query)
    if X_query_cp.shape != (D,):
         # If called with (1,D) query, reshape to (D,)
         if X_query_cp.shape == (1,D):
             X_query_cp = X_query_cp[0]
         else:
             raise ValueError(f"our_knn_hierachy expects query of shape ({D},) or (1,{D}), got {X_query_cp.shape}")


    # Calculate distances from the single query to all points in A
    # using the corrected helper function
    all_distances = compute_all_distances_single_query(A_cp, X_query_cp) # Returns dict

    distances_1d = all_distances.get(CURRENT_DISTANCE) # Get distances for the chosen metric, shape (N,)

    if distances_1d is None:
        raise ValueError(f"CURRENT_DISTANCE='{CURRENT_DISTANCE}' not found or failed in compute_all_distances_single_query.")
    if distances_1d.shape[0] != N:
        raise ValueError(f"Distance calculation returned unexpected shape {distances_1d.shape}, expected ({N},)")


    actual_k = min(K, N)
    if actual_k <= 0: return cp.array([], dtype=cp.int64)

    # Find K smallest/largest depending on distance/similarity metric
    if CURRENT_DISTANCE == "Dot": # Higher score is better (similarity)
         # Find K largest scores
         k_partition = max(0, N - actual_k)
         if N == actual_k: k_indices_unsorted = cp.arange(N)
         else: k_indices_unsorted = cp.argpartition(distances_1d, kth=k_partition)[k_partition:]
         # Sort the top K descending by score
         k_indices = k_indices_unsorted[cp.argsort(-distances_1d[k_indices_unsorted])]
    else: # Assume lower score is better (L2, Cosine Dist, Manhattan)
         # Find K smallest distances
         k_partition = min(actual_k, N) # kth for np/cp argpartition is 0-based index
         if actual_k >= N: # If K >= N, just sort all
             k_indices = cp.argsort(distances_1d)
         else:
             # argpartition needs kth < N-1, but we want indices up to K. Let's use topk logic
             # k_indices_unsorted = cp.argpartition(distances_1d, kth=k_partition-1)[:actual_k] # Find K smallest indices (unsorted)
             # k_indices = k_indices_unsorted[cp.argsort(distances_1d[k_indices_unsorted])] # Sort the K smallest

             # Alternative using topk logic (often simpler than handling argpartition edge cases)
             # Find the K smallest distances and their indices directly
             # Note: CuPy currently lacks a direct equivalent of torch.topk
             # We have to sort all N distances, which is less efficient than partition for large N
             k_indices = cp.argsort(distances_1d)[:actual_k]


    return k_indices.astype(cp.int64) # Ensure int64 indices

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

# ============================================================================
# Main Execution Block (Benchmarking KNN and Distances across Dimensions)
# ============================================================================
if __name__ == "__main__":
    # --- Parameters ---
    N_data = 100000
    N_queries = 1000
    K_val = 10
    NUM_RUNS = 10
    WARMUP_RUNS = 2
    dimensions_to_test = [2, 4, 64, 256, 1024]
    # ... (Rest of parameter definitions) ...

    print(f"--- GPU KNN/DISTANCE BENCHMARKING ---")
    # ... (Device checks) ...
    try:
        # Attempt to initialize and use the default CuPy CUDA device
        cp.cuda.Device(0).use()
        print(f"Using CuPy device: {cp.cuda.Device(0)}")
        cupy_device_ok = True # Set to True if successful
    except Exception as e:
        # If any error occurs (CuPy not installed, CUDA issue, etc.)
        print(f"CuPy device error: {e}")
        cupy_device_ok = False # Set to False if initialization fails

    results = {}
    for Dim in dimensions_to_test:
        print("\n" + "#"*70)
        print(f"# Starting Test for Dimension D = {Dim}")
        print("#"*70)
        results[Dim] = {}

        # --- Generate Base Data ---
        print("\n" + "="*40); print(f"Generating Data (D={Dim})..."); print("="*40)
        A_data = A_data_cp = X_queries = X_queries_cp = None # Ensure cleanup
        try:
            # ... (Generate A_data, X_queries on Torch device) ...
            A_data = torch.randn(N_data, Dim, dtype=torch.float32, device=device)
            X_queries = torch.randn(N_queries, Dim, dtype=torch.float32, device=device)
            torch.cuda.synchronize(device=device)
            print(f"Database shape (Torch): {A_data.shape}")
            print(f"Query shape (Torch): {X_queries.shape}")

            if cupy_device_ok:
                # ... (DLPack transfer to A_data_cp, X_queries_cp) ...
                A_data_contig = A_data.contiguous(); X_queries_contig = X_queries.contiguous()
                dlpack_A = torch.to_dlpack(A_data_contig); dlpack_X = torch.to_dlpack(X_queries_contig)
                A_data_cp = cp.from_dlpack(dlpack_A); X_queries_cp = cp.from_dlpack(dlpack_X)
                cp.cuda.Stream.null.synchronize()
                print(f"Database shape (CuPy): {A_data_cp.shape}")
                print(f"Query shape (CuPy): {X_queries_cp.shape}")
            else: print("CuPy data generation skipped.")
        except Exception as e: # Handle OOM during generation
            print(f"Error during Data Generation (D={Dim}): {e}")
            if 'A_data' in locals() and A_data is not None: del A_data
            # ... (cleanup other vars) ...
            torch.cuda.empty_cache();
            if cupy_device_ok and 'A_data_cp' in locals() and A_data_cp is not None: cp.get_default_memory_pool().free_all_blocks()
            continue # Skip dimension

        # ===--------------------------------------------------===
        # ===              WARM-UP RUNS                      ===
        # ===--------------------------------------------------===
        print("\n" + "="*40); print(f"Performing Warm-up Runs (D={Dim})..."); print("="*40)
        try:
            # --- PyTorch/Triton Warm-up ---
            print("Warming up PyTorch/Triton functions...")
            torch_warmup_results = [] # Store results to delete later
            for _ in range(WARMUP_RUNS):
                try: torch_warmup_results.append(distance_l2(X_queries, A_data)); print("  Warmup distance_l2 OK");
                except NameError: pass
                except Exception as e_inner: print(f"  Warmup distance_l2 Error: {e_inner}")
                # ... (try/except for distance_cosine, distance_manhattan) ...
                try: torch_warmup_results.append(distance_cosine(X_queries, A_data)); print("  Warmup distance_cosine OK");
                except NameError: pass
                except Exception as e_inner: print(f"  Warmup distance_cosine Error: {e_inner}")
                try: torch_warmup_results.append(distance_manhattan(X_queries, A_data)); print("  Warmup distance_manhattan OK");
                except NameError: pass
                except Exception as e_inner: print(f"  Warmup distance_manhattan Error: {e_inner}")

                # --- Warm up our_knn ---
                try:
                    print("  Warming up our_knn...")
                    knn_indices_torch, knn_dists_torch = our_knn(N_data, Dim, A_data, X_queries, K_val)
                    torch_warmup_results.append(knn_indices_torch)
                    torch_warmup_results.append(knn_dists_torch)
                    print("  Warmup our_knn OK")
                except NameError: pass
                except Exception as e_inner: print(f"  Warmup our_knn Error: {e_inner}")
                # -----------------------

            # --- Explicitly Clear PyTorch Intermediate Results ---
            print("Clearing PyTorch warm-up results and emptying cache...")
            del torch_warmup_results # Remove references
            torch.cuda.synchronize() # Wait for GPU
            torch.cuda.empty_cache() # Ask PyTorch to release unused cached memory
            print("PyTorch Cache Cleared.")
            # ----------------------------------------------------

            # --- CuPy Warm-up ---
            if cupy_device_ok and A_data_cp is not None:
                print("Warming up CuPy functions...")
                cupy_warmup_results = []
                for _ in range(WARMUP_RUNS):
                    try: print("  Warming up distance_dot2..."); cupy_warmup_results.append(distance_dot2(X_queries_cp, A_data_cp)); print("  Warmup distance_dot2 OK");
                    except NameError: pass
                    except Exception as e_inner: print(f"  Warmup distance_dot2 Error: {e_inner}")
                    try: print("  Warming up distance_l22..."); cupy_warmup_results.append(distance_l22(X_queries_cp, A_data_cp)); print("  Warmup distance_l22 OK");
                    except NameError: pass
                    except Exception as e_inner: print(f"  Warmup distance_l22 Error: {e_inner}")
                    try: print("  Warming up distance_cosine2..."); cupy_warmup_results.append(distance_cosine2(X_queries_cp, A_data_cp)); print("  Warmup distance_cosine2 OK");
                    except NameError: pass
                    except Exception as e_inner: print(f"  Warmup distance_cosine2 Error: {e_inner}")
                    # Ensure the tiled Manhattan function is used here
                    try: print("  Warming up distance_manhattan2..."); cupy_warmup_results.append(distance_manhattan2(X_queries_cp, A_data_cp)); print("  Warmup distance_manhattan2 OK");
                    except NameError: pass
                    except Exception as e_inner: print(f"  Warmup distance_manhattan2 Error: {e_inner}")
                    try: print("  Warming up our_knn_stream..."); cupy_warmup_results.append(our_knn_stream(N_data, Dim, A_data_cp, X_queries_cp, K_val)); print("  Warmup our_knn_stream OK");
                    except NameError: pass
                    except Exception as e_inner: print(f"  Warmup our_knn_stream Error: {e_inner}")
                    try: print("  Warming up our_knn_hierachy (1 query)..."); cupy_warmup_results.append(our_knn_hierachy(N_data, Dim, A_data_cp, X_queries_cp[0], K_val)); print("  Warmup our_knn_hierachy OK");
                    except NameError: pass
                    except Exception as e_inner: print(f"  Warmup our_knn_hierachy Error: {e_inner}")

                # --- Explicitly Clear CuPy Intermediate Results ---
                print("Clearing CuPy warm-up results and emptying pool...")
                del cupy_warmup_results
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()
                print("CuPy Pool Cleared.")
                # --------------------------------------------------

            torch.cuda.synchronize()
            if cupy_device_ok: cp.cuda.Stream.null.synchronize()
            print("Warm-up complete.")

        except Exception as e: # Catch errors happening during warmup sequence
            print(f"Error during warm-up phase (D={Dim}): {e}") # Print the specific error caught
            print("Skipping benchmarks for this dimension.")
            del A_data, X_queries; torch.cuda.empty_cache()
            if cupy_device_ok: del A_data_cp, X_queries_cp; cp.get_default_memory_pool().free_all_blocks()
            continue


        # ===--------------------------------------------------===
        # ===         DISTANCE FUNCTION BENCHMARKS           ===
        # ===--------------------------------------------------===
        print("\n" + "="*40); print(f"Benchmarking Distance Functions (D={Dim})..."); print("="*40)

        # --- PyTorch/Triton Dot Distance (distance_dot_tiled) --- ADDED
        try:
            start_event = torch.cuda.Event(enable_timing=True); end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start_event.record()
            for r in range(NUM_RUNS): _ = distance_dot_tiled(X_queries, A_data)
            end_event.record(); torch.cuda.synchronize()
            avg_time = (start_event.elapsed_time(end_event) / 1000.0) / NUM_RUNS
            print(f"PyTorch/Triton distance_dot_tiled Avg Time: {avg_time:.6f} seconds")
            results[Dim]['dist_dot_torch'] = avg_time
        except NameError: print("distance_dot_tiled not defined, skipping benchmark.")
        except Exception as e: print(f"Error benchmarking distance_dot_tiled: {e}")

        # --- PyTorch/Triton L2 Distance (distance_l2) ---
        try:
            start_event = torch.cuda.Event(enable_timing=True); end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); start_event.record()
            for r in range(NUM_RUNS): _ = distance_l2(X_queries, A_data)
            end_event.record(); torch.cuda.synchronize()
            avg_time = (start_event.elapsed_time(end_event) / 1000.0) / NUM_RUNS
            print(f"PyTorch/Triton distance_l2 Avg Time:      {avg_time:.6f} seconds")
            results[Dim]['dist_l2_torch'] = avg_time
        except NameError: print("distance_l2 not defined, skipping benchmark.")
        except Exception as e: print(f"Error benchmarking distance_l2: {e}")

        # --- PyTorch/Triton Cosine Distance (distance_cosine) ---
        try:
            start_event = torch.cuda.Event(enable_timing=True); end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); start_event.record()
            for r in range(NUM_RUNS): _ = distance_cosine(X_queries, A_data) # Assuming exists
            end_event.record(); torch.cuda.synchronize()
            avg_time = (start_event.elapsed_time(end_event) / 1000.0) / NUM_RUNS
            print(f"PyTorch/Triton distance_cosine Avg Time:   {avg_time:.6f} seconds")
            results[Dim]['dist_cos_torch'] = avg_time
        except NameError: print("distance_cosine not defined, skipping benchmark.")
        except Exception as e: print(f"Error benchmarking distance_cosine: {e}")

        # --- PyTorch/Triton Manhattan Distance (distance_manhattan) ---
        try:
            start_event = torch.cuda.Event(enable_timing=True); end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); start_event.record()
            for r in range(NUM_RUNS): _ = distance_manhattan(X_queries, A_data) # Assuming exists
            end_event.record(); torch.cuda.synchronize()
            avg_time = (start_event.elapsed_time(end_event) / 1000.0) / NUM_RUNS
            print(f"PyTorch/Triton distance_manhattan Avg Time:{avg_time:.6f} seconds")
            results[Dim]['dist_man_torch'] = avg_time
        except NameError: print("distance_manhattan not defined, skipping benchmark.")
        except Exception as e: print(f"Error benchmarking distance_manhattan: {e}")

        print("-" * 25) # Separator for CuPy distances

        # --- CuPy Dot Distance (distance_dot2) --- ADDED
        if cupy_device_ok:
            try:
                start_event = cp.cuda.Event(); end_event = cp.cuda.Event()
                cp.cuda.Stream.null.synchronize()
                start_event.record()
                for r in range(NUM_RUNS): _ = distance_dot2(X_queries_cp, A_data_cp) # Uses corrected func
                end_event.record(); end_event.synchronize()
                avg_time = (cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0) / NUM_RUNS
                print(f"CuPy distance_dot2 Avg Time:            {avg_time:.6f} seconds")
                results[Dim]['dist_dot_cupy'] = avg_time
            except NameError: print("distance_dot2 not defined, skipping benchmark.")
            except Exception as e: print(f"Error benchmarking distance_dot2: {e}")
        else: print("CuPy distance_dot2 benchmark skipped.")

        # --- CuPy L2 Distance (distance_l22) ---
        if cupy_device_ok:
            try:
                start_event = cp.cuda.Event(); end_event = cp.cuda.Event()
                cp.cuda.Stream.null.synchronize(); start_event.record()
                for r in range(NUM_RUNS): _ = distance_l22(X_queries_cp, A_data_cp) # Uses corrected func
                end_event.record(); end_event.synchronize()
                avg_time = (cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0) / NUM_RUNS
                print(f"CuPy distance_l22 Avg Time:            {avg_time:.6f} seconds")
                results[Dim]['dist_l2_cupy'] = avg_time
            except NameError: print("distance_l22 not defined, skipping benchmark.")
            except Exception as e: print(f"Error benchmarking distance_l22: {e}")
        else: print("CuPy distance_l22 benchmark skipped.")

        # --- CuPy Cosine Distance (distance_cosine2) ---
        if cupy_device_ok:
            try:
                start_event = cp.cuda.Event(); end_event = cp.cuda.Event()
                cp.cuda.Stream.null.synchronize(); start_event.record()
                for r in range(NUM_RUNS): _ = distance_cosine2(X_queries_cp, A_data_cp) # Uses corrected func
                end_event.record(); end_event.synchronize()
                avg_time = (cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0) / NUM_RUNS
                print(f"CuPy distance_cosine2 Avg Time:       {avg_time:.6f} seconds")
                results[Dim]['dist_cos_cupy'] = avg_time
            except NameError: print("distance_cosine2 not defined, skipping benchmark.")
            except Exception as e: print(f"Error benchmarking distance_cosine2: {e}")
        else: print("CuPy distance_cosine2 benchmark skipped.")

        # --- CuPy Manhattan Distance (distance_manhattan2) ---
        if cupy_device_ok:
            try:
                start_event = cp.cuda.Event(); end_event = cp.cuda.Event()
                cp.cuda.Stream.null.synchronize(); start_event.record()
                for r in range(NUM_RUNS): _ = distance_manhattan2(X_queries_cp, A_data_cp) # Uses corrected func
                end_event.record(); end_event.synchronize()
                avg_time = (cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0) / NUM_RUNS
                # Check if result indicates OOM
                if cp.isinf(avg_time): print("CuPy distance_manhattan2 OOM occurred during benchmark runs.")
                else: print(f"CuPy distance_manhattan2 Avg Time:     {avg_time:.6f} seconds")
                results[Dim]['dist_man_cupy'] = avg_time
            except NameError: print("distance_manhattan2 not defined, skipping benchmark.")
            except Exception as e: print(f"Error benchmarking distance_manhattan2: {e}")
        else: print("CuPy distance_manhattan2 benchmark skipped.")


        # ===--------------------------------------------------===
        # ===            KNN FUNCTION BENCHMARKS             ===
        # ===--------------------------------------------------===
        print("\n" + "="*40); print(f"Benchmarking KNN Functions (D={Dim})..."); print("="*40)

        # --- Triton/PyTorch KNN (our_knn) ---
        try:
            start_event = torch.cuda.Event(enable_timing=True); end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); start_event.record()
            for r in range(NUM_RUNS): knn_indices_torch, _ = our_knn(N_data, Dim, A_data, X_queries, K_val)
            end_event.record(); torch.cuda.synchronize()
            avg_time = (start_event.elapsed_time(end_event) / 1000.0) / NUM_RUNS
            qps = N_queries / avg_time if avg_time > 0 else 0
            print(f"Triton/Torch our_knn Avg Time:           {avg_time:.6f} seconds ({qps:.2f} QPS)")
            results[Dim]['knn_torch'] = avg_time
        except NameError: print("our_knn (Triton/Torch) not defined, skipping benchmark.")
        except Exception as e: print(f"Error benchmarking our_knn (Triton/Torch): {e}")

        # --- CuPy Hierarchy KNN (our_knn_hierachy - Looped) ---
        if cupy_device_ok:
            try:
                start_event = cp.cuda.Event(); end_event = cp.cuda.Event()
                times = []
                for r in range(NUM_RUNS):
                    all_knn_indices_hier = [None] * N_queries
                    cp.cuda.Stream.null.synchronize(); start_event.record()
                    for i in range(N_queries):
                        query_vector_cp = X_queries_cp[i]
                        knn_indices_cp_hier = our_knn_hierachy(N_data, Dim, A_data_cp, query_vector_cp, K_val)
                        # all_knn_indices_hier[i] = knn_indices_cp_hier # Storing adds overhead
                    end_event.record(); end_event.synchronize()
                    times.append(cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0)
                avg_time = sum(times) / NUM_RUNS
                qps = N_queries / avg_time if avg_time > 0 else 0
                print(f"CuPy our_knn_hierachy (Looped) Avg Time: {avg_time:.6f} seconds ({qps:.2f} QPS)")
                results[Dim]['knn_hier_cupy'] = avg_time
            except NameError: print("our_knn_hierachy not defined, skipping benchmark.")
            except Exception as e: print(f"Error benchmarking our_knn_hierachy: {e}")
        else: print("CuPy our_knn_hierachy benchmark skipped.")

        # --- CuPy Stream KNN (our_knn_stream) ---
        if cupy_device_ok:
            try:
                start_event = cp.cuda.Event(); end_event = cp.cuda.Event()
                cp.cuda.Stream.null.synchronize(); start_event.record()
                for r in range(NUM_RUNS):
                    knn_indices_stream = our_knn_stream(N_data, Dim, A_data_cp, X_queries_cp, K_val)
                end_event.record(); end_event.synchronize()
                avg_time = (cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0) / NUM_RUNS
                qps = N_queries / avg_time if avg_time > 0 else 0
                print(f"CuPy our_knn_stream Avg Time:          {avg_time:.6f} seconds ({qps:.2f} QPS)")
                results[Dim]['knn_stream_cupy'] = avg_time
            except NameError: print("our_knn_stream not defined, skipping benchmark.")
            except Exception as e: print(f"Error benchmarking our_knn_stream: {e}")
        else: print("CuPy our_knn_stream benchmark skipped.")

        print(f"\n--- Finished Benchmarks for Dimension D = {Dim} ---")
        # Clean up GPU memory
        del A_data, X_queries
        if cupy_device_ok: del A_data_cp, X_queries_cp
        torch.cuda.empty_cache()
        if cupy_device_ok: cp.get_default_memory_pool().free_all_blocks()


    print("\n" + "#"*70); print("# ALL DIMENSION BENCHMARKS FINISHED"); print("#"*70)

    # --- Print Summary Table ---
    # ... (Summary table printing logic as before) ...

    print("\n" + "#"*70); print("# ALL DIMENSION BENCHMARKS FINISHED"); print("#"*70)

    # --- Print Summary Table ---
    print("\nBenchmark Summary (Average Times in Seconds):")
    # ... (Summary table printing logic as before) ...
    print("-" * 150) # Adjust width maybe
    header = f"{'Dim':<6}"
    col_order = [ # Define preferred column order
        'dist_dot_torch', 'dist_l2_torch', 'dist_cos_torch', 'dist_man_torch',
        'dist_dot_cupy', 'dist_l2_cupy', 'dist_cos_cupy', 'dist_man_cupy',
        'knn_torch', 'knn_hier_cupy', 'knn_stream_cupy'
    ]
    present_cols = set();
    for d in results: present_cols.update(results[d].keys())
    final_cols = [col for col in col_order if col in present_cols]
    for col in sorted(present_cols):
        if col not in final_cols: final_cols.append(col)
    for col_key in final_cols: header += f"{col_key:<25}"
    print(header); print("-" * 150)
    for Dim in dimensions_to_test:
        if Dim in results:
            r = results[Dim]; row = f"{Dim:<6}"
            for col_key in final_cols: row += f"{r.get(col_key, -1.0):<25.6f}"
            print(row.replace('-1.000000', '        N/A            '))
        else:
            row = f"{Dim:<6}";
            for _ in final_cols: row += f"{'Skipped':<25}"
            print(row)
    print("-" * 150)