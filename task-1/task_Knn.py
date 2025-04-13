import torch
import cupy as cp
import triton
import triton.language as tl
import numpy as np
import time
import scipy.spatial.distance
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
    stride_xq, stride_xd, # stride_xd/ad will be 1 from call site
    stride_an, stride_ad,
    stride_outq, stride_outn, # stride_outn will be 1 from call site
    # BLOCK_SIZE_D is now set by autotuner
    BLOCK_SIZE_D: tl.constexpr,
):
    """Calculates pairwise dot product using float32. Autotuned on BLOCK_SIZE_D."""
    pid_q = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    dot_prod = tl.zeros((), dtype=tl.float32) # Use float32

    for d_start in range(0, D, BLOCK_SIZE_D):
        offs_d = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offs_d < D
        # Use original pointer logic, assuming strides passed correctly reflect layout
        x_ptrs = X_ptr + pid_q * stride_xq + offs_d * stride_xd
        x_vals = tl.load(x_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        a_ptrs = A_ptr + pid_n * stride_an + offs_d * stride_ad
        a_vals = tl.load(a_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        dot_prod += tl.sum(x_vals * a_vals, axis=0) # Accumulate float32

    out_offset = pid_q * stride_outq + pid_n * stride_outn
    tl.store(Out_ptr + out_offset, dot_prod)

# @triton.autotune(...) # <-- Make sure this is commented out/removed!
@triton.jit
def dot_kernel_pairwise_tiled(
    X_ptr, A_ptr, Out_ptr,
    Q, N, D,
    stride_xq, stride_xd, stride_an, stride_ad, stride_outq, stride_outn,
    # These must be passed explicitly if autotune is off
    BLOCK_Q: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """ Calculates pairwise dot product using tiling and tl.dot. """
    pid_q_block = tl.program_id(axis=0); pid_n_block = tl.program_id(axis=1)
    offs_q = pid_q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_n = pid_n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    accumulator = tl.zeros((BLOCK_Q, BLOCK_N), dtype=tl.float32) # Use float32

    for k_start in range(0, D, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Load X tile (Shape: BLOCK_Q, BLOCK_K)
        x_ptrs = X_ptr + (offs_q[:, None] * stride_xq + offs_k[None, :] * 1) # Assume stride_xd=1
        q_mask = offs_q[:, None] < Q; k_mask_x = offs_k[None, :] < D; x_mask = q_mask & k_mask_x
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

        # Load A tile (Shape: BLOCK_N, BLOCK_K)
        a_ptrs = A_ptr + (offs_n[:, None] * stride_an + offs_k[None, :] * 1) # Assume stride_ad=1
        n_mask = offs_n[:, None] < N; k_mask_a = offs_k[None, :] < D; a_mask = n_mask & k_mask_a
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Compute dot product using tl.dot (expects A transposed)
        accumulator += tl.dot(x_tile, tl.trans(a_tile)) # tl.trans handles transpose

    # Store result tile
    out_ptrs = Out_ptr + (offs_q[:, None] * stride_outq + offs_n[None, :] * 1) # Assume stride_outn=1
    out_mask = (offs_q[:, None] < Q) & (offs_n[None, :] < N)
    tl.store(out_ptrs, accumulator, mask=out_mask)
@triton.jit
def manhattan_kernel_pairwise_simple(
    X_ptr,      # Pointer to Query vectors (Q, D)
    A_ptr,      # Pointer to Database vectors (N, D)
    Out_ptr,    # Pointer to output distances (Q, N)
    # --- Dimensions ---
    Q, N, D,
    # --- Strides ---
    stride_xq, stride_xd,
    stride_an, stride_ad,
    stride_outq, stride_outn,
    # --- Block Size for Looping over D ---
    BLOCK_SIZE_D: tl.constexpr, # Loop step size for the D dimension
):
    """
    Calculates pairwise Manhattan distance: dist(X[q], A[n]) = sum(abs(X[q,d] - A[n,d]))
    Each program instance computes ONE output element Out[q, n].
    Uses a simple loop over the D dimension.
    """
    pid_q = tl.program_id(axis=0) # Represents the query index q
    pid_n = tl.program_id(axis=1) # Represents the database index n (relative to the current chunk)

    l1_dist = tl.zeros((), dtype=tl.float32)

    for d_start in range(0, D, BLOCK_SIZE_D):
        offs_d = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offs_d < D

        x_ptrs = X_ptr + pid_q * stride_xq + offs_d * stride_xd
        x_vals = tl.load(x_ptrs, mask=mask_d, other=0.0)

        # A_ptr points to the START of the current A_chunk
        # pid_n is the index WITHIN the chunk (0 to N_chunk-1)
        a_ptrs = A_ptr + pid_n * stride_an + offs_d * stride_ad
        a_vals = tl.load(a_ptrs, mask=mask_d, other=0.0)

        diff = x_vals - a_vals
        l1_dist += tl.sum(tl.abs(diff), axis=0)

    # Out_ptr points to the START of the current Out_chunk
    out_offset = pid_q * stride_outq + pid_n * stride_outn
    tl.store(Out_ptr + out_offset, l1_dist)

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
# ============================================================================
# Helper Functions
# ============================================================================
def _prepare_tensors(*tensors, target_device=device):
    """
    Ensure tensors are float32, contiguous, and on the correct device.
    Returns single tensor if 1 input, list otherwise.
    """
    prepared = []
    for t in tensors:
        # --- Preparation logic ---
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
        # --- End preparation logic ---
        prepared.append(t) # Append the prepared tensor

    # --- Corrected Return Logic ---
    if len(prepared) == 1:
        return prepared[0] # Return single tensor directly
    else:
        return prepared    # Return list for multiple tensors
    # --- End Corrected Return Logic ---

# ============================================================================
# Python Distance Function Wrappers using Triton / PyTorch
# ============================================================================


def distance_dot(X, A):
    """Computes pairwise dot product using non-tiled Triton kernel (dot_kernel_pairwise)."""
    # _prepare_tensors ensures float32, contiguous inputs
    X_prep, A_prep = _prepare_tensors(X, A)
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    assert D == D_A, f"Dimension mismatch: X({D}) vs A({D_A})"

    # --- FIX 1: Change output dtype to float32 ---
    Out = torch.empty((Q, N), dtype=torch.float32, device=device)
    grid = (Q, N)

    # Ensure DEFAULT_BLOCK_D is defined (e.g., 128)
    global DEFAULT_BLOCK_D
    if 'DEFAULT_BLOCK_D' not in globals(): DEFAULT_BLOCK_D = 128 # Define if needed

    # Call the kernel (now expects float32 output pointer)
    dot_kernel_pairwise[grid](
        X_prep, A_prep, Out,
        Q, N, D,
        X_prep.stride(0), 1, # --- FIX 2: Use stride 1 for contiguous last dim ---
        A_prep.stride(0), 1, # --- FIX 2: Use stride 1 for contiguous last dim ---
        Out.stride(0),    1, # --- FIX 2: Use stride 1 for contiguous last dim ---
        BLOCK_SIZE_D=DEFAULT_BLOCK_D
    )
    # Return POSITIVE dot product
    return Out
# Assume ceil_div, _prepare_tensors are defined correctly
# Assume dot_kernel_pairwise kernel is defined correctly (float32, NO autotune)

def distance_dot_tiled(X, A, N_TILE=4096, prep=True):
    """
    Computes pairwise dot product using the simple 'dot_kernel_pairwise'
    kernel with FIXED BLOCK_SIZE_D=32 and num_warps=2.
    Launched in tiles over A (N dim). Passes actual output strides.
    """
    if prep:
         X_prep = _prepare_tensors(X)
         A_prep = _prepare_tensors(A)
    else: X_prep, A_prep = X, A

    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    if D != D_A: raise ValueError(f"Dimension mismatch: X({D}) vs A({D_A})")

    # Ensure Output tensor matches kernel expectation (float32)
    Out = torch.empty((Q, N), dtype=torch.float32, device=device)

    # --- Define Fixed Kernel Launch Parameters ---
    BLOCK_SIZE_D_FIXED = 32
    NUM_WARPS_FIXED = 2
    # -------------------------------------------

    # print(f"Tiling simple dot kernel (Fixed D={BLOCK_SIZE_D_FIXED}, Warps={NUM_WARPS_FIXED}) N_TILE={N_TILE}")
    for n_start in range(0, N, N_TILE):
        n_end = min(n_start + N_TILE, N)
        N_chunk = n_end - n_start
        if N_chunk <= 0: continue

        # It's crucial these slices are contiguous if stride=1 is assumed by kernel,
        # but _prepare_tensors ensures original tensors are contiguous.
        # Slicing along the first dim (A_chunk) preserves contiguity usually.
        # Slicing the output view might not.
        A_chunk = A_prep[n_start:n_end, :]
        Out_chunk = Out[:, n_start:n_end]

        grid = (Q, N_chunk)
        if grid[0] == 0 or grid[1] == 0: continue

        # --- Launch Kernel with ACTUAL strides for Out_chunk ---
        dot_kernel_pairwise[grid](
            X_prep, A_chunk, Out_chunk,
            Q, N_chunk, D,
            X_prep.stride(0), 1,              # Stride for X (stride_xq, stride_xd=1)
            A_chunk.stride(0), 1,             # Stride for A_chunk (stride_an, stride_ad=1)
            Out_chunk.stride(0), Out_chunk.stride(1), # Stride for Out_chunk (stride_outq, stride_outn)
            BLOCK_SIZE_D=BLOCK_SIZE_D_FIXED,
            num_warps=NUM_WARPS_FIXED
        )
        # -------------------------------------------------------

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
    """
    Computes pairwise **squared** L2 distances using CuPy.
    X_cp: (N, D) data points OR (Q, D) query points
    C_cp: (K, D) centroids OR (N, D) database points
    Returns: (N|Q, K|N) tensor of SQUARED distances.
    """
    # Ensure inputs are CuPy arrays and float32
    if not isinstance(X_cp, cp.ndarray): X_cp = cp.asarray(X_cp, dtype=cp.float32)
    elif X_cp.dtype != cp.float32: X_cp = X_cp.astype(cp.float32)
    if not isinstance(C_cp, cp.ndarray): C_cp = cp.asarray(C_cp, dtype=cp.float32)
    elif C_cp.dtype != cp.float32: C_cp = C_cp.astype(cp.float32)

    if X_cp.ndim == 1: X_cp = X_cp[cp.newaxis, :] # Ensure X is 2D
    if C_cp.ndim == 1: C_cp = C_cp[cp.newaxis, :] # Ensure C is 2D

    # Handle empty inputs gracefully
    if X_cp.shape[0] == 0 or C_cp.shape[0] == 0:
        return cp.empty((X_cp.shape[0], C_cp.shape[0]), dtype=cp.float32)

    # Check dimension compatibility AFTER ensuring 2D and non-empty
    if X_cp.shape[1] != C_cp.shape[1]:
        raise ValueError(f"Dimension mismatch: X_cp has D={X_cp.shape[1]}, C_cp has D={C_cp.shape[1]}")

    # ||x - c||^2 = ||x||^2 - 2<x, c> + ||c||^2 (optimized calculation)
    try:
        X_norm_sq = cp.einsum('ij,ij->i', X_cp, X_cp)[:, cp.newaxis] # Shape (N|Q, 1)
        C_norm_sq = cp.einsum('ij,ij->i', C_cp, C_cp)[cp.newaxis, :] # Shape (1, K|N)
        # Use gemm for dot product (generally fastest) via matmul
        dot_products = cp.matmul(X_cp, C_cp.T) # Shape (N|Q, K|N)

        # Broadcasting: (N|Q, 1) - 2*(N|Q, K|N) + (1, K|N) -> (N|Q, K|N)
        dist_sq = X_norm_sq - 2 * dot_products + C_norm_sq
        return cp.maximum(0.0, dist_sq) # Clamp numerical negatives

    except cp.cuda.memory.OutOfMemoryError as e:
        print(f"OOM Error in pairwise_l2_squared_cupy: Shapes X={X_cp.shape}, C={C_cp.shape}")
        # Estimate memory needed for intermediate products if helpful
        dot_prod_mem = X_cp.shape[0] * C_cp.shape[0] * 4 / (1024**3) # GB
        print(f"Estimated memory for dot product matrix: {dot_prod_mem:.2f} GB")
        raise e # Re-raise the exception
    except Exception as e:
        print(f"Error in pairwise_l2_squared_cupy: {e}")
        raise e

def distance_dot2(X, Y): # Corrected: Pairwise Dot Product
    """ Calculates pairwise dot products between rows of X and rows of Y using CuPy matmul. """
    X_cp = cp.asarray(X, dtype=cp.float32)
    Y_cp = cp.asarray(Y, dtype=cp.float32)
    if X_cp.ndim == 1: X_cp = X_cp[None, :] # Ensure X is 2D (Q, D)
    if Y_cp.ndim == 1: Y_cp = Y_cp[None, :] # Ensure Y is 2D (N, D)
    if X_cp.shape[1] != Y_cp.shape[1]: raise ValueError("Dimension mismatch for dot product")
    # print(f"CuPy Pairwise Dot: {X_cp.shape} @ {Y_cp.T.shape}")
    return cp.matmul(X_cp, Y_cp.T) # (Q, D) @ (D, N) -> (Q, N)

def distance_l223(X, Y):
    # --- Input Handling & Optimization ---
    X_cp = cp.asarray(X, dtype=cp.float32)
    Y_cp = cp.asarray(Y, dtype=cp.float32)
    if X_cp.ndim == 1: X_cp = X_cp[None, :]
    if Y_cp.ndim == 1: Y_cp = Y_cp[None, :]
    Q, D = X_cp.shape
    N, D_Y = Y_cp.shape
    if D != D_Y: raise ValueError(f"Dimension mismatch: X({D}) vs Y({D_Y})")
    if Q == 0 or N == 0: return cp.empty((Q, N), dtype=cp.float32)
    # --- Optimized Calculation ---
    X_norm_sq = cp.sum(cp.square(X_cp), axis=1, keepdims=True) # Shape: (Q, 1)
    Y_norm_sq = cp.sum(cp.square(Y_cp), axis=1)              # Shape: (N,)
    dot_prods = cp.matmul(X_cp, Y_cp.T)                     # Shape: (Q, N)
    dist_sq = X_norm_sq - 2 * dot_prods
    dist_sq += Y_norm_sq[None, :]                           # Shape: (Q, N)
    return cp.maximum(0.0, dist_sq)

# Add a new parameter for tiling the norm calculation
def distance_l22(X, Y, Q_TILE=30000, N_TILE=30000, N_NORM_TILE=100000): # Smaller tile for norm calc
    """
    Calculates pairwise SQUARED L2 distance using CuPy with tiling
    to manage memory for large inputs, INCLUDING tiled norm calculation.
    # ... (rest of docstring) ...
    """
    # --- Input Handling ---
    X_cp = cp.asarray(X, dtype=cp.float32)
    Y_cp = cp.asarray(Y, dtype=cp.float32)
    if X_cp.ndim == 1: X_cp = X_cp[None, :]
    if Y_cp.ndim == 1: Y_cp = Y_cp[None, :]

    Q, D = X_cp.shape
    N, D_Y = Y_cp.shape
    if D != D_Y:
        raise ValueError(f"Dimension mismatch: X({D}) vs Y({D_Y})")
    if Q == 0 or N == 0:
        return cp.empty((Q, N), dtype=cp.float32)

    # --- Initialization ---
    dist_sq_total = cp.empty((Q, N), dtype=cp.float32) # Output matrix

    # --- Precompute Norms (TILED) ---
    # Allocate space for the full norms, but compute them in chunks
    norm_Y_sq_full = cp.empty((N,), dtype=cp.float32)
    print(f"  Calculating Y norms tiled (N_NORM_TILE={N_NORM_TILE})...", flush=True)
    try:
        for n_norm_start in range(0, N, N_NORM_TILE):
            n_norm_end = min(n_norm_start + N_NORM_TILE, N)
            if n_norm_start == n_norm_end: continue

            # Process only a chunk of Y for norm calculation
            Y_chunk_norm = Y_cp[n_norm_start:n_norm_end]

            # This is now the peak memory usage inside the norm loop
            squared_chunk = cp.square(Y_chunk_norm)
            norm_Y_sq_chunk = cp.sum(squared_chunk, axis=1)

            # Store the result for this chunk
            norm_Y_sq_full[n_norm_start:n_norm_end] = norm_Y_sq_chunk

            # Optional: Clean up chunk memory explicitly if needed
            del Y_chunk_norm, squared_chunk, norm_Y_sq_chunk
            # cp.get_default_memory_pool().free_all_blocks() # Use if still getting OOM

        cp.cuda.Stream.null.synchronize() # Ensure calculation is done before proceeding
        print("  Y norms calculation complete.", flush=True)

    except cp.cuda.memory.OutOfMemoryError as e:
        print(f"\n--- OOM Error during TILED NORM calculation ---")
        print(f"--- GPU likely lacks memory even for norm tile size {N_NORM_TILE}x{D}")
        print(f"--- Consider reducing N_NORM_TILE further or using a GPU with more memory.")
        print(f"--- Error: {e}")
        raise # Re-raise the error to stop the benchmark for this dimension

    # --- Tiled Computation (Main part - mostly unchanged) ---
    print(f"  Starting main tiled distance calculation (Q_TILE={Q_TILE}, N_TILE={N_TILE})...", flush=True)
    for q_start in range(0, Q, Q_TILE):
        q_end = min(q_start + Q_TILE, Q)
        X_chunk_q = X_cp[q_start:q_end]
        curr_Q = X_chunk_q.shape[0]
        if curr_Q == 0: continue

        # Calculate X norm chunk (usually small unless Q_TILE is huge)
        try:
             norm_X_sq_chunk = cp.sum(cp.square(X_chunk_q), axis=1, keepdims=True)
        except cp.cuda.memory.OutOfMemoryError as e:
            print(f"\n--- OOM Error during X NORM calculation ---")
            print(f"--- Tile size Q_TILE={Q_TILE} might be too large for D={D}")
            print(f"--- Error: {e}")
            # Clean up Y norms before raising
            del norm_Y_sq_full
            cp.get_default_memory_pool().free_all_blocks()
            raise

        for n_start in range(0, N, N_TILE):
            n_end = min(n_start + N_TILE, N)
            Y_chunk_n = Y_cp[n_start:n_end]
            # Slice the *precomputed* Y norms
            norm_Y_sq_chunk = norm_Y_sq_full[n_start:n_end] # Shape (curr_N,)
            curr_N = Y_chunk_n.shape[0]
            if curr_N == 0: continue

            try:
                # Compute dot products for the tile
                dot_products_tile = cp.matmul(X_chunk_q, Y_chunk_n.T)

                # Calculate squared L2 distance using precomputed norms
                dist_sq_tile = norm_X_sq_chunk - 2 * dot_products_tile
                dist_sq_tile += norm_Y_sq_chunk[None, :] # Broadcast add ||Y||^2

                dist_sq_tile = cp.maximum(0.0, dist_sq_tile)
                dist_sq_total[q_start:q_end, n_start:n_end] = dist_sq_tile

                del dot_products_tile, dist_sq_tile, Y_chunk_n, norm_Y_sq_chunk # Cleanup tile data

            except cp.cuda.memory.OutOfMemoryError:
                print(f"\n--- OOM Error within main L2 tile (D={D}, Tile={curr_Q}x{curr_N}) ---")
                print(f"--- Try reducing Q_TILE/N_TILE in distance_l22 definition ---")
                dist_sq_total[q_start:q_end, n_start:n_end] = cp.inf
                # Clean up norms etc before continuing/raising
                del norm_X_sq_chunk, norm_Y_sq_full
                cp.get_default_memory_pool().free_all_blocks()
                # Re-raising might be better to stop the problematic dimension
                raise

            # Optional: More aggressive memory freeing
            # cp.get_default_memory_pool().free_all_blocks()

        del norm_X_sq_chunk # Clean up X norm chunk

    del norm_Y_sq_full # Clean up Y norms

    return dist_sq_total
def distance_cosine2(X, Y, Q_TILE=1024, N_TILE=1024, epsilon=1e-8): # Add tile sizes
    """ Calculates pairwise cosine distance (1 - similarity) using CuPy with tiling. """
    X_cp = cp.asarray(X, dtype=cp.float32)
    Y_cp = cp.asarray(Y, dtype=cp.float32)
    if X_cp.ndim == 1: X_cp = X_cp[None, :]
    if Y_cp.ndim == 1: Y_cp = Y_cp[None, :]

    Q, D = X_cp.shape
    N = Y_cp.shape[0]
    if D != Y_cp.shape[1]: raise ValueError(f"Dimension mismatch: X({D}) vs Y({N},{Y_cp.shape[1]})")
    if Q == 0 or N == 0: return cp.empty((Q,N), dtype=cp.float32)

    # print(f"CuPy Pairwise Cosine (Tiled): Shapes {X_cp.shape}, {Y_cp.shape}")
    cosine_dist = cp.empty((Q, N), dtype=cp.float32)

    # Precompute Y norms (can do this once)
    # Add epsilon here to avoid dividing by zero later if norm is exactly 0
    norm_Y_full = cp.linalg.norm(Y_cp, axis=1) + epsilon # Shape (N,)

    for q_start in range(0, Q, Q_TILE):
        q_end = min(q_start + Q_TILE, Q)
        X_chunk_q = X_cp[q_start:q_end] # Shape (curr_Q, D)
        curr_Q = X_chunk_q.shape[0]
        if curr_Q == 0: continue

        # Precompute X norms for the current chunk
        norm_X_chunk = cp.linalg.norm(X_chunk_q, axis=1) + epsilon # Shape (curr_Q,)

        for n_start in range(0, N, N_TILE):
            n_end = min(n_start + N_TILE, N)
            Y_chunk_n = Y_cp[n_start:n_end] # Shape (curr_N, D)
            norm_Y_chunk = norm_Y_full[n_start:n_end] # Shape (curr_N,)
            curr_N = Y_chunk_n.shape[0]
            if curr_N == 0: continue

            # Calculate dot products for the tile: (curr_Q, D) @ (D, curr_N) -> (curr_Q, curr_N)
            try:
                dot_products_tile = cp.matmul(X_chunk_q, Y_chunk_n.T)

                # Calculate norm product for the tile using broadcasting: (curr_Q, 1) * (1, curr_N) -> (curr_Q, curr_N)
                norm_product_tile = cp.outer(norm_X_chunk, norm_Y_chunk) # Equivalent to norm_X_chunk[:, None] * norm_Y_chunk[None, :]

                # Calculate cosine similarity for the tile
                cosine_similarity_tile = dot_products_tile / norm_product_tile
                cosine_similarity_tile = cp.clip(cosine_similarity_tile, -1.0, 1.0) # Clip before subtraction

                # Calculate cosine distance for the tile
                dist_tile = cp.maximum(0.0, 1.0 - cosine_similarity_tile) # Ensure non-negative

                # Store result in the output matrix slice
                cosine_dist[q_start:q_end, n_start:n_end] = dist_tile

                # Optional: Clean up intermediate tile explicitly
                del dot_products_tile, norm_product_tile, cosine_similarity_tile, dist_tile

            except cp.cuda.memory.OutOfMemoryError:
                print(f"\n--- OOM Error even within Cosine tile (D={D}, Tile={curr_Q}x{curr_N}) ---")
                print(f"--- Try reducing Q_TILE/N_TILE in distance_cosine2 definition ---")
                cosine_dist[q_start:q_end, n_start:n_end] = cp.inf # Assign placeholder
                cp.get_default_memory_pool().free_all_blocks() # Attempt cleanup
                continue

            # Optional: Free memory more aggressively if needed
            # cp.get_default_memory_pool().free_all_blocks()

    # Clean up precomputed norms
    del norm_Y_full, norm_X_chunk # Ensure cleanup happens outside inner loops where possible
    # cp.get_default_memory_pool().free_all_blocks() # Final cleanup

    return cosine_dist

# Replace the old distance_manhattan2 function with this one:

def distance_manhattan2(X, Y, Q_TILE=256, N_TILE=256): # Tile sizes, can be tuned
    """
    Calculates pairwise Manhattan (L1) distance using CuPy with tiling
    to manage memory usage, suitable for large inputs where broadcasting
    the full intermediate (Q, N, D) tensor would cause OutOfMemory errors.

    Args:
        X (cp.ndarray or array-like): Query vectors, shape (Q, D).
        Y (cp.ndarray or array-like): Database vectors, shape (N, D).
        Q_TILE (int): Tile size for the query dimension (Q).
        N_TILE (int): Tile size for the database dimension (N).
        epsilon (float): Small value for numerical stability (not typically needed for L1).

    Returns:
        cp.ndarray: Pairwise Manhattan distances, shape (Q, N).
    """
    # --- Input Handling ---
    X_cp = cp.asarray(X, dtype=cp.float32)
    Y_cp = cp.asarray(Y, dtype=cp.float32)
    if X_cp.ndim == 1: X_cp = X_cp[None, :] # Reshape (D,) to (1, D)
    if Y_cp.ndim == 1: Y_cp = Y_cp[None, :] # Reshape (D,) to (1, D)

    Q, D = X_cp.shape
    N, D_Y = Y_cp.shape
    if D != D_Y:
        raise ValueError(f"Dimension mismatch: X({D}) vs Y({D_Y})")
    if Q == 0 or N == 0:
        return cp.empty((Q, N), dtype=cp.float32) # Handle empty inputs

    # --- Initialization ---
    # print(f"CuPy Pairwise Manhattan (Tiled): Shapes {X_cp.shape}, {Y_cp.shape}, Tile={Q_TILE}x{N_TILE}")
    l1_distance = cp.empty((Q, N), dtype=cp.float32)

    # --- Tiled Computation ---
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

            # Compute pairwise distances within the tile using broadcasting
            # Intermediate shape: (curr_Q, curr_N, D)
            # Peak memory reduced significantly, e.g. Q_TILE * N_TILE * D * 4 bytes per tile
            try:
                # Broadcasting: X(curr_Q, 1, D) - Y(1, curr_N, D) -> diff(curr_Q, curr_N, D)
                abs_diff_tile = cp.abs(X_chunk_q[:, None, :] - Y_chunk_n[None, :, :])
                # Sum over the feature dimension D
                l1_distance_tile = cp.sum(abs_diff_tile, axis=2) # Shape (curr_Q, curr_N)

                # Store result in the output matrix slice
                l1_distance[q_start:q_end, n_start:n_end] = l1_distance_tile

                # Optional: Clean up intermediate tile explicitly if memory is extremely tight
                del abs_diff_tile, l1_distance_tile

            except cp.cuda.memory.OutOfMemoryError:
                # Handle OOM error even within a tile
                print(f"\n--- OOM Error within Manhattan tile (D={D}, Tile={curr_Q}x{curr_N}) ---")
                print(f"--- Try reducing Q_TILE/N_TILE in distance_manhattan2 definition ---")
                # Fill problematic tile with Inf and continue, or re-raise
                l1_distance[q_start:q_end, n_start:n_end] = cp.inf
                cp.get_default_memory_pool().free_all_blocks() # Attempt cleanup
                continue # Skip to the next tile

            # Optional: More aggressive memory freeing (can impact performance)
            # cp.get_default_memory_pool().free_all_blocks()

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
    #dot_products = distance_dot(X_prep, A_prep)

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
    #dot_products = distance_dot(X_prep, A_prep)
    X_norm = torch.linalg.norm(X_prep, axis=1, keepdims=True) # (Q, 1)
    A_norm = torch.linalg.norm(A_prep, axis=1, keepdims=True) # (N, 1)
    norm_product = (X_norm + epsilon) * (A_norm.T + epsilon) # Add epsilon before multiplication
    cosine_similarity = dot_products / norm_product
    cosine_similarity.clamp_(min=-1.0, max=1.0)
    cosine_distance = 1.0 - cosine_similarity
    return cosine_distance

# Corrected distance_manhattan (Removes kwargs, uses grid lambda for autotuned kernel)

def distance_manhattan(X, A, N_TILE=4096, prep=True): # Keep N_TILE argument
    """
    Computes pairwise Manhattan (L1) distance using the simple
    'manhattan_kernel_pairwise_simple' kernel, launched from Python
    in tiles over the N dimension of A. Mimics the structure of distance_dot_tiled.
    """
    target_device = X.device if isinstance(X, torch.Tensor) else A.device

    # Prepare tensors if requested
    if prep:
        # Ensure float32, correct device, and contiguous inputs
        X_prep, A_prep = _prepare_tensors(X, A, target_device=target_device)
    else:
        # If prep=False, assume user has prepared tensors correctly
        # Add basic checks for safety
        if not isinstance(X, torch.Tensor) or not isinstance(A, torch.Tensor):
             raise TypeError("Inputs must be torch tensors if prep=False")
        if X.device != target_device or A.device != target_device:
             raise ValueError(f"Inputs must be on the target device ({target_device}) if prep=False")
        if X.dtype != torch.float32 or A.dtype != torch.float32:
             raise TypeError("Inputs must be float32 if prep=False")
        # Note: Contiguity isn't strictly required if passing actual strides,
        # but slicing is generally more efficient on contiguous tensors.
        X_prep, A_prep = X, A

    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    if D != D_A: raise ValueError(f"Dimension mismatch: X({D}) vs A({D_A})")

    # Create the full output tensor (ensure it's on the correct device)
    Out = torch.empty((Q, N), dtype=torch.float32, device=target_device)

    # Define block size for the D-loop inside the simple kernel (tune if needed)
    BLOCK_D_PARAM = 128

    # print(f"Running distance_manhattan (simple kernel, N_TILE={N_TILE})") # Optional debug

    # Loop over A (database) in chunks of N_TILE
    for n_start in range(0, N, N_TILE):
        n_end = min(n_start + N_TILE, N)
        N_chunk = n_end - n_start
        if N_chunk <= 0: continue # Skip empty chunks

        # Get the current chunk of A. Slicing preserves device.
        A_chunk = A_prep[n_start:n_end, :]
        # Get the corresponding slice of the output tensor view. Slicing preserves device.
        Out_chunk = Out[:, n_start:n_end]

        # Grid for this chunk launch is (Q, N_chunk)
        grid = (Q, N_chunk)
        if grid[0] == 0 or grid[1] == 0: continue

        # --- Optional Debug Print for Chunk ---
        # print(f"  Processing chunk N=[{n_start}:{n_end}] (Size={N_chunk})")
        # print(f"    Kernel: manhattan_kernel_pairwise_simple")
        # print(f"    Grid={grid}, Q={Q}, N_chunk={N_chunk}, D={D}, BLOCK_SIZE_D={BLOCK_D_PARAM}")
        # print(f"    A_chunk strides={A_chunk.stride()}, Out_chunk strides={Out_chunk.stride()}")
        # print(f"    Launching kernel...")
        # --- End Debug Print ---

        # Launch the simple kernel for THIS CHUNK
        # Pass pointers to the START of X_prep, A_chunk, and Out_chunk
        # Pass Q, N_chunk (size of this chunk), and D
        # Pass strides for X_prep, A_chunk, and Out_chunk
        manhattan_kernel_pairwise_simple[grid](
            X_prep, A_chunk, Out_chunk,
            Q, N_chunk, D,
            X_prep.stride(0), X_prep.stride(1),     # Strides for X
            A_chunk.stride(0), A_chunk.stride(1),   # Strides for A_chunk
            Out_chunk.stride(0), Out_chunk.stride(1), # Strides for Out_chunk view
            BLOCK_SIZE_D=BLOCK_D_PARAM
        )
        # torch.cuda.synchronize() # Optional: sync after each chunk if debugging OOM or specific chunk errors

    torch.cuda.synchronize(device=target_device) # Sync after all chunks are launched
    return Out

# ============================================================================
# CPU Distance Functions (NumPy / SciPy)
# ============================================================================

def distance_dot_cpu(X_np, A_np):
    """ CPU Pairwise Dot Product using NumPy """
    # Ensure inputs are NumPy arrays
    if not isinstance(X_np, np.ndarray): X_np = np.asarray(X_np, dtype=np.float32)
    if not isinstance(A_np, np.ndarray): A_np = np.asarray(A_np, dtype=np.float32)
    # Ensure correct dtypes (though matmul might handle some cases)
    if X_np.dtype != np.float32: X_np = X_np.astype(np.float32)
    if A_np.dtype != np.float32: A_np = A_np.astype(np.float32)

    # Perform matrix multiplication: (Q, D) @ (D, N) -> (Q, N)
    return X_np @ A_np.T

def distance_l2_squared_cpu(X_np, A_np):
    """ CPU Pairwise Squared Euclidean (L2) distance using SciPy """
    # Ensure inputs are NumPy arrays and float32 for consistency
    if not isinstance(X_np, np.ndarray): X_np = np.asarray(X_np, dtype=np.float32)
    if not isinstance(A_np, np.ndarray): A_np = np.asarray(A_np, dtype=np.float32)
    if X_np.dtype != np.float32: X_np = X_np.astype(np.float32)
    if A_np.dtype != np.float32: A_np = A_np.astype(np.float32)

    # 'sqeuclidean' computes squared L2 distance
    return scipy.spatial.distance.cdist(X_np, A_np, metric='sqeuclidean')

def distance_cosine_cpu(X_np, A_np):
    """ CPU Pairwise Cosine distance using SciPy """
    # Ensure inputs are NumPy arrays and float32 for consistency
    if not isinstance(X_np, np.ndarray): X_np = np.asarray(X_np, dtype=np.float32)
    if not isinstance(A_np, np.ndarray): A_np = np.asarray(A_np, dtype=np.float32)
    if X_np.dtype != np.float32: X_np = X_np.astype(np.float32)
    if A_np.dtype != np.float32: A_np = A_np.astype(np.float32)

    # 'cosine' computes 1 - cosine_similarity
    return scipy.spatial.distance.cdist(X_np, A_np, metric='cosine')

def distance_manhattan_cpu(X_np, A_np):
    """ CPU Pairwise Manhattan (L1) distance using SciPy """
    # Ensure inputs are NumPy arrays and float32 for consistency
    if not isinstance(X_np, np.ndarray): X_np = np.asarray(X_np, dtype=np.float32)
    if not isinstance(A_np, np.ndarray): A_np = np.asarray(A_np, dtype=np.float32)
    if X_np.dtype != np.float32: X_np = X_np.astype(np.float32)
    if A_np.dtype != np.float32: A_np = A_np.astype(np.float32)

    # 'cityblock' is the metric name for Manhattan distance in SciPy
    return scipy.spatial.distance.cdist(X_np, A_np, metric='cityblock')

# ============================================================================
# CPU KNN Function (NumPy / SciPy)
# ============================================================================

def our_knn_cpu(N_A, D, A_np, X_np, K):
    """
    Finds the K nearest neighbors on the CPU using SciPy/NumPy.
    Uses Squared L2 distance.

    Args:
        N_A (int): Number of database points (matches A_np.shape[0]).
        D (int): Dimensionality (matches A_np.shape[1] and X_np.shape[1]).
        A_np (np.ndarray): Database vectors (N_A, D) on CPU.
        X_np (np.ndarray): Query vectors (Q, D) on CPU.
        K (int): Number of neighbors to find.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - topk_indices (np.ndarray): Indices of the K nearest neighbors (Q, K).
            - topk_distances (np.ndarray): Squared L2 distances of the K nearest neighbors (Q, K).
    """
    Q = X_np.shape[0]
    if A_np.shape[0] != N_A or A_np.shape[1] != D or X_np.shape[1] != D:
        raise ValueError("Dimension mismatch or N_A incorrect in our_knn_cpu")
    if K <= 0 or K > N_A:
        raise ValueError(f"Invalid K value ({K}) for N_A={N_A}")

    # 1. Calculate all pairwise squared L2 distances using SciPy's cdist
    all_distances_sq = distance_l2_squared_cpu(X_np, A_np) # Shape (Q, N_A)

    # 2. Find the top K smallest distances for each query using argpartition
    # np.argpartition finds the indices that *would* partition the array.
    # We want the first K indices if the array were sorted (smallest distances).
    # kth=K-1 means elements with index < K will be <= the element at index K-1
    k_indices_unsorted = np.argpartition(all_distances_sq, kth=K-1, axis=1)[:, :K] # Shape (Q, K)

    # 3. Get the actual distances corresponding to these indices
    topk_distances_unsorted = np.take_along_axis(all_distances_sq, k_indices_unsorted, axis=1) # Shape (Q, K)

    # 4. Sort the results within the top K for each query
    # Argsort along the K dimension of the unsorted distances
    sorted_order_in_k = np.argsort(topk_distances_unsorted, axis=1) # Shape (Q, K)

    # Apply this sort order to both the indices and the distances
    topk_indices = np.take_along_axis(k_indices_unsorted, sorted_order_in_k, axis=1)
    topk_distances = np.take_along_axis(topk_distances_unsorted, sorted_order_in_k, axis=1)

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
    all_distances = distance_l2(X_prep, A_prep) # Shape (Q, N_A)
    #all_distances = distance_dot(X_prep, A_prep)
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


def our_knn_stream(N_A, D, A_cp, X_cp, K, batch_size_q=256): # Add batch_size_q parameter
    """
    Finds the K nearest neighbors using brute-force pairwise SQUARED L2 distance (CuPy).
    Uses query batching to handle large Q * N distance matrices.
    Returns original indices (int64) and SQUARED L2 distances (float32).
    Handles K > N_A by padding results.
    """
    if not cupy_device_ok: raise RuntimeError("CuPy device not available.")
    # --- Input Validation ---
    if not isinstance(A_cp, cp.ndarray): A_cp = cp.asarray(A_cp, dtype=cp.float32)
    elif A_cp.dtype != cp.float32: A_cp = A_cp.astype(cp.float32)
    if not isinstance(X_cp, cp.ndarray): X_cp = cp.asarray(X_cp, dtype=cp.float32)
    elif X_cp.dtype != cp.float32: X_cp = X_cp.astype(cp.float32)

    if A_cp.ndim != 2: raise ValueError(f"Database A_cp must be 2D (N, D), got shape {A_cp.shape}")
    if X_cp.ndim != 2: raise ValueError(f"Queries X_cp must be 2D (Q, D), got shape {X_cp.shape}")

    actual_N_A, actual_D = A_cp.shape
    Q, query_D = X_cp.shape

    if query_D != actual_D: raise ValueError(f"Dimension mismatch: A_cp D={actual_D}, X_cp D={query_D}")

    # Use actual N_A
    N_A = actual_N_A

    # Handle empty database or query set
    if N_A == 0:
         print("Warning: Brute force called with empty database A_cp.")
         return cp.full((Q, K), -1, dtype=cp.int64), cp.full((Q, K), cp.inf, dtype=cp.float32)
    if Q == 0:
         print("Warning: Brute force called with empty query set X_cp.")
         return cp.empty((0, K), dtype=cp.int64), cp.empty((0, K), dtype=cp.float32)

    if not K > 0: raise ValueError("K must be positive")

    # Adjust K if it's larger than the number of data points
    effective_K = min(K, N_A)
    if effective_K != K:
         print(f"Note: Brute force K={K} requested > N_A={N_A}. Using K={effective_K}.")

    if effective_K == 0: # Handle K=0 or effective_K=0 case
         return cp.empty((Q, 0), dtype=cp.int64), cp.empty((Q, 0), dtype=cp.float32)

    print(f"Running k-NN Brute Force (CuPy, Batched): Q={Q}, N={N_A}, D={actual_D}, K={effective_K}, BatchSize={batch_size_q}")
    start_time = time.time()

    # Pre-allocate result arrays
    all_topk_indices_cp = cp.full((Q, effective_K), -1, dtype=cp.int64)
    all_topk_distances_sq_cp = cp.full((Q, effective_K), cp.inf, dtype=cp.float32)

    # --- Batch Processing Loop ---
    for q_start in range(0, Q, batch_size_q):
        q_end = min(q_start + batch_size_q, Q)
        batch_q_indices = slice(q_start, q_end) # Slice for indexing results
        X_batch_cp = X_cp[batch_q_indices]     # Current batch of queries
        current_batch_size = X_batch_cp.shape[0]
        if current_batch_size == 0: continue

        # print(f"  Processing query batch {q_start}-{q_end-1}...") # Optional progress

        # Calculate SQUARED L2 distances for the current batch
        try:
            batch_distances_sq = pairwise_l2_squared_cupy(X_batch_cp, A_cp) # Shape (current_batch_size, N_A)
        except cp.cuda.memory.OutOfMemoryError as e:
            batch_mem_gb = current_batch_size * N_A * 4 / (1024**3)
            print(f"OOM Error during Brute Force batch distance calculation:")
            print(f"  Batch Q={current_batch_size}, N={N_A}. Estimated matrix memory: {batch_mem_gb:.2f} GB")
            print(f"  Try reducing batch_size_q (current={batch_size_q}).")
            raise e # Re-raise
        except Exception as e:
            print(f"Error during Brute Force batch distance calculation: {e}")
            raise e


        # Find top K for the current batch
        # Need k < N for argpartition
        k_partition = min(effective_K, N_A - 1) if N_A > 0 else 0
        if k_partition < 0: k_partition = 0

        batch_topk_indices = None
        batch_topk_distances_sq = None

        try:
            if effective_K >= N_A: # If K includes all points, just sort all
                batch_topk_indices = cp.argsort(batch_distances_sq, axis=1)[:, :effective_K]
            else:
                 # Ensure N_A > 0 before calling argpartition
                 if N_A > 0:
                      topk_indices_unstructured = cp.argpartition(batch_distances_sq, k_partition, axis=1)[:, :effective_K]
                      topk_distances_sq_unstructured = cp.take_along_axis(batch_distances_sq, topk_indices_unstructured, axis=1)
                      sorted_order_in_k = cp.argsort(topk_distances_sq_unstructured, axis=1)
                      batch_topk_indices = cp.take_along_axis(topk_indices_unstructured, sorted_order_in_k, axis=1)
                 else: # Should not happen if N_A=0 check passed
                      batch_topk_indices = cp.empty((current_batch_size, 0), dtype=cp.int64)

            # Retrieve the final sorted distances
            if batch_topk_indices is not None and batch_topk_indices.size > 0:
                 batch_topk_distances_sq = cp.take_along_axis(batch_distances_sq, batch_topk_indices, axis=1)
            elif effective_K == 0:
                 batch_topk_distances_sq = cp.empty((current_batch_size, 0), dtype=cp.float32)
            else: # Handle unexpected empty indices
                 print(f"Warning: batch_topk_indices empty/None in batch {q_start}-{q_end-1}")
                 batch_topk_distances_sq = cp.full((current_batch_size, effective_K), cp.inf, dtype=cp.float32)
                 if batch_topk_indices is None: batch_topk_indices = cp.full((current_batch_size, effective_K), -1, dtype=cp.int64)

            # Store batch results in the main arrays
            all_topk_indices_cp[batch_q_indices] = batch_topk_indices
            all_topk_distances_sq_cp[batch_q_indices] = batch_topk_distances_sq

            # Optional: Clear intermediate batch results to free memory sooner
            del batch_distances_sq, batch_topk_indices, batch_topk_distances_sq
            if 'topk_indices_unstructured' in locals(): del topk_indices_unstructured
            if 'topk_distances_sq_unstructured' in locals(): del topk_distances_sq_unstructured
            if 'sorted_order_in_k' in locals(): del sorted_order_in_k
            # cp.get_default_memory_pool().free_all_blocks() # Use cautiously, can impact performance


        except Exception as e:
            print(f"Error during Brute Force batch top-K ({q_start}-{q_end-1}): {e}")
            # Decide how to handle: continue with potentially missing batch results or raise error?
            # For recall, probably better to raise the error
            raise e
    # --- End Batch Loop ---

    cp.cuda.Stream.null.synchronize()
    end_time = time.time()
    print(f"k-NN Brute Force (CuPy, Batched) total computation time: {end_time - start_time:.4f} seconds")

    # Pad results if original K > effective_K (i.e., K > N_A)
    if K > effective_K:
        pad_width = K - effective_K
        indices_pad = cp.full((Q, pad_width), -1, dtype=cp.int64)
        dists_pad = cp.full((Q, pad_width), cp.inf, dtype=cp.float32)
        all_topk_indices_cp = cp.hstack((all_topk_indices_cp, indices_pad))
        all_topk_distances_sq_cp = cp.hstack((all_topk_distances_sq_cp, dists_pad))

    return all_topk_indices_cp.astype(cp.int64), all_topk_distances_sq_cp.astype(cp.float32)

    # Or retu


# ============================================================================
# Main Execution Block (Benchmarking KNN and Distances across Dimensions)
# ============================================================================
# Assume necessary imports (torch, triton, cupy as cp, time, traceback)
# Assume necessary function definitions are present above:
# - device = torch.device("cuda:0")
# - PyTorch/Triton: distance_dot_tiled, distance_l2, distance_cosine, distance_manhattan, our_knn
# - CuPy: distance_dot2, distance_l22, distance_cosine2, distance_manhattan2, our_knn_stream
#   (and their helpers like pairwise_l2_squared_cupy if needed)

# ============================================================================
# Main Execution Block (Benchmarking KNN and Distances across Dimensions)
# ============================================================================
# ============================================================================
# Main Execution Block (Benchmarking + Numerical Check)
# ============================================================================
# ============================================================================
# Main Execution Block (Enhanced for Debugging)
# ============================================================================
# ============================================================================
# Main Execution Block (Benchmarking + Numerical Check)
# ============================================================================
if __name__ == "__main__":
    # --- Fixed Parameters ---
    N_data = 1000000 # Using 4 Million points
    N_queries = 1    # Using 1 query as per your last log
    K_val = 10          # K for KNN
    NUM_RUNS = 2       # Number of timed runs for averaging
    WARMUP_RUNS = 1     # Number of warm-up runs
    # --- CPU BENCHMARKING FLAG ---
    BENCHMARK_CPU = False # Set to False to skip CPU tests (can be slow)

    # --- Dimensions to Test ---
    dimensions_to_test = [2,2,4,64,256,1024]

    # --- Tolerance for Numerical Check ---
    rtol_check = 1e-4
    atol_check = 1e-5

    # --- DEBUG FLAG ---
    DETAILED_DEBUG = False # Set to True for extra info during GPU gen/warmup

    print(f"--- GPU & CPU KNN/DISTANCE BENCHMARKING & VALIDATION ---") # Updated title
    print(f"Fixed Params: N={N_data}, Q={N_queries}, K={K_val}, Warmup={WARMUP_RUNS}, Runs={NUM_RUNS}")
    print(f"Testing Dimensions: {dimensions_to_test}")
    print(f"Benchmark CPU: {BENCHMARK_CPU}")

    print(f"--- GPU KNN/DISTANCE BENCHMARKING & VALIDATION ---")
    print(f"Fixed Params: N={N_data}, Q={N_queries}, K={K_val}, Warmup={WARMUP_RUNS}, Runs={NUM_RUNS}")
    print(f"Testing Dimensions: {dimensions_to_test}")
    print(f"Numerical Check Tolerance: rtol={rtol_check}, atol={atol_check}")
    if DETAILED_DEBUG: print(f"DETAILED_DEBUG MODE: ON")

    # --- Check Devices ---
    try: # PyTorch Device Check
        if not torch.cuda.is_available(): raise RuntimeError("Torch CUDA not available.")
        device = torch.device("cuda:0"); print(f"Using PyTorch device: {device}")
    except Exception as e: print(f"PyTorch device error: {e}"); exit()
    try: # CuPy Device Check
        cp.cuda.Device(0).use(); print(f"Using CuPy device: {cp.cuda.Device(0)}")
        cupy_device_ok = True
    except Exception as e: print(f"CuPy device error: {e}"); cupy_device_ok = False

    # --- Storage for results ---
    results = {}

    # Loop through each dimension
    for Dim in dimensions_to_test:
        print("\n" + "#"*70)
        print(f"# Starting Test for Dimension D = {Dim}")
        print("#"*70 + "\n")

        results[Dim] = {}
        dimension_failed = False
        A_data = A_data_cp = X_queries = X_queries_cp = None
        A_data_np = X_queries_np = None # Add NumPy vars

        # --- Generate Base Data (GPU and transfer to CPU) ---
        print("="*40); print(f"Generating Data (D={Dim})..."); print("="*40)
        try:
            # Generate on GPU
            A_data = torch.randn(N_data, Dim, dtype=torch.float32, device=device)
            X_queries = torch.randn(N_queries, Dim, dtype=torch.float32, device=device)
            torch.cuda.synchronize(device=device)
            print(f"Database shape (Torch): {A_data.shape}")
            print(f"Query shape (Torch): {X_queries.shape}")

            # Transfer to CPU (NumPy) - Outside benchmark timing
            if BENCHMARK_CPU:
                print("Transferring data to CPU (NumPy)...")
                start_transfer = time.perf_counter()
                A_data_np = A_data.cpu().numpy()
                X_queries_np = X_queries.cpu().numpy()
                end_transfer = time.perf_counter()
                print(f"  Transfer time: {end_transfer - start_transfer:.4f} s")
                print(f"Database shape (NumPy): {A_data_np.shape}")
                print(f"Query shape (NumPy): {X_queries_np.shape}")

            # Transfer to CuPy (if needed)
            if cupy_device_ok:
                A_data_contig = A_data.contiguous(); X_queries_contig = X_queries.contiguous()
                dlpack_A = torch.to_dlpack(A_data_contig); dlpack_X = torch.to_dlpack(X_queries_contig)
                A_data_cp = cp.from_dlpack(dlpack_A); X_queries_cp = cp.from_dlpack(dlpack_X)
                cp.cuda.Stream.null.synchronize()
                print(f"Database shape (CuPy): {A_data_cp.shape}")
            print("-" * 40)

        except Exception as e:
            # ...(Error handling for data generation remains the same)...
            print(f"*** CRITICAL ERROR during Data Generation/Transfer (D={Dim}): {e} ***")
            import traceback; traceback.print_exc()
            dimension_failed = True
            # Clean up CPU arrays too
            if 'A_data_np' in locals() and A_data_np is not None: del A_data_np
            if 'X_queries_np' in locals() and X_queries_np is not None: del X_queries_np
            #...(rest of cleanup)...
            continue

        # ===---------------------------------------------------------===
        # ===              WARM-UP RUNS (Individual Checks)         ===
        # ===---------------------------------------------------------===
        print("="*40); print(f"Performing Warm-up Runs (D={Dim})..."); print("="*40)
        # Define the functions to warm up
        warmup_functions_torch = {
            "distance_dot_tiled": lambda: distance_dot_tiled(X_queries, A_data),
            "distance_l2": lambda: distance_l2(X_queries, A_data),
            "distance_cosine": lambda: distance_cosine(X_queries, A_data),
            "distance_manhattan": lambda: distance_manhattan(X_queries, A_data), # Assumes this uses the simple kernel now
            "our_knn": lambda: our_knn(N_data, Dim, A_data, X_queries, K_val),
        }
        warmup_functions_cpu = {} # CPU NumPy/SciPy
        if BENCHMARK_CPU and A_data_np is not None:
            warmup_functions_cpu = {
                "distance_dot_cpu": lambda: distance_dot_cpu(X_queries_np, A_data_np),
                "distance_l2_squared_cpu": lambda: distance_l2_squared_cpu(X_queries_np, A_data_np),
                "distance_cosine_cpu": lambda: distance_cosine_cpu(X_queries_np, A_data_np),
                "distance_manhattan_cpu": lambda: distance_manhattan_cpu(X_queries_np, A_data_np),
                "our_knn_cpu": lambda: our_knn_cpu(N_data, Dim, A_data_np, X_queries_np, K_val),
            }
        if cupy_device_ok and A_data_cp is not None:
             warmup_functions_cupy = {
                 "distance_dot2": lambda: distance_dot2(X_queries_cp, A_data_cp),
                 "distance_l22": lambda: distance_l22(X_queries_cp, A_data_cp),
                 "distance_cosine2": lambda: distance_cosine2(X_queries_cp, A_data_cp),
                 "distance_manhattan2": lambda: distance_manhattan2(X_queries_cp, A_data_cp),
                 "our_knn_stream": lambda: our_knn_stream(N_data, Dim, A_data_cp, X_queries_cp, K_val),
             }
        else:
             warmup_functions_cupy = {}

        # Execute warm-up runs
        for i in range(WARMUP_RUNS):
            print(f"--- Warm-up Run {i+1}/{WARMUP_RUNS} ---")
            if dimension_failed: break

            print("  Warming up PyTorch/Triton functions...")
            for name, func in warmup_functions_torch.items():
                if dimension_failed: break
                if DETAILED_DEBUG: print(f"    Attempting warm-up for: {name}")
                try:
                    _ = func(); torch.cuda.synchronize(device=device)
                    if DETAILED_DEBUG: print(f"      Warm-up OK: {name}")
                except Exception as e:
                    print(f"    *** ERROR during warm-up for {name} (D={Dim}, Run={i+1}): {e} ***")
                    import traceback; traceback.print_exc()
                    dimension_failed = True

            if warmup_functions_cupy:
                 print("  Warming up CuPy functions...")
                 for name, func in warmup_functions_cupy.items():
                     if dimension_failed: break
                     if DETAILED_DEBUG: print(f"    Attempting warm-up for: {name}")
                     try:
                         _ = func(); cp.cuda.Stream.null.synchronize()
                         if DETAILED_DEBUG: print(f"      Warm-up OK: {name}")
                     except Exception as e:
                        print(f"    *** ERROR during warm-up for {name} (D={Dim}, Run={i+1}): {e} ***")
                        import traceback; traceback.print_exc()
                        dimension_failed = True
                 # --- Add CPU Warmup ---
            # --- Add CPU Warmup ---
            if warmup_functions_cpu:
                print("  Warming up CPU functions...")
                BENCHMARK_CPU_THIS_DIM = BENCHMARK_CPU # Assume true initially if flag is set
                cpu_warmup_ok = True
                for name, func in warmup_functions_cpu.items():
                    if dimension_failed:
                        cpu_warmup_ok = False
                        break
                    print(f"    Attempting warm-up for CPU function: {name} ...", end='', flush=True) # Print before, no newline
                    start_cpu_warmup_call = time.perf_counter()
                    try:
                        _ = func() # Execute CPU function
                        end_cpu_warmup_call = time.perf_counter()
                        print(f" done. (took {end_cpu_warmup_call - start_cpu_warmup_call:.4f} s)") # Print after + time
                    except MemoryError as e:
                        print(f"\n    *** MEMORY ERROR during warm-up for {name} (D={Dim}): {e} ***")
                        import traceback; traceback.print_exc()
                        print(f"    Likely insufficient RAM for this operation on CPU at N={N_data}, D={Dim}.")
                        print(f"    Skipping CPU benchmarks for dimension {Dim}.")
                        cpu_warmup_ok = False
                        break # Stop CPU warmup
                    except Exception as e:
                        print(f"\n    *** ERROR during warm-up for {name} (D={Dim}): {e} ***")
                        import traceback; traceback.print_exc()
                        print(f"    Skipping CPU benchmarks for dimension {Dim}.")
                        cpu_warmup_ok = False
                        break # Stop CPU warmup

        # Set the flag based on whether warmup completed ok
                BENCHMARK_CPU_THIS_DIM = cpu_warmup_ok and BENCHMARK_CPU
                if not BENCHMARK_CPU_THIS_DIM:
                    print("    CPU warmup aborted or skipped.")

            else: # No CPU functions defined or flag is false
                BENCHMARK_CPU_THIS_DIM = False


        # --- Post Warm-up Check ---
        if dimension_failed:
            print("\n*** ERROR occurred during warm-up phase. Skipping benchmarks and checks for this dimension. ***")
        else:
            print("Warm-up complete for D={Dim}.")
            # Optional: Clear memory after warm-up
            torch.cuda.empty_cache()
            if cupy_device_ok: cp.get_default_memory_pool().free_all_blocks()

        # ===--------------------------------------------------===
        # ===      BENCHMARKING (Skip if warm-up failed)     ===
        # ===--------------------------------------------------===
        if not dimension_failed:

            # --- Numerical Stability Check (Optional but recommended) ---
            print("\n" + "="*40); print(f"Numerical Stability Check (Dot Product, D={Dim})..."); print("="*40)
            try:
                # Using distance_dot_tiled as it's Triton based
                dot_triton = distance_dot_tiled(X_queries, A_data)
                dot_matmul = torch.matmul(X_queries.contiguous(), A_data.contiguous().T)
                torch.cuda.synchronize()
                if dot_triton.shape != dot_matmul.shape:
                     print(f"  ERROR: Shape mismatch! Triton={dot_triton.shape}, Matmul={dot_matmul.shape}")
                else:
                    are_close = torch.allclose(dot_triton, dot_matmul, rtol=rtol_check, atol=atol_check)
                    print(f"  Numerical Check Passed: {are_close}" + (f"" if are_close else " - WARNING: Results differ!"))
                    if not are_close:
                         max_diff = torch.max(torch.abs(dot_triton - dot_matmul)).item()
                         print(f"    Max Abs Diff: {max_diff:.6e}")
                del dot_triton, dot_matmul
            except Exception as e:
                print(f"  ERROR during numerical check: {e}")
                import traceback; traceback.print_exc()
            print("-" * 40)


            # --- Distance Function Benchmarks ---
            print("\n" + "="*40); print(f"Benchmarking Distance Functions (D={Dim})..."); print("="*40)
            # PyTorch/Triton Distances
            try:
                start_event = torch.cuda.Event(enable_timing=True); end_event = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); start_event.record()
                for r in range(NUM_RUNS): _ = distance_dot_tiled(X_queries, A_data)
                end_event.record(); torch.cuda.synchronize(); avg_time = (start_event.elapsed_time(end_event)/1000.0)/NUM_RUNS
                print(f"Torch distance_dot_tiled Avg Time:   {avg_time:.6f} seconds")
                results[Dim]['dist_dot_torch_tiled'] = avg_time
            except Exception as e: print(f"Error benchmarking distance_dot_tiled: {e}")
            try:
                start_event = torch.cuda.Event(enable_timing=True); end_event = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); start_event.record()
                for r in range(NUM_RUNS): _ = distance_l2(X_queries, A_data)
                end_event.record(); torch.cuda.synchronize(); avg_time = (start_event.elapsed_time(end_event)/1000.0)/NUM_RUNS
                print(f"Torch distance_l2 Avg Time:          {avg_time:.6f} seconds")
                results[Dim]['dist_l2_torch'] = avg_time
            except Exception as e: print(f"Error benchmarking distance_l2: {e}")
            try:
                start_event = torch.cuda.Event(enable_timing=True); end_event = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); start_event.record()
                for r in range(NUM_RUNS): _ = distance_cosine(X_queries, A_data)
                end_event.record(); torch.cuda.synchronize(); avg_time = (start_event.elapsed_time(end_event)/1000.0)/NUM_RUNS
                print(f"Torch distance_cosine Avg Time:      {avg_time:.6f} seconds")
                results[Dim]['dist_cos_torch'] = avg_time
            except Exception as e: print(f"Error benchmarking distance_cosine: {e}")
            try: # Assuming distance_manhattan now uses the simple kernel + N_TILE wrapper
                start_event = torch.cuda.Event(enable_timing=True); end_event = torch.cuda.Event(enable_timing=True); torch.cuda.synchronize(); start_event.record()
                for r in range(NUM_RUNS): _ = distance_manhattan(X_queries, A_data)
                end_event.record(); torch.cuda.synchronize(); avg_time = (start_event.elapsed_time(end_event)/1000.0)/NUM_RUNS
                print(f"Torch distance_manhattan Avg Time:   {avg_time:.6f} seconds")
                results[Dim]['dist_man_torch'] = avg_time # Store under the main key
            except Exception as e: print(f"Error benchmarking distance_manhattan: {e}")

            print("-" * 25) # Separator

            # CuPy Distances
            if cupy_device_ok and A_data_cp is not None:
                try:
                    start_event=cp.cuda.Event();end_event=cp.cuda.Event();cp.cuda.Stream.null.synchronize();start_event.record()
                    for r in range(NUM_RUNS): _ = distance_dot2(X_queries_cp, A_data_cp)
                    end_event.record();end_event.synchronize();avg_time=(cp.cuda.get_elapsed_time(start_event,end_event)/1000.0)/NUM_RUNS
                    print(f"CuPy distance_dot2 Avg Time:         {avg_time:.6f} seconds")
                    results[Dim]['dist_dot_cupy'] = avg_time
                except Exception as e: print(f"Error benchmarking distance_dot2: {e}")
                try:
                    start_event=cp.cuda.Event();end_event=cp.cuda.Event();cp.cuda.Stream.null.synchronize();start_event.record()
                    for r in range(NUM_RUNS): _ = distance_l22(X_queries_cp, A_data_cp)
                    end_event.record();end_event.synchronize();avg_time=(cp.cuda.get_elapsed_time(start_event,end_event)/1000.0)/NUM_RUNS
                    print(f"CuPy distance_l22 Avg Time:         {avg_time:.6f} seconds")
                    results[Dim]['dist_l2_cupy'] = avg_time
                except Exception as e: print(f"Error benchmarking distance_l22: {e}")
                try:
                    start_event=cp.cuda.Event();end_event=cp.cuda.Event();cp.cuda.Stream.null.synchronize();start_event.record()
                    for r in range(NUM_RUNS): _ = distance_cosine2(X_queries_cp, A_data_cp)
                    end_event.record();end_event.synchronize();avg_time=(cp.cuda.get_elapsed_time(start_event,end_event)/1000.0)/NUM_RUNS
                    print(f"CuPy distance_cosine2 Avg Time:    {avg_time:.6f} seconds")
                    results[Dim]['dist_cos_cupy'] = avg_time
                except Exception as e: print(f"Error benchmarking distance_cosine2: {e}")
                try:
                    start_event=cp.cuda.Event();end_event=cp.cuda.Event();cp.cuda.Stream.null.synchronize();start_event.record()
                    for r in range(NUM_RUNS): _ = distance_manhattan2(X_queries_cp, A_data_cp)
                    end_event.record();end_event.synchronize();avg_time=(cp.cuda.get_elapsed_time(start_event,end_event)/1000.0)/NUM_RUNS
                    if cp.isinf(avg_time): print("CuPy distance_manhattan2 likely OOM occurred.")
                    else: print(f"CuPy distance_manhattan2 Avg Time:  {avg_time:.6f} seconds")
                    results[Dim]['dist_man_cupy'] = avg_time
                except Exception as e: print(f"Error benchmarking distance_manhattan2: {e}")
            else: print("CuPy distance benchmarks skipped.")
             # --- Add CPU Distance Benchmarks ---
            print("--- CPU (NumPy/SciPy) ---")
            if BENCHMARK_CPU_THIS_DIM and A_data_np is not None:
                try: # Dot Product CPU
                    start_time = time.perf_counter()
                    for r in range(NUM_RUNS): _ = distance_dot_cpu(X_queries_np, A_data_np)
                    end_time = time.perf_counter(); avg_time = (end_time - start_time) / NUM_RUNS
                    print(f"CPU distance_dot_cpu Avg Time:       {avg_time:.6f} seconds")
                    results[Dim]['dist_dot_cpu'] = avg_time
                except Exception as e: print(f"Error benchmarking distance_dot_cpu: {e}")
                try: # L2 Squared CPU
                    start_time = time.perf_counter()
                    for r in range(NUM_RUNS): _ = distance_l2_squared_cpu(X_queries_np, A_data_np)
                    end_time = time.perf_counter(); avg_time = (end_time - start_time) / NUM_RUNS
                    print(f"CPU distance_l2_squared_cpu Avg Time:{avg_time:.6f} seconds")
                    results[Dim]['dist_l2_cpu'] = avg_time
                except Exception as e: print(f"Error benchmarking distance_l2_squared_cpu: {e}")
                try: # Cosine CPU
                    start_time = time.perf_counter()
                    for r in range(NUM_RUNS): _ = distance_cosine_cpu(X_queries_np, A_data_np)
                    end_time = time.perf_counter(); avg_time = (end_time - start_time) / NUM_RUNS
                    print(f"CPU distance_cosine_cpu Avg Time:    {avg_time:.6f} seconds")
                    results[Dim]['dist_cos_cpu'] = avg_time
                except Exception as e: print(f"Error benchmarking distance_cosine_cpu: {e}")
                try: # Manhattan CPU
                    start_time = time.perf_counter()
                    for r in range(NUM_RUNS): _ = distance_manhattan_cpu(X_queries_np, A_data_np)
                    end_time = time.perf_counter(); avg_time = (end_time - start_time) / NUM_RUNS
                    print(f"CPU distance_manhattan_cpu Avg Time: {avg_time:.6f} seconds")
                    results[Dim]['dist_man_cpu'] = avg_time
                except Exception as e: print(f"Error benchmarking distance_manhattan_cpu: {e}")
            else:
                 print("  CPU distance benchmarks skipped.")
            # --- End Distance Benchmarks ---



            # --- KNN Function Benchmarks ---
            print("\n" + "="*40); print(f"Benchmarking KNN Functions (D={Dim})..."); print("="*40)

            # --- Triton/PyTorch KNN (our_knn) ---
            # This uses distance_l2 -> distance_dot_tiled internally
            knn_indices_torch = None # Define outside try block
            try:
                start_event = torch.cuda.Event(enable_timing=True); end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(); start_event.record()
                for r in range(NUM_RUNS):
                    knn_indices_torch, _ = our_knn(N_data, Dim, A_data, X_queries, K_val) # Capture results
                end_event.record(); torch.cuda.synchronize()
                avg_time = (start_event.elapsed_time(end_event) / 1000.0) / NUM_RUNS
                qps = N_queries / avg_time if avg_time > 0 else 0
                print(f"Triton/Torch our_knn Avg Time:       {avg_time:.6f} seconds ({qps:.2f} QPS)")
                results[Dim]['knn_torch'] = avg_time
                # Optional: Verify results (e.g., shape)
                # if knn_indices_torch is not None: print(f"  Torch KNN indices shape: {knn_indices_torch.shape}")
            except Exception as e:
                print(f"Error benchmarking our_knn (Triton/Torch): {e}")
                import traceback; traceback.print_exc()


            # --- CuPy Stream KNN (our_knn_stream) ---
            # Check if CuPy is available and data exists before benchmarking
            if cupy_device_ok and A_data_cp is not None and X_queries_cp is not None:
                knn_indices_cupy = None # Define outside try block
                try:
                    start_event = cp.cuda.Event(); end_event = cp.cuda.Event()
                    cp.cuda.Stream.null.synchronize(); start_event.record()
                    for r in range(NUM_RUNS):
                        # our_knn_stream expects N, D, A, X, K
                        knn_indices_cupy = our_knn_stream(N_data, Dim, A_data_cp, X_queries_cp, K_val)
                    end_event.record(); end_event.synchronize()
                    avg_time = (cp.cuda.get_elapsed_time(start_event, end_event) / 1000.0) / NUM_RUNS
                    qps = N_queries / avg_time if avg_time > 0 else 0
                    print(f"CuPy our_knn_stream Avg Time:      {avg_time:.6f} seconds ({qps:.2f} QPS)")
                    results[Dim]['knn_cupy_stream'] = avg_time # Use appropriate key
                    # Optional: Verify results (e.g., shape or type)
                    # if knn_indices_cupy is not None:
                    #    print(f"  CuPy KNN result type: {type(knn_indices_cupy)}")
                    #    # our_knn_stream returns a list if B>1, or single array if B=1
                    #    if isinstance(knn_indices_cupy, list):
                    #       if knn_indices_cupy: print(f"  CuPy KNN indices shape (first query): {knn_indices_cupy[0].shape}")
                    #    else:
                    #        print(f"  CuPy KNN indices shape: {knn_indices_cupy.shape}")

                except Exception as e:
                    print(f"Error benchmarking our_knn_stream: {e}")
                    import traceback; traceback.print_exc()
            else:
                print("CuPy our_knn_stream benchmark skipped (CuPy unavailable or data missing).")
            print("--- CPU (NumPy/SciPy) ---")
            if BENCHMARK_CPU_THIS_DIM and A_data_np is not None:
                knn_indices_cpu = None
                try:
                    start_time = time.perf_counter()
                    for r in range(NUM_RUNS):
                        knn_indices_cpu, _ = our_knn_cpu(N_data, Dim, A_data_np, X_queries_np, K_val)
                    end_time = time.perf_counter(); avg_time = (end_time - start_time) / NUM_RUNS
                    qps = N_queries / avg_time if avg_time > 0 else 0
                    print(f"CPU our_knn_cpu Avg Time:          {avg_time:.6f} seconds ({qps:.2f} QPS)")
                    results[Dim]['knn_cpu'] = avg_time
                    # Optional: verify shape
                    # if knn_indices_cpu is not None: print(f"  CPU KNN indices shape: {knn_indices_cpu.shape}")
                except Exception as e:
                     print(f"Error benchmarking our_knn_cpu: {e}")
                     import traceback; traceback.print_exc()
            else:
                print("  CPU KNN benchmark skipped.")
            # --- END KNN Benchmarks ---

        # --- Cleanup for the dimension ---
        # Run cleanup regardless of benchmark success/failure, if data was loaded
        print(f"\n--- Finished Processing Dimension D = {Dim} ---")
        if 'A_data' in locals() and A_data is not None: del A_data
        if 'X_queries' in locals() and X_queries is not None: del X_queries
        if cupy_device_ok and 'A_data_cp' in locals() and A_data_cp is not None: del A_data_cp
        if cupy_device_ok and 'X_queries_cp' in locals() and X_queries_cp is not None: del X_queries_cp
        torch.cuda.empty_cache()
        if cupy_device_ok: cp.get_default_memory_pool().free_all_blocks()
        if 'A_data_np' in locals() and A_data_np is not None: del A_data_np
        if 'X_queries_np' in locals() and X_queries_np is not None: del X_queries_np
        print("-" * 70)
        print("-" * 70)


    int("\n" + "#"*70); print("# ALL DIMENSION BENCHMARKS FINISHED"); print("#"*70)

    # --- Print Summary Table ---
    print("\nBenchmark Summary (Average Times in Seconds):")
    # Adjust width and columns as needed
    table_width = 210 # Increased width
    print("-" * table_width)
    header = f"{'Dim':<6}"
    # Define column order dynamically based on results collected + preference
    col_order = [
        # Torch Distances
        'dist_dot_torch_tiled', 'dist_l2_torch', 'dist_cos_torch', 'dist_man_torch',
        # CuPy Distances
        'dist_dot_cupy', 'dist_l2_cupy', 'dist_cos_cupy', 'dist_man_cupy',
        # CPU Distances - ADDED
        'dist_dot_cpu', 'dist_l2_cpu', 'dist_cos_cpu', 'dist_man_cpu',
        # KNNs (GPU and CPU) - ADDED CPU
        'knn_torch', 'knn_cupy_stream', 'knn_cpu'
    ]
    # ...(rest of dynamic column finding logic)...
    present_cols = set();
    for d in results: present_cols.update(results[d].keys())
    final_cols = [col for col in col_order if col in present_cols]
    for col in sorted(present_cols):
        if col not in final_cols: final_cols.append(col)

    col_width = 25 # Adjust spacing between columns
    for col_key in final_cols: header += f"{col_key:<{col_width}}"
    print(header);
    # Recalculate width based on final columns
    table_width = 6 + len(final_cols) * col_width
    print("-" * table_width)

    for Dim in dimensions_to_test:
        row = f"{Dim:<6}"
        if Dim in results:
            r = results[Dim]
            for col_key in final_cols:
                row += f"{r.get(col_key, float('nan')):<{col_width}.6f}"
            # Adjust N/A spacing based on col_width
            na_spacing = ' ' * ((col_width - 3) // 2)
            na_string = f"{na_spacing}N/A{na_spacing}"
            if col_width % 2 == 0: na_string += " " # Add extra space if even width
            print(row.replace(' ' * col_width + 'nan', na_string)) # More robust NaN replace
        else:
            for _ in final_cols: row += f"{'Skipped':<{col_width}}"
            print(row)
    print("-" * table_width)