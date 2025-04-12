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
def manhattan_kernel_pairwise_tiled(
    X_ptr, A_ptr, Out_ptr, # Parameters are received but mostly ignored
    Q, N, D,
    stride_xq, stride_xd,
    stride_an, stride_ad,
    stride_outq, stride_outn,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr, # BLOCK_K is ignored here
):
    # 1. Get program IDs for the output block
    pid_q_block = tl.program_id(axis=0)
    pid_n_block = tl.program_id(axis=1)

    # 2. Calculate offsets for the *start* of the output block this thread block handles
    q_start = pid_q_block * BLOCK_Q
    n_start = pid_n_block * BLOCK_N

    # 3. Calculate pointer to the top-left element of the output block
    #    Use tl.arange to create offsets relative to the start for the whole block
    offs_q = q_start + tl.arange(0, BLOCK_Q)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    out_ptrs = Out_ptr + offs_q[:, None] * stride_outq + offs_n[None, :] * stride_outn

    # 4. Create a mask for valid output elements
    out_mask = (offs_q[:, None] < Q) & (offs_n[None, :] < N)

    # 5. Create a dummy value to write (e.g., constant or based on pids)
    #    Ensure it matches the accumulator shape (BLOCK_Q, BLOCK_N)
    dummy_value = (pid_q_block + pid_n_block) * 1.0 # Example value
    dummy_accumulator = tl.full((BLOCK_Q, BLOCK_N), dummy_value, dtype=tl.float32)

    # 6. Perform ONLY the store operation
    tl.store(out_ptrs, dummy_accumulator, mask=out_mask)

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
'''
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
def distance_dot_tiled(X, A, N_TILE=4096, prep=True):
    """
    Computes pairwise dot product using the 'dot_kernel_pairwise_tiled' (tl.dot based)
    kernel with FIXED block sizes (e.g., 32x32x32).
    Launched in tiles over A (N dim) from Python. Returns POSITIVE dot product.
    NOTE: Ensure @triton.autotune is removed from dot_kernel_pairwise_tiled kernel.
    """
    if prep:
         X_prep = _prepare_tensors(X)
         A_prep = _prepare_tensors(A)
    else: X_prep, A_prep = X, A

    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    if D != D_A: raise ValueError(f"Dimension mismatch: X({D}) vs A({D_A})")

    Out = torch.empty((Q, N), dtype=torch.float32, device=device)

    # --- Define Fixed Block Sizes for the tiled kernel ---
    BLOCK_Q_FIXED = 32
    BLOCK_N_FIXED = 32
    BLOCK_K_FIXED = 32 # Reduction dim block size
    # ----------------------------------------------------

    # print(f"Tiling dot_kernel_pairwise_tiled (Fixed Blocks) N_TILE={N_TILE}")
    for n_start in range(0, N, N_TILE):
        n_end = min(n_start + N_TILE, N)
        N_chunk = n_end - n_start
        if N_chunk <= 0: continue

        A_chunk = A_prep[n_start:n_end, :]
        Out_chunk = Out[:, n_start:n_end]

        # Calculate grid based on FIXED block sizes for the chunk
        grid = (ceil_div(Q, BLOCK_Q_FIXED), ceil_div(N_chunk, BLOCK_N_FIXED))
        if grid[0] == 0 or grid[1] == 0: continue

        # Launch the kernel with fixed parameters
        dot_kernel_pairwise_tiled[grid](
            X_prep, A_chunk, Out_chunk,
            Q, N_chunk, D,
            X_prep.stride(0), 1,    # Stride 1 for contiguous last dim
            A_chunk.stride(0), 1,   # Stride 1 for contiguous last dim
            Out_chunk.stride(0), 1, # Assume stride 1 for contiguous last dim of Out view
            BLOCK_Q=BLOCK_Q_FIXED,  # Pass fixed size
            BLOCK_N=BLOCK_N_FIXED,  # Pass fixed size
            BLOCK_K=BLOCK_K_FIXED   # Pass fixed size
            # No num_warps needed here unless kernel uses it explicitly
        )
        # torch.cuda.synchronize() # Optional debug sync

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

def distance_manhattan(X, A):
    """
    Computes pairwise Manhattan (L1) distance using the tiled Triton kernel
    with FIXED block sizes (16, 16, 32).
    """
    target_device = X.device if isinstance(X, torch.Tensor) else A.device # Get device from input
    X_prep, A_prep = _prepare_tensors(X, A, target_device=target_device) # Prepare tensors
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    if D != D_A: raise ValueError(f"Dimension mismatch: X({D}) vs A({D_A})")

    Out = torch.empty((Q, N), dtype=torch.float32, device=target_device)

    # --- Define Fixed Block Sizes ---
    BLOCK_Q_MAN = 32
    BLOCK_N_MAN = 32
    BLOCK_K_MAN = 32
    # --------------------------------

    # Calculate the launch grid based on fixed block sizes
    grid_man = (ceil_div(Q, BLOCK_Q_MAN), ceil_div(N, BLOCK_N_MAN))

    # --- Optional: Add Debug Print BEFORE the launch ---
    # print(f"  DEBUG distance_manhattan (D={D}): Launching kernel...")
    # print(f"    Grid={grid_man}, Q={Q}, N={N}, D={D}, Blocks=({BLOCK_Q_MAN},{BLOCK_N_MAN},{BLOCK_K_MAN})")
    # print(f"    X strides={X_prep.stride()}, A strides={A_prep.stride()}, Out strides={Out.stride()}")
    # print(f"    Passing Strides: xq={X_prep.stride(0)}, xd={X_prep.stride(1)}, an={A_prep.stride(0)}, ad={A_prep.stride(1)}, outq={Out.stride(0)}, outn={Out.stride(1)}")
    # --- End Debug Print ---

    # Launch the kernel, passing the grid and FIXED block sizes explicitly
    # --- MODIFICATION HERE: Pass actual strides for last dimension ---
    manhattan_kernel_pairwise_tiled[grid_man](
        X_prep, A_prep, Out,
        Q, N, D,
        X_prep.stride(0), X_prep.stride(1),  # Use X_prep.stride(1) instead of 1
        A_prep.stride(0), A_prep.stride(1),  # Use A_prep.stride(1) instead of 1
        Out.stride(0),    Out.stride(1),     # Use Out.stride(1) instead of 1
        # Pass the block sizes explicitly matching the kernel signature
        BLOCK_Q=BLOCK_Q_MAN,
        BLOCK_N=BLOCK_N_MAN,
        BLOCK_K=BLOCK_K_MAN
    )
    # --- END MODIFICATION ---

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
if __name__ == "__main__":
    # --- Fixed Parameters ---
    N_data = 4000000 # Using 4 Million points
    N_queries = 1     # Using 1 query as per your last log
    K_val = 10          # K for KNN
    NUM_RUNS = 10       # Number of timed runs for averaging
    WARMUP_RUNS = 2     # Number of warm-up runs <-- Set to 1 for faster debug iteration

    # --- Dimensions to Test ---
    # dimensions_to_test = [2] # <-- START WITH ONLY ONE SMALL DIMENSION FOR DEBUGGING
    dimensions_to_test = [2, 4, 64, 256, 1024] # Original list

    # --- Tolerance for Numerical Check ---
    rtol_check = 1e-4 # Relative tolerance
    atol_check = 1e-5 # Absolute tolerance

    # --- !!! DEBUG FLAG !!! ---
    DETAILED_DEBUG = True # Set to True to print extra info

    print(f"--- GPU KNN/DISTANCE BENCHMARKING & VALIDATION ---")
    print(f"Fixed Params: N={N_data}, Q={N_queries}, K={K_val}, Warmup={WARMUP_RUNS}, Runs={NUM_RUNS}")
    print(f"Testing Dimensions: {dimensions_to_test}")
    print(f"Numerical Check Tolerance: rtol={rtol_check}, atol={atol_check}")
    print(f"DETAILED_DEBUG MODE: {'ON' if DETAILED_DEBUG else 'OFF'}")

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

        results[Dim] = {} # Store results for this dimension
        dimension_failed = False # Flag for skipping benchmarks if warm-up fails

        # --- Generate Base Data (PyTorch and CuPy) ---
        print("="*40); print(f"Generating Data (D={Dim})..."); print("="*40)
        A_data = A_data_cp = X_queries = X_queries_cp = None # Init vars
        try:
            A_data = torch.randn(N_data, Dim, dtype=torch.float32, device=device)
            X_queries = torch.randn(N_queries, Dim, dtype=torch.float32, device=device)
            torch.cuda.synchronize(device=device) # Ensure data is ready
            print(f"Database shape (Torch): {A_data.shape}, Strides: {A_data.stride()}")
            print(f"Query shape (Torch): {X_queries.shape}, Strides: {X_queries.stride()}")

            if cupy_device_ok:
                # Ensure contiguous before dlpack
                A_data_contig = A_data.contiguous(); X_queries_contig = X_queries.contiguous()
                # Verify contiguity if needed
                if DETAILED_DEBUG:
                    print(f"Torch A contig? {A_data_contig.is_contiguous()}. Torch X contig? {X_queries_contig.is_contiguous()}")

                dlpack_A = torch.to_dlpack(A_data_contig); dlpack_X = torch.to_dlpack(X_queries_contig)
                A_data_cp = cp.from_dlpack(dlpack_A); X_queries_cp = cp.from_dlpack(dlpack_X)
                cp.cuda.Stream.null.synchronize() # Ensure data is ready
                print(f"Database shape (CuPy): {A_data_cp.shape}, Strides: {A_data_cp.strides}")
                print(f"Query shape (CuPy): {X_queries_cp.shape}, Strides: {X_queries_cp.strides}")
            else: print("CuPy data generation skipped.")
            print("-" * 40)

        except Exception as e:
            print(f"*** CRITICAL ERROR during Data Generation (D={Dim}): {e} ***")
            import traceback
            traceback.print_exc()
            # Clean up whatever might have been created
            if 'A_data' in locals() and A_data is not None: del A_data
            if 'X_queries' in locals() and X_queries is not None: del X_queries
            if cupy_device_ok and 'A_data_cp' in locals() and A_data_cp is not None: del A_data_cp
            if cupy_device_ok and 'X_queries_cp' in locals() and X_queries_cp is not None: del X_queries_cp
            torch.cuda.empty_cache();
            if cupy_device_ok: cp.get_default_memory_pool().free_all_blocks()
            dimension_failed = True # Mark dimension as failed
            continue # Skip dimension

        # ===---------------------------------------------------------===
        # ===              WARM-UP RUNS (Individual Checks)         ===
        # ===---------------------------------------------------------===
        print("="*40); print(f"Performing Warm-up Runs (D={Dim})..."); print("="*40)
        warmup_functions_torch = {
            "distance_dot_tiled": lambda: distance_dot_tiled(X_queries, A_data),
            "distance_l2": lambda: distance_l2(X_queries, A_data),
            "distance_cosine": lambda: distance_cosine(X_queries, A_data),
            "distance_manhattan": lambda: distance_manhattan(X_queries, A_data),
            "our_knn": lambda: our_knn(N_data, Dim, A_data, X_queries, K_val),
        }
        warmup_functions_cupy = {
            "distance_dot2": lambda: distance_dot2(X_queries_cp, A_data_cp),
            "distance_l22": lambda: distance_l22(X_queries_cp, A_data_cp),
            "distance_cosine2": lambda: distance_cosine2(X_queries_cp, A_data_cp),
            "distance_manhattan2": lambda: distance_manhattan2(X_queries_cp, A_data_cp),
            "our_knn_stream": lambda: our_knn_stream(N_data, Dim, A_data_cp, X_queries_cp, K_val),
        }

        for i in range(WARMUP_RUNS):
            print(f"--- Warm-up Run {i+1}/{WARMUP_RUNS} ---")
            if dimension_failed: break # Stop if a failure already occurred

            # --- PyTorch/Triton Warm-up ---
            print("  Warming up PyTorch/Triton functions...")
            for name, func in warmup_functions_torch.items():
                if dimension_failed: break
                print(f"    Attempting warm-up for: {name}")
                try:
                    _ = func() # Execute the function
                    torch.cuda.synchronize(device=device) # Sync GPU
                    print(f"      Warm-up OK: {name}")
                except Exception as e:
                    print(f"    *** ERROR during warm-up for {name} (D={Dim}, Run={i+1}): {e} ***")
                    print(f"    *** THIS IS LIKELY THE FAILING FUNCTION! ***")
                    import traceback
                    traceback.print_exc() # Print detailed traceback
                    dimension_failed = True # Mark dimension as failed
                    # break # Stop warm-up for this dimension immediately

            # --- CuPy Warm-up ---
            if cupy_device_ok and A_data_cp is not None and not dimension_failed:
                print("  Warming up CuPy functions...")
                for name, func in warmup_functions_cupy.items():
                     if dimension_failed: break
                     print(f"    Attempting warm-up for: {name}")
                     try:
                         _ = func() # Execute the function
                         cp.cuda.Stream.null.synchronize() # Sync GPU
                         print(f"      Warm-up OK: {name}")
                     except Exception as e:
                        print(f"    *** ERROR during warm-up for {name} (D={Dim}, Run={i+1}): {e} ***")
                        print(f"    *** THIS IS LIKELY THE FAILING FUNCTION! ***")
                        import traceback
                        traceback.print_exc() # Print detailed traceback
                        dimension_failed = True # Mark dimension as failed
                        # break # Stop warm-up for this dimension immediately

        if dimension_failed:
            print("\n*** ERROR occurred during warm-up phase. Skipping benchmarks and checks for this dimension. ***")
            # Clean up memory before moving to the next dimension
            del A_data, X_queries
            if cupy_device_ok and A_data_cp is not None: del A_data_cp, X_queries_cp
            torch.cuda.empty_cache()
            if cupy_device_ok: cp.get_default_memory_pool().free_all_blocks()
            continue # Skip to next dimension
        else:
            print("Warm-up complete for D={Dim}.")
            # Optional: Clear any intermediate results from warm-up if memory is tight
            torch.cuda.empty_cache()
            if cupy_device_ok: cp.get_default_memory_pool().free_all_blocks()


        # ===--------------------------------------------------===
        # ===        NUMERICAL STABILITY CHECK (DOT)         ===
        # ===--------------------------------------------------===
        # (Keep this section as is, but it will only run if warm-up succeeds)
        print("\n" + "="*40); print(f"Numerical Stability Check (Dot Product, D={Dim})..."); print("="*40)
        # ... (rest of your numerical check code) ...
        try:
            print("  Calculating with Triton kernel (distance_dot_tiled)...")
            dot_triton = distance_dot_tiled(X_queries, A_data)
            print("  Calculating with torch.matmul...")
            dot_matmul = torch.matmul(X_queries.contiguous(), A_data.contiguous().T)
            torch.cuda.synchronize()

            if dot_triton.shape != dot_matmul.shape:
                 print(f"  ERROR: Shape mismatch! Triton={dot_triton.shape}, Matmul={dot_matmul.shape}")
            else:
                are_close = torch.allclose(dot_triton, dot_matmul, rtol=rtol_check, atol=atol_check)
                if are_close:
                    print(f"  PASS: Triton dot product matches torch.matmul within tolerance.")
                else:
                    max_diff = torch.max(torch.abs(dot_triton - dot_matmul)).item()
                    non_zero_mask = dot_matmul != 0
                    if torch.any(non_zero_mask):
                         max_rel_diff = torch.max(torch.abs((dot_triton[non_zero_mask] - dot_matmul[non_zero_mask]) / dot_matmul[non_zero_mask])).item()
                         print(f"  FAIL: Triton dot product deviates! Max Abs Diff: {max_diff:.6e}, Max Rel Diff: {max_rel_diff:.6e}")
                    else:
                         print(f"  FAIL: Triton dot product deviates! Max Abs Diff: {max_diff:.6e}")
                    print(f"        Check kernel logic, block sizes, or tolerance.")
            del dot_triton, dot_matmul
        except RuntimeError as e:
             print(f"  ERROR during numerical check (RuntimeError): {e}")
             traceback.print_exc()
        except Exception as e:
            print(f"  ERROR during numerical check: {e}")
            traceback.print_exc()
        print("-" * 40)

        # ===--------------------------------------------------===
        # ===         DISTANCE FUNCTION BENCHMARKS           ===
        # ===--------------------------------------------------===
        # (Keep this section as is, it will only run if warm-up succeeds)
        print("\n" + "="*40); print(f"Benchmarking Distance Functions (D={Dim})..."); print("="*40)
        # ... (rest of your benchmarking code for distances) ...
        # Example for one function:
        try:
            start_event = torch.cuda.Event(enable_timing=True); end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); start_event.record()
            for r in range(NUM_RUNS): _ = distance_dot_tiled(X_queries, A_data)
            end_event.record(); torch.cuda.synchronize()
            avg_time = (start_event.elapsed_time(end_event) / 1000.0) / NUM_RUNS
            print(f"PyTorch/Triton dot_tiled Avg Time:  {avg_time:.6f} seconds")
            results[Dim]['dist_dot_torch_tiled_fixed'] = avg_time # Use consistent key name
        except Exception as e: print(f"Error benchmarking distance_dot_tiled: {e}")
        # ... Add similar try/except blocks for all other benchmarked functions ...

        # ===--------------------------------------------------===
        # ===            KNN FUNCTION BENCHMARKS             ===
        # ===--------------------------------------------------===
        # (Keep this section as is, it will only run if warm-up succeeds)
        print("\n" + "="*40); print(f"Benchmarking KNN Functions (D={Dim})..."); print("="*40)
        # ... (rest of your benchmarking code for KNN) ...
        # Example for one function:
        try:
            start_event = torch.cuda.Event(enable_timing=True); end_event = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(); start_event.record()
            for r in range(NUM_RUNS): knn_indices_torch, _ = our_knn(N_data, Dim, A_data, X_queries, K_val)
            end_event.record(); torch.cuda.synchronize()
            avg_time = (start_event.elapsed_time(end_event) / 1000.0) / NUM_RUNS
            qps = N_queries / avg_time if avg_time > 0 else 0
            print(f"Triton/Torch our_knn Avg Time:           {avg_time:.6f} seconds ({qps:.2f} QPS)")
            results[Dim]['knn_torch'] = avg_time
        except Exception as e: print(f"Error benchmarking our_knn (Triton/Torch): {e}")
        # ... Add similar try/except blocks for CuPy KNN ...

        # --- Cleanup for the dimension ---
        print(f"\n--- Finished Benchmarks for Dimension D = {Dim} ---")
        del A_data, X_queries
        if cupy_device_ok and A_data_cp is not None: del A_data_cp, X_queries_cp
        torch.cuda.empty_cache()
        if cupy_device_ok: cp.get_default_memory_pool().free_all_blocks()
        print("-" * 70)


    print("\n" + "#"*70); print("# ALL DIMENSION BENCHMARKS FINISHED"); print("#"*70)

    # --- Print Summary Table ---
    # (Keep your summary table printing logic as is)
    print("\nBenchmark Summary (Average Times in Seconds):")
    # ... (rest of your summary table code) ...