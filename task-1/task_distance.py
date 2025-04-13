import torch
import cupy as cp
import triton
import triton.language as tl
import numpy as np
import time
import scipy.spatial.distance
import json
# Removed testdata imports as they are not used in the remaining code
import csv
import os
import math

# Check CUDA availability
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
DEFAULT_BLOCK_D = 512 # Default block size for simple kernels if needed
DEFAULT_BLOCK_K = 16 # Block size for the reduction dimension D (used by tiled kernels)

def ceil_div(a, b):
    return (a + b - 1) // b

# --- Optimized Tiled Dot Product Kernel ---
# Note: This kernel is used by distance_dot_tiled, which is then used by distance_l2 and distance_cosine.
@triton.jit
def dot_kernel_pairwise(
    X_ptr, A_ptr, Out_ptr,
    Q, N, D,
    stride_xq, stride_xd, # stride_xd will be 1 from call site
    stride_an, stride_ad, # stride_ad will be 1 from call site
    stride_outq, stride_outn, # stride_outn will be 1 from call site
    BLOCK_SIZE_D: tl.constexpr,
):
    """Calculates pairwise dot product using float32. Loops over D."""
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


# --- Manhattan Distance Kernel ---
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
        return prepared     # Return list for multiple tensors
    # --- End Corrected Return Logic ---

# ============================================================================
# Python Distance Function Wrappers using Triton / PyTorch
# ============================================================================

# Note: This function is kept mainly for comparison or simpler cases.
# distance_dot_tiled is generally preferred for performance.
def distance_dot(X, A):
    """Computes pairwise dot product using non-tiled Triton kernel (dot_kernel_pairwise)."""
    # _prepare_tensors ensures float32, contiguous inputs
    X_prep, A_prep = _prepare_tensors(X, A)
    Q, D = X_prep.shape
    N, D_A = A_prep.shape
    assert D == D_A, f"Dimension mismatch: X({D}) vs A({D_A})"

    # Output dtype is float32
    Out = torch.empty((Q, N), dtype=torch.float32, device=device)
    grid = (Q, N)

    # Ensure DEFAULT_BLOCK_D is defined (e.g., 512 or tuned value)
    global DEFAULT_BLOCK_D
    if 'DEFAULT_BLOCK_D' not in globals(): DEFAULT_BLOCK_D = 512 # Define if needed

    # Call the kernel
    dot_kernel_pairwise[grid](
        X_prep, A_prep, Out,
        Q, N, D,
        X_prep.stride(0), 1, # Use stride 1 for contiguous last dim
        A_prep.stride(0), 1, # Use stride 1 for contiguous last dim
        Out.stride(0),    1, # Use stride 1 for contiguous last dim
        BLOCK_SIZE_D=DEFAULT_BLOCK_D
    )
    return Out

# Assume ceil_div, _prepare_tensors are defined correctly
# Assume dot_kernel_pairwise kernel is defined correctly
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
    BLOCK_SIZE_D_FIXED = 32 # Potentially tune this
    NUM_WARPS_FIXED = 2     # Potentially tune this
    # -------------------------------------------

    # print(f"Tiling simple dot kernel (Fixed D={BLOCK_SIZE_D_FIXED}, Warps={NUM_WARPS_FIXED}) N_TILE={N_TILE}")
    for n_start in range(0, N, N_TILE):
        n_end = min(n_start + N_TILE, N)
        N_chunk = n_end - n_start
        if N_chunk <= 0: continue

        # Slicing along the first dim (A_chunk) preserves contiguity usually.
        # Slicing the output view might not, hence we pass strides explicitly.
        A_chunk = A_prep[n_start:n_end, :]
        Out_chunk = Out[:, n_start:n_end]

        grid = (Q, N_chunk)
        if grid[0] == 0 or grid[1] == 0: continue

        # --- Launch Kernel with ACTUAL strides for Out_chunk ---
        dot_kernel_pairwise[grid](
            X_prep, A_chunk, Out_chunk,
            Q, N_chunk, D,
            X_prep.stride(0), 1,               # Stride for X (stride_xq, stride_xd=1)
            A_chunk.stride(0), 1,              # Stride for A_chunk (stride_an, stride_ad=1)
            Out_chunk.stride(0), Out_chunk.stride(1), # Stride for Out_chunk (stride_outq, stride_outn)
            BLOCK_SIZE_D=BLOCK_SIZE_D_FIXED,
            num_warps=NUM_WARPS_FIXED
        )
        # -------------------------------------------------------

    return Out


# Corrected distance_l2 (Uses distance_dot_tiled)
def distance_l2(X, A):
    """
    Computes pairwise SQUARED L2 distances using the tiled dot product kernel.
    dist^2 = ||x||^2 + ||a||^2 - 2*dot(x,a)
    """
    X_prep, A_prep = _prepare_tensors(X, A) # Prepare tensors internally
    Q, D = X_prep.shape; N, D_A = A_prep.shape
    if D != D_A: raise ValueError(f"Dimension mismatch: X({D}) vs A({D_A})")

    # Call distance_dot_tiled, tensors are prepared so prep=False
    dot_products = distance_dot_tiled(X_prep, A_prep, prep=False) # Shape (Q, N)

    X_norm_sq = torch.sum(X_prep**2, axis=1, keepdims=True)  # Shape (Q, 1)
    A_norm_sq = torch.sum(A_prep**2, axis=1, keepdims=True)  # Shape (N, 1)
    dist_sq = X_norm_sq + A_norm_sq.T - 2 * dot_products # Shape (Q, N)
    dist_sq.clamp_(min=0.0) # Ensure non-negativity due to potential floating point errors
    return dist_sq

# Corrected distance_cosine (Uses distance_dot_tiled)
def distance_cosine(X, A, epsilon=1e-8):
    """
    Computes pairwise Cosine distances (1 - similarity) using the tiled dot product kernel.
    cos_sim = dot(x,a) / (||x|| * ||a||)
    cos_dist = 1 - cos_sim
    """
    target_device = X.device if isinstance(X, torch.Tensor) else A.device
    X_prep, A_prep = _prepare_tensors(X, A, target_device=target_device)
    Q, D = X_prep.shape; N, D_A = A_prep.shape
    if D != D_A: raise ValueError(f"Dimension mismatch: X({D}) vs A({D_A})")

    # Call distance_dot_tiled, tensors are prepared so prep=False
    dot_products = distance_dot_tiled(X_prep, A_prep, prep=False) # (Q, N)

    X_norm = torch.linalg.norm(X_prep, axis=1, keepdims=True) # (Q, 1)
    A_norm = torch.linalg.norm(A_prep, axis=1, keepdims=True) # (N, 1)
    # Add epsilon for numerical stability BEFORE multiplying norms
    norm_product = (X_norm + epsilon) * (A_norm.T + epsilon) # (Q, N)
    cosine_similarity = dot_products / norm_product
    cosine_similarity.clamp_(min=-1.0, max=1.0) # Clamp results to valid range
    cosine_distance = 1.0 - cosine_similarity
    return cosine_distance

# Corrected distance_manhattan (Uses simple manhattan kernel with tiling wrapper)
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
    BLOCK_D_PARAM = 128 # Example value, might need tuning

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

        # Launch the simple kernel for THIS CHUNK
        manhattan_kernel_pairwise_simple[grid](
            X_prep, A_chunk, Out_chunk,
            Q, N_chunk, D,
            X_prep.stride(0), X_prep.stride(1),      # Strides for X
            A_chunk.stride(0), A_chunk.stride(1),    # Strides for A_chunk
            Out_chunk.stride(0), Out_chunk.stride(1),# Strides for Out_chunk view
            BLOCK_SIZE_D=BLOCK_D_PARAM
        )

    torch.cuda.synchronize(device=target_device) # Sync after all chunks are launched
    return Out

# ============================================================================
# Python Distance Function Wrappers using CuPy
# ============================================================================

def distance_dot2(X, Y): # Pairwise Dot Product using CuPy matmul
    """ Calculates pairwise dot products between rows of X and rows of Y using CuPy matmul. """
    X_cp = cp.asarray(X, dtype=cp.float32)
    Y_cp = cp.asarray(Y, dtype=cp.float32)
    if X_cp.ndim == 1: X_cp = X_cp[None, :] # Ensure X is 2D (Q, D)
    if Y_cp.ndim == 1: Y_cp = Y_cp[None, :] # Ensure Y is 2D (N, D)
    if X_cp.shape[1] != Y_cp.shape[1]: raise ValueError("Dimension mismatch for dot product")
    # (Q, D) @ (D, N) -> (Q, N)
    return cp.matmul(X_cp, Y_cp.T)

def pairwise_l2_squared_cupy(X_cp, C_cp):
    """ Computes pairwise SQUARED L2 distances using CuPy. """
    if not isinstance(X_cp, cp.ndarray): X_cp = cp.asarray(X_cp, dtype=cp.float32)
    elif X_cp.dtype != cp.float32: X_cp = X_cp.astype(cp.float32)
    if not isinstance(C_cp, cp.ndarray): C_cp = cp.asarray(C_cp, dtype=cp.float32)
    elif C_cp.dtype != cp.float32: C_cp = C_cp.astype(cp.float32)
    if X_cp.ndim == 1: X_cp = X_cp[None, :]
    if C_cp.ndim == 1: C_cp = C_cp[None, :]
    if X_cp.shape[1] != C_cp.shape[1]: raise ValueError("Dimension mismatch")

    X_norm_sq = cp.einsum('ij,ij->i', X_cp, X_cp)[:, cp.newaxis] # (Q, 1)
    C_norm_sq = cp.einsum('ij,ij->i', C_cp, C_cp)[cp.newaxis, :] # (1, N)
    # Use matmul for dot product: (Q, D) @ (D, N) -> (Q, N)
    dot_products = cp.matmul(X_cp, C_cp.T)
    dist_sq = X_norm_sq - 2 * dot_products + C_norm_sq
    return cp.maximum(0.0, dist_sq) # Ensure non-negativity

def distance_l22(X, Y): # Pairwise SQUARED L2 using CuPy helper
    """ Calculates pairwise SQUARED L2 distance using CuPy. """
    return pairwise_l2_squared_cupy(X, Y)

def distance_cosine2(X, Y, epsilon=1e-8): # Pairwise Cosine Distance using CuPy
    """ Calculates pairwise cosine distance (1 - similarity) using CuPy. """
    X_cp = cp.asarray(X, dtype=cp.float32)
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
    distance = cp.maximum(0.0, 1.0 - cosine_similarity) # Ensure non-negative distance
    return distance

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
            try:
                abs_diff_tile = cp.abs(X_chunk_q[:, None, :] - Y_chunk_n[None, :, :])
                l1_distance_tile = cp.sum(abs_diff_tile, axis=2) # Shape (curr_Q, curr_N)
            except cp.cuda.memory.OutOfMemoryError:
                print(f"\n--- OOM Error within Manhattan tile (D={D}, Tile={curr_Q}x{curr_N}) ---")
                print(f"--- Try reducing Q_TILE/N_TILE in distance_manhattan2 definition ---")
                # Fill problematic tile with Inf and continue if possible, or re-raise
                l1_distance[q_start:q_end, n_start:n_end] = cp.inf
                cp.get_default_memory_pool().free_all_blocks() # Attempt cleanup
                continue # Skip this tile, maybe others work

            # Store result in the output matrix slice
            l1_distance[q_start:q_end, n_start:n_end] = l1_distance_tile

            # Clean up intermediate tile explicitly (optional, helps memory management)
            del abs_diff_tile, l1_distance_tile

    return l1_distance


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
# Main Execution Block (Benchmarking Distances across Dimensions)
# ============================================================================
if __name__ == "__main__":
    # --- Fixed Parameters ---
    N_data = 100000 # Using 4 Million points
    N_queries = 1     # Using 1 query
    # K_val removed as KNN code is gone
    NUM_RUNS = 4      # Number of timed runs for averaging
    WARMUP_RUNS = 1   # Number of warm-up runs
    # --- CPU BENCHMARKING FLAG ---
    BENCHMARK_CPU = False# Set to False to skip CPU tests (can be slow)

    # --- Dimensions to Test ---
    dimensions_to_test = [2,4, 64, 256, 1024, 8192, 32768] # Example dimension

    # --- Tolerance for Numerical Check ---
    rtol_check = 1e-4
    atol_check = 1e-5

    # --- DEBUG FLAG ---
    DETAILED_DEBUG = False # Set to True for extra info during GPU gen/warmup

    print(f"--- GPU & CPU DISTANCE BENCHMARKING & VALIDATION ---") # Updated title
    print(f"Fixed Params: N={N_data}, Q={N_queries}, Warmup={WARMUP_RUNS}, Runs={NUM_RUNS}")
    print(f"Testing Dimensions: {dimensions_to_test}")
    print(f"Benchmark CPU: {BENCHMARK_CPU}")
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
            print(f"*** CRITICAL ERROR during Data Generation/Transfer (D={Dim}): {e} ***")
            import traceback; traceback.print_exc()
            dimension_failed = True
            # Clean up CPU arrays too
            if 'A_data_np' in locals() and A_data_np is not None: del A_data_np
            if 'X_queries_np' in locals() and X_queries_np is not None: del X_queries_np
            # Torch/CuPy cleanup happens later
            continue # Skip to next dimension

        # ===---------------------------------------------------------===
        # ===             WARM-UP RUNS (Individual Checks)            ===
        # ===---------------------------------------------------------===
        print("="*40); print(f"Performing Warm-up Runs (D={Dim})..."); print("="*40)
        # Define the functions to warm up
        warmup_functions_torch = {
            "distance_dot_tiled": lambda: distance_dot_tiled(X_queries, A_data),
            "distance_l2": lambda: distance_l2(X_queries, A_data),
            "distance_cosine": lambda: distance_cosine(X_queries, A_data),
            "distance_manhattan": lambda: distance_manhattan(X_queries, A_data),
            # KNN function removed
        }
        warmup_functions_cpu = {} # CPU NumPy/SciPy
        if BENCHMARK_CPU and A_data_np is not None:
            warmup_functions_cpu = {
                "distance_dot_cpu": lambda: distance_dot_cpu(X_queries_np, A_data_np),
                "distance_l2_squared_cpu": lambda: distance_l2_squared_cpu(X_queries_np, A_data_np),
                "distance_cosine_cpu": lambda: distance_cosine_cpu(X_queries_np, A_data_np),
                "distance_manhattan_cpu": lambda: distance_manhattan_cpu(X_queries_np, A_data_np),
                # KNN function removed
            }
        if cupy_device_ok and A_data_cp is not None:
             warmup_functions_cupy = {
                 "distance_dot2": lambda: distance_dot2(X_queries_cp, A_data_cp),
                 "distance_l22": lambda: distance_l22(X_queries_cp, A_data_cp),
                 "distance_cosine2": lambda: distance_cosine2(X_queries_cp, A_data_cp),
                 "distance_manhattan2": lambda: distance_manhattan2(X_queries_cp, A_data_cp),
                 # KNN function removed
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
        # ===       BENCHMARKING (Skip if warm-up failed)      ===
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
                    print(f"CuPy distance_l22 Avg Time:          {avg_time:.6f} seconds")
                    results[Dim]['dist_l2_cupy'] = avg_time
                except Exception as e: print(f"Error benchmarking distance_l22: {e}")
                try:
                    start_event=cp.cuda.Event();end_event=cp.cuda.Event();cp.cuda.Stream.null.synchronize();start_event.record()
                    for r in range(NUM_RUNS): _ = distance_cosine2(X_queries_cp, A_data_cp)
                    end_event.record();end_event.synchronize();avg_time=(cp.cuda.get_elapsed_time(start_event,end_event)/1000.0)/NUM_RUNS
                    print(f"CuPy distance_cosine2 Avg Time:      {avg_time:.6f} seconds")
                    results[Dim]['dist_cos_cupy'] = avg_time
                except Exception as e: print(f"Error benchmarking distance_cosine2: {e}")
                try:
                    start_event=cp.cuda.Event();end_event=cp.cuda.Event();cp.cuda.Stream.null.synchronize();start_event.record()
                    for r in range(NUM_RUNS): _ = distance_manhattan2(X_queries_cp, A_data_cp)
                    end_event.record();end_event.synchronize();avg_time=(cp.cuda.get_elapsed_time(start_event,end_event)/1000.0)/NUM_RUNS
                    if cp.isinf(avg_time): print("CuPy distance_manhattan2 likely OOM occurred.")
                    else: print(f"CuPy distance_manhattan2 Avg Time:   {avg_time:.6f} seconds")
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

            # --- KNN Function Benchmarks Removed ---

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


    print("\n" + "#"*70); print("# ALL DIMENSION BENCHMARKS FINISHED"); print("#"*70)

    # --- Print Summary Table ---
    print("\nBenchmark Summary (Average Times in Seconds):")
    # Define column order dynamically based on results collected + preference
    col_order = [
        # Torch Distances
        'dist_dot_torch_tiled', 'dist_l2_torch', 'dist_cos_torch', 'dist_man_torch',
        # CuPy Distances
        'dist_dot_cupy', 'dist_l2_cupy', 'dist_cos_cupy', 'dist_man_cupy',
        # CPU Distances
        'dist_dot_cpu', 'dist_l2_cpu', 'dist_cos_cpu', 'dist_man_cpu',
        # KNN columns removed
    ]

    # Find all columns present in the results
    present_cols = set()
    for d in results: present_cols.update(results[d].keys())

    # Create the final column list: preferred order first, then any others
    final_cols = [col for col in col_order if col in present_cols]
    for col in sorted(present_cols):
        if col not in final_cols: final_cols.append(col) # Add any extras alphabetically

    col_width = 25 # Adjust spacing between columns
    header = f"{'Dim':<6}"
    for col_key in final_cols: header += f"{col_key:<{col_width}}"

    # Recalculate width based on final columns
    table_width = 6 + len(final_cols) * col_width
    print("-" * table_width)
    print(header)
    print("-" * table_width)

    for Dim in dimensions_to_test:
        row = f"{Dim:<6}"
        if Dim in results:
            r = results[Dim]
            for col_key in final_cols:
                # Format the time value, use float('nan') if key is missing
                time_val = r.get(col_key, float('nan'))
                row += f"{time_val:<{col_width}.6f}"

            # Adjust N/A string generation based on col_width
            na_spacing_len = (col_width - 3) // 2 # Length of spaces around N/A
            na_spacing = ' ' * na_spacing_len
            na_string = f"{na_spacing}N/A{na_spacing}"
            if (col_width - 3) % 2 != 0: na_string += " " # Add extra space if needed for centering

            # Replace NaN representation with the centered N/A string
            # Need to be careful with floating point representation of NaN
            nan_rep = f"{float('nan'):<{col_width}.6f}" # How NaN is formatted
            row = row.replace(nan_rep, na_string)

            print(row)
        else:
            # If dimension was skipped entirely (e.g., data gen failed)
            skipped_spacing_len = (col_width - 7) // 2 # Length of spaces around Skipped
            skipped_spacing = ' ' * skipped_spacing_len
            skipped_string = f"{skipped_spacing}Skipped{skipped_spacing}"
            if (col_width - 7) % 2 != 0: skipped_string += " "

            for _ in final_cols: row += skipped_string
            print(row)
    print("-" * table_width)