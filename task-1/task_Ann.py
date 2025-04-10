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
import cupy as cp
# (Keeping the existing K-Means implementation using its specific Triton kernel)
DEFAULT_BLOCK_D = 512
DEFAULT_BLOCK_K = 16


if not torch.cuda.is_available():
    print("CUDA not available, exiting.")
    exit()
device = torch.device("cuda:0")
print(f"Using device: {device}")
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
def kmeans_assign_kernel_cosine( # Renamed for clarity
    A_norm_ptr,           # Pointer to L2-NORMALIZED data points (N, D)
    centroids_norm_ptr,   # Pointer to L2-NORMALIZED centroids (K, D)
    assignments_ptr,      # Pointer to output assignments (N,)
    N, D, K,
    stride_an, stride_ad,   # Strides for NORMALIZED A
    stride_ck, stride_cd,   # Strides for NORMALIZED centroids
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K_CHUNK: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Assigns each point in A to the centroid with the highest Cosine Similarity.
    ASSUMES A_norm_ptr and centroids_norm_ptr point to L2-NORMALIZED vectors.
    Cosine Similarity = Dot Product for normalized vectors.
    """
    # --- Program ID and Offsets (Same as before) ---
    pid_n_block = tl.program_id(axis=0) # ID for the N dimension block
    offs_n = pid_n_block * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N) # Range for points in block
    mask_n = offs_n < N # Mask for points in this block

    # --- Initialization for Maximum Similarity Search ---
    # Initialize max similarity found so far for each point in the block.
    # Cosine similarity ranges from -1 to 1. Initialize to a value lower than -1.
    max_similarity = tl.full((BLOCK_SIZE_N,), -2.0, dtype=tl.float32)
    # Initialize best assignment for each point in the block.
    best_assignment = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)

    # --- Loop through Chunks of Centroids (Same as before) ---
    for k_start in range(0, K, BLOCK_SIZE_K_CHUNK):
        k_end = tl.minimum(k_start + BLOCK_SIZE_K_CHUNK, K)

        # --- Loop through Centroids within the Chunk (Same as before) ---
        for k_idx in range(BLOCK_SIZE_K_CHUNK):
            actual_k = k_start + k_idx
            # Only process if the centroid index is valid for this chunk
            if actual_k < k_end:

                # --- Calculate Dot Product (Cosine Similarity) for current centroid ---
                # Initialize dot product accumulator for the current points block vs actual_k centroid
                current_dot_product = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

                # --- Loop through Dimensions to compute Dot Product ---
                for d_start in range(0, D, BLOCK_SIZE_D):
                    offs_d = d_start + tl.arange(0, BLOCK_SIZE_D)
                    mask_d = offs_d < D # Mask for valid dimensions in this block

                    # Load NORMALIZED centroid block for the current dimension chunk
                    # Shape: (BLOCK_SIZE_D,)
                    centroid_d_ptr = centroids_norm_ptr + actual_k * stride_ck + offs_d * stride_cd
                    # Need boundary check for D dimension
                    centroid_vals = tl.load(centroid_d_ptr, mask=mask_d, other=0.0)

                    # Load NORMALIZED points block for the current dimension chunk
                    # Shape: (BLOCK_SIZE_N, BLOCK_SIZE_D)
                    points_d_ptr = A_norm_ptr + offs_n[:, None] * stride_an + offs_d[None, :] * stride_ad
                    # Need boundary checks for both N and D dimensions
                    points_vals = tl.load(points_d_ptr, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

                    # Accumulate dot product component for this dimension block
                    # points_vals * centroid_vals[None, :] performs element-wise multiplication broadcasted
                    # tl.sum(..., axis=1) sums across the D dimension block for each point in N
                    current_dot_product += tl.sum(points_vals * centroid_vals[None, :], axis=1)
                # End of dimension loop: current_dot_product now holds the full dot product (Cosine Similarity)

                # --- Compare and Update Assignment ---
                # Check if the similarity with the current centroid (actual_k) is higher than the max found so far
                is_more_similar = current_dot_product > max_similarity

                # Update max_similarity where current centroid is better
                max_similarity = tl.where(is_more_similar, current_dot_product, max_similarity)
                # Update best_assignment where current centroid is better
                best_assignment = tl.where(is_more_similar, actual_k, best_assignment)
            # End if actual_k < k_end
        # End loop k_idx
    # End loop k_start

    # --- Store Results (Same as before) ---
    assignments_out_ptrs = assignments_ptr + offs_n
    tl.store(assignments_out_ptrs, best_assignment, mask=mask_n)
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

    return -Out

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

    dot_products = distance_dot_tiled(X_prep, A_prep, **kwargs) # (Q, N)
    X_norm = torch.linalg.norm(X_prep, axis=1, keepdims=True) # (Q, 1)
    A_norm = torch.linalg.norm(A_prep, axis=1, keepdims=True) # (N, 1)
    norm_product = X_norm * A_norm.T # (Q, N)
    cosine_similarity = dot_products / (norm_product + epsilon)
    cosine_similarity.clamp_(min=-1.0, max=1.0)
    cosine_distance = 1.0 - cosine_similarity
    return cosine_distance

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
    A_norm = normalize_vectors(A) # Using your normalize_vectors function
    centroids_norm = normalize_vectors(centroids)

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
        kmeans_assign_kernel_cosine[grid_assign](
        A_norm, centroids_norm, assignments_int32, # Pass normalized tensors
        N_A, D, K,
        A_norm.stride(0), A_norm.stride(1),
        centroids_norm.stride(0), centroids_norm.stride(1),
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
    all_distances = distance_cosine(X_prep, A_prep, prep = False) # Shape (Q, N_A)

    # 2. Find the top K smallest distances for each query
    #    largest=False gives smallest distances (nearest neighbors)
    topk_distances, topk_indices = torch.topk(all_distances, k=K, dim=1, largest=False)

    end_time = time.time()
    print(f"k-NN computation time: {end_time - start_time:.6f} seconds")

    return topk_indices, topk_distances

# --- Normalization Helper ---
def normalize_vectors(vectors, epsilon=1e-12):
    """L2 normalize vectors row-wise."""
    norms = torch.linalg.norm(vectors, dim=1, keepdim=True)
    return vectors / (norms + epsilon)

# ============================================================================
# Task 2.2: Approximate Nearest Neighbors (Optimized K-Means IVF - Dot Product)
# ============================================================================
class KMeansANN_Optimized:
    """
    Optimized Approximate Nearest Neighbor search using K-Means clustering (IVF)
    with Dot Product similarity on normalized vectors and a GPU-friendly index structure.

    Builds an index by clustering data points (using L2 K-Means).
    Search involves finding the most similar cluster centroid(s) via dot product,
    then performing exact dot product search within candidates from those clusters.
    """
    def __init__(self, dim, k_clusters, nprobe=1):
        """
        Args:
            dim (int): Dimensionality of the vectors.
            k_clusters (int): Number of clusters for K-Means.
            nprobe (int): Number of nearest clusters (by dot product) to search.
        """
        self.dim = dim
        self.k_clusters = k_clusters
        self.nprobe = max(1, nprobe) # Ensure nprobe >= 1
        self.centroids_norm = None       # Shape: (k_clusters, dim), L2 normalized
        self.sorted_data_norm = None     # Shape: (N, dim), sorted by cluster, L2 normalized
        self.cluster_offsets = None      # Shape: (k_clusters,), start index of each cluster in sorted_data
        self.cluster_counts = None       # Shape: (k_clusters,), count of points in each cluster
        self.sorted_to_original_idx = None # Shape: (N,), mapping from sorted index back to original A index
        self.is_built = False
        self.original_n = 0
        print(f"Initialized KMeansANN_Optimized (Dot Product): k_clusters={k_clusters}, nprobe={nprobe}")

    def build_index(self, A, max_kmeans_iters=50, kmeans_tol=1e-4):
        """
        Builds the optimized K-Means based index using Dot Product similarity.

        Args:
            A (torch.Tensor): The database vectors (N, dim) on GPU.
            max_kmeans_iters (int): Max iterations for K-Means.
            kmeans_tol (float): Tolerance for K-Means convergence.
        """
        N_A, D = A.shape
        self.original_n = N_A
        assert D == self.dim, f"Data dimension ({D}) doesn't match index dimension ({self.dim})"
        assert self.k_clusters <= N_A, f"k_clusters ({self.k_clusters}) cannot be larger than N_A ({N_A})"
        # nprobe check already done in __init__

        print("Building Optimized KMeansANN index (Dot Product)...")
        start_build = time.time()
        target_device = A.device

        # 1. Run K-Means (still using L2 distance internally for clustering)
        print(f"  Running K-Means (L2 based) with K={self.k_clusters}...")
        # our_kmeans uses the triton kernel internally
        centroids, assignments = our_kmeans(N_A, self.dim, A, self.k_clusters,
                                                  max_iters=max_kmeans_iters, tol=kmeans_tol)
        # centroids shape: (k_clusters, dim)
        # assignments shape: (N_A,) dtype=int64

        # 2. Normalize Centroids
        print("  Normalizing centroids...")
        self.centroids_norm = normalize_vectors(centroids).contiguous()

        # 3. Sort data based on cluster assignments for GPU-friendly access
        print("  Sorting data by cluster assignment...")
        start_sort = time.time()
        # Argsort gives the indices that would sort the assignments array
        sorted_indices = torch.argsort(assignments)
        # Keep track of mapping from sorted position back to original index
        self.sorted_to_original_idx = sorted_indices.clone() # Store this mapping

        # Sort the original data and the assignments array
        sorted_A = A[sorted_indices]
        sorted_assignments = assignments[sorted_indices]
        end_sort = time.time()
        print(f"  Data sorting time: {end_sort - start_sort:.4f}s")

        # 4. Normalize the sorted data
        print("  Normalizing sorted data...")
        self.sorted_data_norm = normalize_vectors(sorted_A).contiguous()

        # 5. Create GPU-based cluster offsets and counts
        print("  Calculating cluster offsets and counts...")
        start_offset = time.time()

        # --- CORRECTED SECTION ---
        # Find unique cluster IDs present and their counts using unique_consecutive
        # It returns 2 values when return_counts=True
        unique_clusters, counts = torch.unique_consecutive(
            sorted_assignments, return_counts=True
        )

        # Calculate the indices where each unique consecutive block starts
        # The first block starts at index 0. Subsequent blocks start after the previous block ends.
        first_occurrence_indices = torch.zeros_like(unique_clusters, dtype=torch.long) # Use long for indices
        # Calculate cumulative sum of counts *excluding* the last one, then offset by 1 position
        # Prepend a 0 for the start index of the very first block
        if counts.numel() > 1:
             # Compute offsets based on counts of preceding blocks
             cumulative_counts = torch.cumsum(counts[:-1], dim=0)
             # First index is 0, subsequent indices are the cumulative counts
             first_occurrence_indices[1:] = cumulative_counts
        elif counts.numel() == 0:
            # Handle edge case of empty assignments (shouldn't happen if N_A > 0)
            first_occurrence_indices = torch.empty(0, dtype=torch.long, device=target_device)
        # If counts.numel() == 1, first_occurrence_indices remains [0], which is correct.

        # --- End of Correction ---

        # Create full offset and count arrays (handling potentially empty clusters)
        # Using N_A as the offset sentinel might be slightly cleaner than -1
        self.cluster_offsets = torch.full((self.k_clusters,), self.original_n, dtype=torch.long, device=target_device)
        self.cluster_counts = torch.zeros(self.k_clusters, dtype=torch.long, device=target_device)

        # Fill in the values for clusters that actually have points
        # Ensure unique_clusters are within the expected range [0, k_clusters-1] before using them as indices
        valid_cluster_mask = (unique_clusters >= 0) & (unique_clusters < self.k_clusters)
        valid_unique_clusters = unique_clusters[valid_cluster_mask]

        # Filter the calculated indices and counts to match the valid clusters
        valid_first_indices = first_occurrence_indices[valid_cluster_mask]
        valid_counts = counts[valid_cluster_mask]

        # Use scatter_ for safe assignment, especially if valid_unique_clusters might somehow
        # contain duplicates (though unique_consecutive shouldn't cause this)
        # Or use direct assignment which is likely fine here:
        # self.cluster_offsets[valid_unique_clusters] = valid_first_indices
        # self.cluster_counts[valid_unique_clusters] = valid_counts
        if valid_unique_clusters.numel() > 0: # Check if there are any valid clusters to update
             self.cluster_offsets.scatter_(0, valid_unique_clusters, valid_first_indices)
             self.cluster_counts.scatter_(0, valid_unique_clusters, valid_counts)

        end_offset = time.time()
        print(f"  Offset calculation time: {end_offset - start_offset:.4f}s")

        # Cleanup intermediate tensors if needed
        del sorted_A, sorted_assignments, assignments, centroids
        # Also cleanup tensors created in this step
        del unique_clusters, counts, first_occurrence_indices, valid_cluster_mask, valid_unique_clusters, valid_first_indices, valid_counts
        if 'cumulative_counts' in locals(): del cumulative_counts # Only delete if it was created
        torch.cuda.empty_cache() # Optional aggressive cleanup

        self.is_built = True
        end_build = time.time()
        print(f"Index build time: {end_build - start_build:.2f} seconds")


    def search_knn(self, query_vec, k):
        """
        Searches for the k most similar neighbors (highest dot product)
        for a single query vector using the optimized IVF index.

        Args:
            query_vec (torch.Tensor): The query vector (1, dim) or (dim,) on GPU.
            k (int): The number of nearest neighbors to find.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - topk_indices (torch.Tensor): Original indices of the K most similar neighbors (k,).
                - topk_similarities (torch.Tensor): Dot product similarity scores (k,).
                                                     Returns dot product values (higher is better).
        """
        if not self.is_built:
            raise RuntimeError("Index has not been built yet. Call build_index() first.")

        k = min(k, self.original_n) # Cannot return more neighbors than exist
        if k == 0:
            return torch.full((0,), -1, dtype=torch.int64), torch.full((0,), -float('inf'), dtype=torch.float32)


        target_device = self.centroids_norm.device
        query_vec = query_vec.view(1, -1) # Ensure shape (1, dim)
        query_vec_prep, = _prepare_tensors(query_vec, target_device=target_device)

        # 1. Normalize the query vector
        query_norm = normalize_vectors(query_vec_prep).contiguous() # Shape (1, dim)

        # 2. Find the `nprobe` most similar centroids (highest dot product)
        # Use distance_dot_triton: query_norm @ self.centroids_norm.T
        # Result shape: (1, k_clusters)
        dot_q_to_centroids = distance_cosine(query_norm, self.centroids_norm)

        # Get the indices and dot products of the top `nprobe` most similar centroids
        # We want the LARGEST dot products
        # If less than nprobe centroids exist/are valid, topk handles it.
        actual_nprobe = min(self.nprobe, self.k_clusters)
        top_centroid_dots, top_centroid_indices = torch.topk(
            dot_q_to_centroids, k=actual_nprobe, dim=1, largest=True, sorted=False # Don't need them sorted
        )
        top_centroid_indices = top_centroid_indices.squeeze(0) # Shape (nprobe,)

        # 3. Identify candidate indices from the selected clusters using offsets/counts
        candidate_sorted_indices = []
        valid_counts = self.cluster_counts[top_centroid_indices]
        valid_offsets = self.cluster_offsets[top_centroid_indices]

        total_candidates = 0
        indices_to_fetch = []
        for i in range(actual_nprobe):
            count = valid_counts[i].item()
            offset = valid_offsets[i].item()
            if count > 0 and offset != -1: # Check if cluster is valid and non-empty
                indices_to_fetch.append(torch.arange(offset, offset + count, device=target_device))
                total_candidates += count

        if not indices_to_fetch:
            # No candidates found in probed clusters
            return torch.full((k,), -1, dtype=torch.int64, device=target_device), \
                   torch.full((k,), -float('inf'), dtype=torch.float32, device=target_device)

        # Concatenate indices from different clusters
        candidate_sorted_idx_tensor = torch.cat(indices_to_fetch)

        # 4. Retrieve the actual normalized vectors for the candidates (already sorted and normalized)
        candidate_vectors_norm = self.sorted_data_norm[candidate_sorted_idx_tensor] # Shape (total_candidates, dim)

        # 5. Calculate exact dot products between the query and candidate vectors
        # Use distance_dot_triton: query_norm @ candidate_vectors_norm.T
        # Result shape: (1, total_candidates)
        dot_q_to_candidates = distance_cosine(query_norm, candidate_vectors_norm)

        # 6. Find the top k most similar neighbors (largest dot product) among the candidates
        actual_k = min(k, total_candidates) # Cannot return more than candidates found

        if actual_k == 0:
             return torch.full((k,), -1, dtype=torch.int64, device=target_device), \
                    torch.full((k,), -float('inf'), dtype=torch.float32, device=target_device)

        # Find k largest dot products
        topk_similarities, topk_relative_indices = torch.topk(
            dot_q_to_candidates, k=actual_k, dim=1, largest=True, sorted=True # Sort by similarity
        )
        topk_similarities = topk_similarities.squeeze(0)          # Shape (actual_k,)
        topk_relative_indices = topk_relative_indices.squeeze(0)  # Shape (actual_k,)


        # 7. Map relative indices (within candidates) back to **original** data indices
        # First, get the indices within the *sorted* data array
        topk_sorted_indices = candidate_sorted_idx_tensor[topk_relative_indices]
        # Then, map these back to the original indices using the stored mapping
        topk_original_indices = self.sorted_to_original_idx[topk_sorted_indices]

        # Prepare final output tensors of size k, padding if necessary
        final_indices = torch.full((k,), -1, dtype=torch.int64, device=target_device)
        final_similarities = torch.full((k,), -float('inf'), dtype=torch.float32, device=target_device) # Pad with worst similarity

        final_indices[:actual_k] = topk_original_indices
        final_similarities[:actual_k] = topk_similarities

        # Return original indices and dot product similarities
        return final_indices, final_similarities


# Wrapper function needs slight modification for the new class
def our_ann_kmeans_optimized(N_A, D, A, X, K, k_clusters=100, nprobe=5, max_kmeans_iters=50):
    """
    Performs Optimized Approximate Nearest Neighbor search using K-Means (IVF)
    with Dot Product similarity.

    Args:
        N_A (int): Number of database points.
        D (int): Dimensionality.
        A (torch.Tensor): Database vectors (N_A, D) on GPU.
        X (torch.Tensor): Query vectors (Q, D) on GPU.
        K (int): Number of neighbors to find for each query.
        k_clusters (int): Number of clusters for the K-Means index.
        nprobe (int): Number of clusters to probe during search.
        max_kmeans_iters (int): Max iterations for K-Means during index build.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - all_indices (torch.Tensor): Original indices of the K most similar neighbors (Q, K).
            - all_similarities (torch.Tensor): Dot product similarity scores (Q, K). Higher is better.
    """
    target_device = X.device
    A_prep, X_prep = _prepare_tensors(A, X, target_device=target_device)
    Q = X_prep.shape[0]
    assert A_prep.shape[0] == N_A, "N_A doesn't match A.shape[0]"
    assert A_prep.shape[1] == D, "D doesn't match A.shape[1]"
    assert X_prep.shape[1] == D, "D doesn't match X.shape[1]"
    assert K > 0, "K must be positive"
    assert k_clusters > 0, "k_clusters must be positive"
    assert nprobe > 0, "nprobe must be positive"
    # assert nprobe <= k_clusters - handled internally now

    print(f"Running Optimized ANN (KMeans-IVF / Dot Product): Q={Q}, N={N_A}, D={D}, K={K}")
    print(f"IVF Params: k_clusters={k_clusters}, nprobe={nprobe}")

    # 1. Build the KMeansANN_Optimized Index
    index = KMeansANN_Optimized(dim=D, k_clusters=k_clusters, nprobe=nprobe)
    index.build_index(A_prep, max_kmeans_iters=max_kmeans_iters)

    if not index.is_built:
        print("Error: Index build failed.")
        return torch.full((Q, K), -1, dtype=torch.int64), torch.full((Q, K), -float('inf'), dtype=torch.float32)

    # 2. Perform Search for each query
    start_search = time.time()
    all_indices = torch.full((Q, K), -1, dtype=torch.int64, device=target_device)
    all_similarities = torch.full((Q, K), -float('inf'), dtype=torch.float32, device=target_device) # Store Dot Sim

    print("Searching queries...")
    for q_idx in range(Q):
        # Returns original indices and dot product similarities
        q_indices, q_sims = index.search_knn(X_prep[q_idx], K)
        all_indices[q_idx] = q_indices
        all_similarities[q_idx] = q_sims # Store similarities

        if (q_idx + 1) % (max(1, Q // 10)) == 0: # Adjusted print frequency
             print(f"  Searched {q_idx+1}/{Q} queries...")

    end_search = time.time()
    print(f"Optimized ANN search time: {end_search - start_search:.4f} seconds")

    # Returns indices and Dot Product similarities
    return all_indices, all_similarities


class SimpleHNSW_DotProduct_Optimized:
    """
    Optimized HNSW implementation using Dot Product similarity (1 - dot_product distance)
    on normalized vectors, leveraging Triton kernels.
    """
    def __init__(self, dim, M=16, ef_construction=200, ef_search=50, mL=0.5):
        self.dim = dim
        self.M = M # Max connections per node per layer
        self.ef_construction = ef_construction # Size of dynamic candidate list during construction
        self.ef_search = ef_search # Size of dynamic candidate list during search
        self.mL = mL # Normalization factor for level generation
        # Store only normalized vectors
        self.vectors_norm = torch.empty((0, dim), dtype=torch.float32, device=device)
        # Graph structure: List of layers, each layer is a list of nodes,
        # each node is a list of neighbor indices for that layer.
        self.graph = []
        self.node_count = 0
        self.entry_point = -1 # Index of the entry point node
        self.max_level = -1 # Highest level currently in the graph
        # Note: level_assignments might not be strictly needed anymore if not used elsewhere
        # self.level_assignments = [] # Store the max level for each node

        print(f"Initialized HNSW (Dot Product): M={M}, efC={ef_construction}, efS={ef_search}")

    def _get_level_for_new_node(self):
        # Generates a random level based on the mL factor (higher mL -> lower levels)
        return int(-math.log(random.uniform(0, 1)) * self.mL)

    # --- Distance Calculation using Dot Product ---
    def _get_distances(self, query_norm_vector, candidate_indices):
        """
        Calculates distances (1 - dot_product) between a single normalized query
        and multiple candidate nodes using optimized Triton dot product.

        Args:
            query_norm_vector (torch.Tensor): The L2-normalized query vector (1, D).
            candidate_indices (list or torch.Tensor): Indices of candidate nodes.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - distances (torch.Tensor): Tensor of distances (1 - dot). Shape (num_valid_candidates,).
                - valid_indices (torch.Tensor): Tensor of valid candidate indices used.
        """
        if isinstance(candidate_indices, list):
            candidate_indices = torch.tensor(candidate_indices, device=query_norm_vector.device, dtype=torch.long)

        # Filter out invalid indices (e.g., padding -1 or out of bounds)
        valid_mask = (candidate_indices >= 0) & (candidate_indices < self.node_count)
        valid_indices = candidate_indices[valid_mask]

        if valid_indices.numel() == 0:
            return torch.empty(0, device=query_norm_vector.device), valid_indices # Return empty tensor of indices

        # Retrieve normalized candidate vectors
        candidate_vectors_norm = self.vectors_norm[valid_indices] # Shape (num_valid, D)

        # Calculate dot products using the optimized Triton kernel
        # Input shapes: (1, D) and (num_valid, D)
        # Output shape: (1, num_valid)
        dot_products = distance_cosine(query_norm_vector, candidate_vectors_norm)

        # Calculate distance = 1 - dot_product (similarity)
        # Squeeze to get shape (num_valid,)
        distances = 1.0 - dot_products.squeeze(0)

        return distances, valid_indices

    def _get_distances_batch(self, query_indices, candidate_indices):
        """
        Calculates pairwise distances (1 - dot_product) between batches of nodes.

        Args:
            query_indices (list or torch.Tensor): Indices of query nodes.
            candidate_indices (list or torch.Tensor): Indices of candidate nodes.

        Returns:
            torch.Tensor: Pairwise distances (1 - dot). Shape (num_valid_query, num_valid_candidate).
                          Returns empty tensor if no valid pairs.
        """
        if isinstance(query_indices, list):
            query_indices = torch.tensor(query_indices, device=self.vectors_norm.device, dtype=torch.long)
        if isinstance(candidate_indices, list):
            candidate_indices = torch.tensor(candidate_indices, device=self.vectors_norm.device, dtype=torch.long)

        valid_query_mask = (query_indices >= 0) & (query_indices < self.node_count)
        valid_query_indices = query_indices[valid_query_mask]

        valid_candidate_mask = (candidate_indices >= 0) & (candidate_indices < self.node_count)
        valid_candidate_indices = candidate_indices[valid_candidate_mask]

        if valid_query_indices.numel() == 0 or valid_candidate_indices.numel() == 0:
            return torch.empty((valid_query_indices.numel(), valid_candidate_indices.numel()),
                               device=self.vectors_norm.device)

        query_vectors_norm = self.vectors_norm[valid_query_indices]
        candidate_vectors_norm = self.vectors_norm[valid_candidate_indices]

        # Calculate dot products: shape (num_valid_query, num_valid_candidate)
        dot_products = distance_cosine(query_vectors_norm, candidate_vectors_norm)

        # Calculate distance = 1 - dot_product
        distances = 1.0 - dot_products

        return distances

    # --- Neighbor Selection Heuristic (adapted for 1-dot distance) ---
    def _select_neighbors_heuristic(self, query_norm_vec, candidates_with_dist, M_target):
        """
        Selects M_target neighbors from candidates using a heuristic based on
        the '1 - dot_product' distance metric. Keep closest overall, prune others.

        Args:
            query_norm_vec (torch.Tensor): The L2-normalized query vector (1, D).
            candidates_with_dist (list): List of tuples (distance, node_id).
                                         distance is 1 - dot_product.
            M_target (int): The maximum number of neighbors to select.

        Returns:
            list: List of selected neighbor node indices.
        """
        selected_neighbors = []
        # Use a min-heap directly on distances (smaller 1-dot distance is better)
        working_candidates_heap = list(candidates_with_dist) # Make a copy
        heapq.heapify(working_candidates_heap) # Min-heap based on distance
        discarded_candidates = set() # Track nodes to discard

        while working_candidates_heap and len(selected_neighbors) < M_target:
            # Get the node closest to the query among remaining candidates
            dist_best, best_nid = heapq.heappop(working_candidates_heap)

            # Skip if already discarded by a previously selected neighbor
            if best_nid in discarded_candidates:
                continue

            selected_neighbors.append(best_nid)

            # --- Heuristic Pruning ---
            # Prune remaining candidates 'r' if dist(r, best_nid) < dist(r, query)
            # This means 'r' is closer to the newly selected 'best_nid' than to the query,
            # suggesting 'best_nid' is a better connection point for 'r'.

            # Create a temporary list of remaining valid candidates to check against best_nid
            remaining_candidates_info = {} # {nid: dist_to_query}
            temp_heap_storage = [] # To rebuild the heap later
            while working_candidates_heap:
                 dist_r, nid_r = heapq.heappop(working_candidates_heap)
                 if nid_r not in discarded_candidates:
                      remaining_candidates_info[nid_r] = dist_r
                      # Add back to storage to rebuild heap later if not pruned
                      temp_heap_storage.append((dist_r, nid_r))

            # If no remaining candidates, we are done pruning for this step
            if not remaining_candidates_info:
                working_candidates_heap = temp_heap_storage # Should be empty
                break # Exit outer loop if no candidates left

            remaining_nids = list(remaining_candidates_info.keys())

            # Calculate distances between the selected 'best_nid' and all remaining candidates 'r'
            # Shape: (1, num_remaining)
            dists_best_to_remaining = self._get_distances_batch([best_nid], remaining_nids)

            # Add candidates to discard set based on the heuristic
            if dists_best_to_remaining.numel() > 0:
                dists_best_to_remaining = dists_best_to_remaining.squeeze(0) # Shape (num_remaining,)
                for i, r_nid in enumerate(remaining_nids):
                    dist_r_query = remaining_candidates_info[r_nid] # Dist(r, query)
                    dist_r_best = dists_best_to_remaining[i].item() # Dist(r, best_nid)

                    # Apply heuristic check: if r is closer to best_nid than to query
                    if dist_r_best < dist_r_query:
                        discarded_candidates.add(r_nid) # Mark 'r' for discarding

            # Rebuild the heap only with candidates that *weren't* discarded
            working_candidates_heap = [(dist_r, nid_r) for dist_r, nid_r in temp_heap_storage if nid_r not in discarded_candidates]
            heapq.heapify(working_candidates_heap) # Convert back to heap

        return selected_neighbors

    # --- Core HNSW Methods (adapted for normalization and 1-dot distance) ---
    def add_point(self, point_vec):
        """Adds a single point to the HNSW graph."""
        if point_vec.ndim == 1: point_vec = point_vec.unsqueeze(0) # Ensure (1, D)
        point_vec_prep, = _prepare_tensors(point_vec, target_device=device)

        # --- Normalize the incoming vector ---
        point_norm = normalize_vectors(point_vec_prep) # Shape (1, D)

        new_node_id = self.node_count

        # Append normalized vector to storage
        if self.node_count == 0:
             self.vectors_norm = point_norm
        else:
             self.vectors_norm = torch.cat((self.vectors_norm, point_norm), dim=0)
        self.node_count += 1

        # Determine level for the new node
        node_level = self._get_level_for_new_node()
        # self.level_assignments.append(node_level) # Store level if needed

        # --- Expand graph structure (CPU lists) ---
        # This part remains a potential bottleneck for huge datasets / high insert rates
        # Needs to accommodate the new node index at all levels up to node_level
        while node_level >= len(self.graph): self.graph.append([]) # Add new empty levels if needed
        for lvl_idx in range(len(self.graph)):
             # Ensure each existing level list is long enough
             while len(self.graph[lvl_idx]) <= new_node_id:
                 self.graph[lvl_idx].append([]) # Add empty neighbor lists for new/intermediate nodes

        # --- Find entry point and search down levels ---
        current_entry_point_id = self.entry_point
        current_max_level = self.max_level

        # If graph is empty, set new node as entry point
        if current_entry_point_id == -1:
            self.entry_point = new_node_id
            self.max_level = node_level
            # No connections to make yet
            return new_node_id

        # Search from top level down to node_level + 1 to find the best entry point for insertion levels
        ep_id = current_entry_point_id
        for level in range(current_max_level, node_level, -1):
             # Ensure level and entry point are valid within graph structure
             if level >= len(self.graph) or ep_id >= len(self.graph[level]): continue

             # Search returns [(distance, node_id)] sorted by distance (1-dot)
             search_results = self._search_layer(point_norm, [ep_id], level, ef=1)
             if not search_results: break # Should not happen if ep_id is valid? Error check maybe needed.
             # Update entry point for the next lower level to the closest node found
             ep_id = search_results[0][1] # Closest node ID

        # --- Insert node at levels from min(node_level, current_max_level) down to 0 ---
        ep_ids = [ep_id] # Start insertion with the entry point found above
        for level in range(min(node_level, current_max_level), -1, -1):
             if level >= len(self.graph): continue # Should not happen based on graph expansion logic

             # Find nearest neighbors at this level using the current entry points
             # Search returns [(distance, node_id)] sorted by distance (1-dot)
             neighbors_found_with_dist = self._search_layer(point_norm, ep_ids, level, self.ef_construction)
             if not neighbors_found_with_dist:
                 # Fallback if search fails unexpectedly
                 if current_entry_point_id < len(self.graph[level]):
                     neighbors_found_with_dist = self._search_layer(point_norm, [current_entry_point_id], level, self.ef_construction)
                 if not neighbors_found_with_dist: continue # Skip level if still no neighbors


             # Select neighbors using heuristic (returns list of node IDs)
             selected_neighbor_ids = self._select_neighbors_heuristic(point_norm, neighbors_found_with_dist, self.M)

             # --- Add connections for the new node ---
             # Connect new_node_id -> selected_neighbor_ids
             self.graph[level][new_node_id] = selected_neighbor_ids

             # --- Add connections from selected neighbors back to the new node ---
             for neighbor_id in selected_neighbor_ids:
                 # Ensure neighbor_id is valid before accessing its connections
                 if neighbor_id >= len(self.graph[level]): continue

                 neighbor_connections = self.graph[level][neighbor_id]

                 # Only add connection if it doesn't already exist (shouldn't happen here)
                 # and if the neighbor has less than M connections
                 if new_node_id not in neighbor_connections:
                     if len(neighbor_connections) < self.M:
                         neighbor_connections.append(new_node_id)
                     else:
                         # --- Pruning: If neighbor is full, check if new node is closer than its furthest neighbor ---
                         # Calculate distance from neighbor_id to new_node_id
                         dist_new, _ = self._get_distances(self.vectors_norm[neighbor_id].unsqueeze(0), [new_node_id])
                         if dist_new.numel() == 0: continue # Should not happen if new_node_id is valid
                         dist_new_val = dist_new.item()

                         # Calculate distances from neighbor_id to its current neighbors
                         current_neighbor_ids = list(neighbor_connections) # Get current neighbors
                         dists_to_current, valid_curr_ids = self._get_distances(self.vectors_norm[neighbor_id].unsqueeze(0), current_neighbor_ids)

                         if dists_to_current.numel() > 0:
                              # Find the furthest neighbor among the valid ones
                              furthest_dist = -1.0
                              furthest_idx_in_list = -1
                              dist_map = {nid.item(): d.item() for nid, d in zip(valid_curr_ids, dists_to_current)}

                              for list_idx, current_nid in enumerate(current_neighbor_ids):
                                   # Use .get with a large default distance if ID wasn't valid for distance calc
                                   d = dist_map.get(current_nid, float('inf'))
                                   if d > furthest_dist:
                                        furthest_dist = d
                                        furthest_idx_in_list = list_idx # Index within neighbor_connections list

                              # If new node is closer than the furthest current neighbor
                              if furthest_idx_in_list != -1 and dist_new_val < furthest_dist:
                                   # Replace the furthest neighbor with the new node
                                   neighbor_connections[furthest_idx_in_list] = new_node_id
                         # Else: If dists_to_current is empty, something is wrong, but we can't prune.

             # Update entry points for the next level down (use selected neighbors)
             ep_ids = selected_neighbor_ids
             # Fallback if no neighbors selected (use closest found)
             if not ep_ids and neighbors_found_with_dist:
                 ep_ids = [nid for _, nid in neighbors_found_with_dist[:1]]


        # Update graph entry point if new node's level is highest
        if node_level > self.max_level:
            self.max_level = node_level
            self.entry_point = new_node_id

        return new_node_id

    def _search_layer(self, query_norm_vec, entry_point_ids, target_level, ef):
        """
        Performs greedy search on a single layer using the '1 - dot_product' distance.

        Args:
            query_norm_vec (torch.Tensor): The L2-normalized query vector (1, D).
            entry_point_ids (list): List of node indices to start the search from.
            target_level (int): The graph level to search within.
            ef (int): The size of the dynamic candidate list (beam width).

        Returns:
            list: Sorted list of tuples `(distance, node_id)` for the nearest neighbors found.
                  Distance is `1 - dot_product`. Smaller is better.
        """
        if self.entry_point == -1: return [] # Empty graph

        # Ensure entry points are valid
        valid_entry_points = [ep for ep in entry_point_ids if ep < self.node_count and ep >=0]
        if not valid_entry_points:
             # Fallback to global entry point if provided EPs are invalid
             if self.entry_point != -1 and self.entry_point < self.node_count:
                 valid_entry_points = [self.entry_point]
             else:
                 return [] # Cannot start search

        # Calculate initial distances from query to entry points
        initial_distances, valid_indices_init = self._get_distances(query_norm_vec, valid_entry_points)
        if valid_indices_init.numel() == 0: return [] # Failed to get distances even to entry points


        # --- Initialize Heaps ---
        # Candidate heap (min-heap): stores (distance, node_id), ordered by distance. Nodes to visit.
        # Results heap (max-heap): stores (-distance, node_id), ordered by -distance. Best nodes found so far.
        candidate_heap = []
        results_heap = []
        visited = set() # Keep track of visited node IDs

        # Populate initial heaps
        dist_map_init = {nid.item(): d.item() for nid, d in zip(valid_indices_init, initial_distances)}
        for ep_id in valid_entry_points: # Iterate original list to handle potential invalid IDs
             dist = dist_map_init.get(ep_id, float('inf')) # Use inf if distance calculation failed
             if dist != float('inf'):
                  # Add to both heaps initially
                  heapq.heappush(candidate_heap, (dist, ep_id))
                  heapq.heappush(results_heap, (-dist, ep_id)) # Max heap stores negative distance
                  visited.add(ep_id)

        # --- Greedy Search Loop ---
        while candidate_heap:
            # Get the closest candidate node (lowest 1-dot distance)
            dist_candidate, current_node_id = heapq.heappop(candidate_heap)

            # Get the furthest node currently in the results (highest 1-dot distance)
            # Note: results_heap[0][0] is the *negative* of the largest distance
            furthest_dist_in_results = -results_heap[0][0] if results_heap else float('inf')

            # Stop condition: If the closest candidate is further than the furthest result found,
            # and we already have enough results (or more), we can stop early.
            # (Original HNSW condition, adapted for 1-dot distance)
            if dist_candidate > furthest_dist_in_results and len(results_heap) >= ef :
                 break

            # Get neighbors of the current node at the target level
            try:
                # Access graph structure (CPU list)
                neighbors = self.graph[target_level][current_node_id]
            except IndexError:
                neighbors = [] # Node might not exist at this level or has no connections

            # Process unvisited neighbors
            unvisited_neighbor_ids = [n for n in neighbors if n not in visited]
            if unvisited_neighbor_ids:
                 visited.update(unvisited_neighbor_ids) # Mark as visited

                 # Calculate distances from query to these unvisited neighbors
                 neighbor_distances, valid_neighbor_indices = self._get_distances(query_norm_vec, unvisited_neighbor_ids)

                 # Process valid neighbors
                 if valid_neighbor_indices.numel() > 0:
                      dist_map_neighbors = {nid.item(): d.item() for nid, d in zip(valid_neighbor_indices, neighbor_distances)}

                      for neighbor_id_tensor in valid_neighbor_indices: # Iterate through tensor
                           neighbor_id = neighbor_id_tensor.item()
                           neighbor_dist = dist_map_neighbors[neighbor_id]

                           # Get the current furthest distance in results again (might have changed)
                           furthest_dist_in_results = -results_heap[0][0] if results_heap else float('inf')

                           # Check if this neighbor is potentially better than the furthest result
                           # or if we don't have enough results yet
                           if len(results_heap) < ef or neighbor_dist < furthest_dist_in_results:
                                # Add to results heap (using negative distance for max-heap behavior)
                                heapq.heappush(results_heap, (-neighbor_dist, neighbor_id))
                                # If results heap exceeds ef, remove the furthest one (highest distance)
                                if len(results_heap) > ef:
                                     heapq.heappop(results_heap)

                                # Add to candidate heap for exploration
                                heapq.heappush(candidate_heap, (neighbor_dist, neighbor_id))

        # Convert results heap (negative distances) to sorted list of (positive distance, node_id)
        # Sorted by distance ascending (most similar first)
        final_results = sorted([(abs(neg_dist), node_id) for neg_dist, node_id in results_heap])
        return final_results # Returns [(distance, node_id)], distance = 1 - dot

    def search_knn(self, query_vec, k):
        """
        Searches for the k nearest neighbors (highest dot product similarity) for a single query vector.

        Args:
            query_vec (torch.Tensor): The query vector (D,) or (1, D).
            k (int): The number of neighbors to find.

        Returns:
            list: List of tuples `(distance, node_id)` for the top k neighbors.
                  Distance is `1 - dot_product`. Smaller is better.
                  Returns empty list if graph is empty or search fails.
        """
        if self.entry_point == -1: return [] # Graph is empty
        target_device = self.vectors_norm.device

        # Prepare and normalize query vector
        query_vec_prep, = _prepare_tensors(query_vec.flatten(), target_device=target_device)
        query_norm = normalize_vectors(query_vec_prep.unsqueeze(0)) # Shape (1, D)


        # --- Search Hierarchy ---
        ep_id = self.entry_point
        current_max_level = self.max_level
        # Search from top level down to level 1
        for level in range(current_max_level, 0, -1):
            # Ensure level and entry point are valid
            if level >= len(self.graph) or ep_id < 0 or ep_id >= len(self.graph[level]):
                # Attempt to recover if ep_id became invalid, use global entry point
                if self.entry_point >= 0 and self.entry_point < len(self.graph[level]):
                    ep_id = self.entry_point
                else:
                    break # Cannot proceed if entry point is invalid for this level

            # Find the single closest node at this level to be the entry point for the next
            search_results = self._search_layer(query_norm, [ep_id], level, ef=1)
            if not search_results: break # Stop if search fails at this level
            ep_id = search_results[0][1] # Update entry point

        # --- Final search at level 0 ---
        # Ensure level 0 exists and entry point is valid
        if 0 >= len(self.graph) or ep_id < 0 or ep_id >= len(self.graph[0]):
            # Attempt fallback to global entry point if ep_id became invalid
             if self.entry_point != -1 and 0 < len(self.graph) and self.entry_point < len(self.graph[0]):
                  ep_id = self.entry_point
             else:
                  return [] # Cannot perform search at level 0


        # Perform the detailed search at the base layer (level 0)
        neighbors_found = self._search_layer(query_norm, [ep_id], 0, self.ef_search)

        # Return the top k results (already sorted by distance)
        return neighbors_found[:k] # Returns [(distance, node_id)], distance = 1 - dot

# --- Updated our_ann Wrapper ---
def our_ann_optimized_hnsw(N_A, D, A, X, K, M=16, ef_construction=100, ef_search=50):
     """Wrapper for the optimized HNSW using Dot Product."""
     target_device = X.device
     # Prepare data (initial tensors)
     A_prep, X_prep = _prepare_tensors(A, X, target_device=target_device)
     Q = X_prep.shape[0]

     # Basic input validation
     if not (A_prep.shape[0]==N_A and A_prep.shape[1]==D and X_prep.shape[1]==D and K>0):
         raise ValueError("Input shape mismatch or invalid K.")

     print(f"Running Optimized ANN (HNSW / Dot Product): Q={Q}, N={N_A}, D={D}, K={K}, M={M}, efC={ef_construction}, efS={ef_search}")

     # --- Build Index ---
     start_build = time.time()
     hnsw_index = SimpleHNSW_DotProduct_Optimized(dim=D, M=M, ef_construction=ef_construction, ef_search=ef_search)
     print("Building index...");
     # Add points one by one (normalization happens inside add_point)
     # Consider adding progress reporting back if N_A is large
     for i in range(N_A):
         hnsw_index.add_point(A_prep[i])
         # if (i + 1) % max(1, N_A // 10) == 0: print(f"  Added {i+1}/{N_A}...")
     end_build = time.time()
     build_time = end_build - start_build
     print(f"Index build time: {build_time:.2f} seconds")

     # Check if index build was successful
     if hnsw_index.node_count == 0 or hnsw_index.entry_point == -1 :
         print("Error: Index build resulted in an empty or invalid graph.")
         # Return empty/sentinel tensors
         return torch.full((Q, K), -1, dtype=torch.int64, device=device), \
                torch.full((Q, K), float('inf'), dtype=torch.float32, device=device), \
                build_time, 0.0

     # --- Perform Search ---
     start_search = time.time()
     all_indices = torch.full((Q, K), -1, dtype=torch.int64, device=device)
     # Store distances (1 - dot_product). Initialize with worst possible distance (>= 2.0 or inf)
     all_distances = torch.full((Q, K), float('inf'), dtype=torch.float32, device=device)

     print("Searching queries...");
     for q_idx in range(Q):
          # search_knn returns list of (distance, node_id), distance = 1 - dot
          results = hnsw_index.search_knn(X_prep[q_idx], K)
          num_results = len(results)
          k_actual = min(num_results, K) # How many results we actually got (<= K)

          if k_actual > 0:
               # Extract distances and indices from results
               q_dists = torch.tensor([res[0] for res in results[:k_actual]], dtype=torch.float32, device=device)
               q_indices = torch.tensor([res[1] for res in results[:k_actual]], dtype=torch.int64, device=device)

               # Assign to output tensors
               all_distances[q_idx, :k_actual] = q_dists
               all_indices[q_idx, :k_actual] = q_indices

          # Consider adding progress reporting back if Q is large
          # if (q_idx + 1) % max(1, Q // 10) == 0: print(f"  Searched {q_idx+1}/{Q}...")

     end_search = time.time()
     search_time = end_search - start_search
     print(f"Optimized HNSW ANN search time: {search_time:.4f} seconds")

     # Return original indices and distances (1 - dot_product)
     return all_indices, all_distances, build_time, search_time

# ============================================================================
# Example Usage (Illustrative) - MODIFIED
# ============================================================================
if __name__ == "__main__":
    N_data = 2000000    # Larger dataset to see benefits
    N_queries = 500
    Dim = 128
    K_val = 10

    print("="*50)
    print(" Initializing Optimized IVF Test Environment")
    print("="*50)
    print(f"Dataset: N={N_data}, Queries={N_queries}, Dim={Dim}, K={K_val}")

    # --- Ensure GPU is available ---
    if not torch.cuda.is_available():
        print("CUDA not available, exiting.")
        exit()
    device = torch.device("cuda:0")
    print(f"Using device: {device}")

    # --- Generate Data ---
    print("\n" + "="*40)
    print("Generating Data...")
    print("="*40)
    A_data = torch.randn(N_data, Dim, dtype=torch.float32, device=device)
    X_queries = torch.randn(N_queries, Dim, dtype=torch.float32, device=device)
    print("Data generated.")

    # --- Brute-Force KNN using Dot Product for Ground Truth ---
    # We need a brute-force KNN using the same metric (dot product) for fair recall comparison
    def knn_bruteforce_dot(A, X, K):
        print("Running Brute-Force k-NN (Dot Product)...")
        target_device = X.device
        A_prep, X_prep = _prepare_tensors(A, X, target_device=target_device)
        N, D = A_prep.shape
        Q, _ = X_prep.shape

        # Normalize
        A_norm = normalize_vectors(A_prep)
        X_norm = normalize_vectors(X_prep)

        # Compute all pairwise dot products
        # shape: (Q, N)
        all_dot_products = distance_cosine(X_norm, A_norm)

        # Find top K largest dot products for each query
        k_actual = min(K, N)
        topk_sims, topk_indices = torch.topk(all_dot_products, k=k_actual, dim=1, largest=True, sorted=True)

        # Pad if K > k_actual (e.g., K > N)
        if k_actual < K:
             pad_indices = torch.full((Q, K - k_actual), -1, dtype=torch.int64, device=target_device)
             pad_sims = torch.full((Q, K - k_actual), -float('inf'), dtype=torch.float32, device=target_device)
             final_indices = torch.cat((topk_indices, pad_indices), dim=1)
             final_sims = torch.cat((topk_sims, pad_sims), dim=1)
        else:
             final_indices = topk_indices
             final_sims = topk_sims

        return final_indices, final_sims # Return indices and similarities

    print("\n" + "="*40)
    print(f"Testing Brute-Force k-NN (Dot Product, K={K_val})...")
    print("="*40)
    start_knn = time.time()
    knn_indices_dot, knn_sims_dot = knn_bruteforce_dot(A_data, X_queries, K_val)
    end_knn = time.time()
    print(f"Brute-Force k-NN (Dot) Time: {end_knn - start_knn:.4f} seconds")
    print("KNN (Dot) results shape (Indices):", knn_indices_dot.shape)
    print("KNN (Dot) results shape (Similarities):", knn_sims_dot.shape)
    print("Sample KNN (Dot) Indices (Query 0):\n", knn_indices_dot[0])
    print("Sample KNN (Dot) Sims (Query 0):\n", knn_sims_dot[0])


    # --- Test Optimized ANN (K-Means IVF with Dot Product) ---
    print("\n" + "="*40)
    print(f"Testing Optimized K-Means IVF ANN (Dot Product, K={K_val})...")
    print("="*40)
    # Parameters (tune these)
    num_clusters_ann = int(math.sqrt(N_data)) # Heuristic starting point
    num_probes_ann = 100 # Increase probes for potentially better recall
    start_ann_kmeans_opt = time.time()
    # Ensure KMeansANN_Optimized class and our_ann_kmeans_optimized function are defined
    ann_indices_kmeans_opt, ann_sims_kmeans_opt = our_ann_kmeans_optimized(
        N_data, Dim, A_data, X_queries, K_val,
        k_clusters=num_clusters_ann, nprobe=num_probes_ann, max_kmeans_iters=50 # Kmeans iters limit
        )
    end_ann_kmeans_opt = time.time()
    kmeans_opt_total_time = end_ann_kmeans_opt - start_ann_kmeans_opt
    print(f"Optimized K-Means IVF ANN Total Time: {kmeans_opt_total_time:.4f} seconds")
    print("Optimized ANN results shape (Indices):", ann_indices_kmeans_opt.shape)
    print("Optimized ANN results shape (Similarities - Dot):", ann_sims_kmeans_opt.shape)
    print("Sample Optimized ANN Indices (Query 0):\n", ann_indices_kmeans_opt[0])
    print("Sample Optimized ANN Sims (Query 0):\n", ann_sims_kmeans_opt[0])

    # --- Recall Calculation (Optimized IVF vs Brute Force Dot) ---
    print("\n" + "="*50)
    print(f"Recall Calculation @ {K_val} (Optimized IVF vs Brute Force Dot)")
    print("="*50)

    if N_queries > 0 and K_val > 0:
        correct_count_kmeans_opt = 0
        total_possible = 0

        # Use the ground truth from the dot-product based KNN
        true_knn_indices_for_recall = knn_indices_dot

        for i in range(N_queries):
            # Ground truth neighbors for query i (using dot product KNN results)
            true_knn_ids_set = set(true_knn_indices_for_recall[i].cpu().tolist())
            true_knn_ids_set.discard(-1) # Remove padding

            if not true_knn_ids_set: continue

            # Optimized K-Means IVF results for query i
            approx_kmeans_opt_ids_set = set(ann_indices_kmeans_opt[i].cpu().tolist())
            approx_kmeans_opt_ids_set.discard(-1) # Remove padding

            # Count correct matches
            correct_count_kmeans_opt += len(true_knn_ids_set.intersection(approx_kmeans_opt_ids_set))
            total_possible += len(true_knn_ids_set)

        # Calculate overall recall
        recall_kmeans_opt = correct_count_kmeans_opt / total_possible if total_possible > 0 else 0.0

        print(f"Optimized K-Means IVF ANN Recall @ {K_val}: {recall_kmeans_opt:.2%}")
        print(f" -> Total Time: {kmeans_opt_total_time:.4f}s")
        print(f" -> Brute-Force Dot KNN Time: {end_knn - start_knn:.4f}s (for reference)")

    else:
        print("Cannot calculate recall (N_queries=0 or K_val=0).")

    print("\n" + "="*50)
    print(" Optimized IVF Test Complete")
    print("="*50)