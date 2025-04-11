# task-1/task_Ann.py (Modified for Pure CuPy K-Means)

# Keep PyTorch and Triton imports if needed by HNSW or other parts of the file
import torch
import triton
import triton.language as tl
import math
import heapq # For HNSW priority queues
import random
import time
import cupy as cp
import cupyx # Required for cupyx.scatter_add

# --- Device Setup ---
# Keep PyTorch device setup if HNSW/other parts use it
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"PyTorch using device: {device}")
else:
    # Fallback for PyTorch if needed, but CuPy requires CUDA
    device = torch.device("cpu")
    print("PyTorch falling back to CPU.")

# Check and set CuPy device
try:
    cp.cuda.Device(0).use()
    print(f"CuPy using GPU: {cp.cuda.Device(0)}")
except cp.cuda.runtime.CUDARuntimeError as e:
    print(f"CuPy CUDA Error: {e}")
    print("Cannot run CuPy K-Means without CUDA.")
    # Decide how to handle this - exit or let subsequent CuPy calls fail?
    # exit()


# --- Helper Functions (Keep if needed by HNSW/Other PyTorch parts) ---
def _prepare_tensors(*tensors, target_device=device):
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

def normalize_vectors(vectors, epsilon=1e-12):
    """L2 normalize vectors row-wise using PyTorch."""
    # Keep this if HNSW needs it
    norms = torch.linalg.norm(vectors, dim=1, keepdim=True)
    return vectors / (norms + epsilon)

# --- Distance Functions & Kernels (Keep if needed by HNSW/Other Parts) ---
# (Keep Triton kernels like dot_kernel_pairwise, l2_dist_kernel_1_vs_M, etc.
# and their PyTorch wrappers like distance_dot, distance_cosine if HNSW uses them)
@triton.jit
def l2_dist_kernel_1_vs_M( # Keep if HNSW uses it
    query_ptr, candidates_ptr, output_ptr,
    M, D, stride_cand_m, stride_cand_d,
    BLOCK_SIZE_D: tl.constexpr,
):
    # --- Kernel code remains the same ---
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
    pass

# ============================================================================
# Task 2.1: K-Means Clustering (Pure CuPy Implementation)
# ============================================================================

def pairwise_l2_squared_cupy(X_cp, C_cp):
    """
    Computes pairwise squared L2 distances between points in X and centroids C using CuPy.
    X_cp: (N, D) data points
    C_cp: (K, D) centroids
    Returns: (N, K) tensor of squared distances.
    """
    # ||x - c||^2 = ||x||^2 - 2<x, c> + ||c||^2
    X_norm_sq = cp.sum(X_cp**2, axis=1, keepdims=True) # Shape (N, 1)
    C_norm_sq = cp.sum(C_cp**2, axis=1, keepdims=True) # Shape (K, 1)
    dot_products = X_cp @ C_cp.T                       # Shape (N, K)

    # Broadcasting: (N, 1) - 2*(N, K) + (1, K) -> (N, K)
    dist_sq = X_norm_sq - 2 * dot_products + C_norm_sq.T
    return cp.maximum(0, dist_sq) # Clamp to avoid small negatives

def our_kmeans(N_A, D, A_cp, K, max_iters=100, tol=1e-4):
    """
    Performs K-means clustering entirely using CuPy.

    Args:
        N_A (int): Number of data points.
        D (int): Dimensionality.
        A_cp (cp.ndarray): Data points (N_A, D) on GPU (as CuPy ndarray).
        K (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance for centroid movement convergence check.

    Returns:
        tuple[cp.ndarray, cp.ndarray]:
            - centroids_cp (cp.ndarray): Final centroids (K, D).
            - assignments_cp (cp.ndarray): Final cluster assignment (N_A,).
    """
    # --- Input Validation ---
    if not isinstance(A_cp, cp.ndarray):
        raise TypeError("Input data 'A_cp' must be a CuPy ndarray.")
    if A_cp.shape[0] != N_A or A_cp.shape[1] != D:
         # Warning or error if N_A/D don't match shape
         print(f"Warning: N_A/D ({N_A}/{D}) mismatch with A_cp shape {A_cp.shape}. Using shape from A_cp.")
         N_A, D = A_cp.shape
    if not (K > 0 and K <= N_A):
         raise ValueError("K must be positive and less than or equal to N_A.")
    if A_cp.dtype != cp.float32:
        print(f"Warning: Input data dtype is {A_cp.dtype}. Converting to float32.")
        A_cp = A_cp.astype(cp.float32)


    print(f"Running K-Means (Pure CuPy): N={N_A}, D={D}, K={K}")
    start_time_total = time.time()

    # --- Initialization (using CuPy) ---
    initial_indices = cp.random.permutation(N_A)[:K]
    centroids_cp = A_cp[initial_indices].copy() # Use copy to avoid modifying A_cp

    assignments_cp = cp.empty(N_A, dtype=cp.int32) # Use int32 for assignments

    for i in range(max_iters):
        iter_start_time = time.time()

        # --- 1. Assignment Step (CuPy) ---
        all_dist_sq_cp = pairwise_l2_squared_cupy(A_cp, centroids_cp) # Shape (N_A, K)
        assignments_cp = cp.argmin(all_dist_sq_cp, axis=1).astype(cp.int32) # Shape (N_A,)
        cp.cuda.Stream.null.synchronize()
        assign_time = time.time() - iter_start_time

        # --- 2. Update Step (CuPy using cupyx.scatter_add) ---
        update_start_time = time.time()

        new_sums_cp = cp.zeros((K, D), dtype=cp.float32)
        cluster_counts_cp = cp.zeros(K, dtype=cp.float32)

        # Use cupyx.scatter_add for efficient update
        # Add each point in A_cp to the sum of its assigned cluster
        cupyx.scatter_add(new_sums_cp, assignments_cp, A_cp, axis=0)
        # Count points in each cluster by scattering 1s
        cupyx.scatter_add(cluster_counts_cp, assignments_cp, 1.0)

        # Calculate new centroids, handle empty clusters
        final_counts_safe_cp = cp.maximum(cluster_counts_cp, 1.0) # Avoid division by zero
        new_centroids_cp = new_sums_cp / final_counts_safe_cp[:, None] # Broadcast counts

        # Handle empty clusters (where count is 0) - keep old centroid
        empty_cluster_mask = (cluster_counts_cp == 0)
        new_centroids_cp[empty_cluster_mask] = centroids_cp[empty_cluster_mask]

        cp.cuda.Stream.null.synchronize()
        update_time = time.time() - update_start_time

        # --- Check Convergence ---
        centroid_diff_cp = cp.linalg.norm(new_centroids_cp - centroids_cp)
        centroids_cp = new_centroids_cp # Update centroids

        print(f"  Iter {i+1}/{max_iters} | Centroid Diff: {centroid_diff_cp:.4f} | Assign Time: {assign_time:.4f}s | Update Time: {update_time:.4f}s")

        if centroid_diff_cp < tol:
            print(f"Converged after {i+1} iterations.")
            break

    if i == max_iters - 1:
        print(f"Reached max iterations ({max_iters}).")

    total_time = time.time() - start_time_total
    print(f"Total K-Means time: {total_time:.4f}s")

    # Return CuPy arrays
    return centroids_cp, assignments_cp.astype(cp.int64) # Cast assignments to int64 if needed downstream

# ============================================================================
# Task 2.2: Approximate Nearest Neighbors (ANN - Placeholder/Keep Original)
# ============================================================================
# Keep the existing SimpleHNSW_for_ANN class and our_ann wrapper function
# They use PyTorch/Triton internally. If they need to interact with the
# CuPy K-Means output, the CALLER of our_kmeans would need to handle
# the CuPy -> PyTorch conversion (e.g., via DLPack).

class SimpleHNSW_for_ANN:
     # (Paste the full HNSW class definition from the previous version here)
     # Ensure it uses PyTorch/_prepare_tensors/Triton kernels as intended
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
        self.BLOCK_SIZE_D_DIST = 128 # Or link to DEFAULT_BLOCK_D if defined globally

    def _get_level_for_new_node(self):
        level = int(-math.log(random.uniform(0, 1)) * self.mL)
        return level

    def _distance(self, query_vec, candidate_indices):
        """Internal distance calc using the 1-vs-M Triton kernel (Squared L2)."""
        if not isinstance(candidate_indices, list): candidate_indices = list(candidate_indices)
        if not candidate_indices: return torch.empty(0, device=device), []
        target_device = query_vec.device if isinstance(query_vec, torch.Tensor) else self.vectors.device
        query_vec_prep, = _prepare_tensors(query_vec.flatten(), target_device=target_device)
        valid_indices = [idx for idx in candidate_indices if idx < self.node_count and idx >= 0]
        if not valid_indices: return torch.empty(0, device=device), []
        num_valid_candidates = len(valid_indices)
        candidate_vectors, = _prepare_tensors(self.vectors[valid_indices], target_device=target_device)
        distances_out = torch.empty(num_valid_candidates, dtype=torch.float32, device=device)
        grid = (num_valid_candidates,)
        # Ensure l2_dist_kernel_1_vs_M is defined globally
        l2_dist_kernel_1_vs_M[grid](
            query_vec_prep, candidate_vectors, distances_out,
            num_valid_candidates, self.dim,
            candidate_vectors.stride(0), candidate_vectors.stride(1),
            BLOCK_SIZE_D=self.BLOCK_SIZE_D_DIST
        )
        return distances_out, valid_indices # Returns squared L2

    def _distance_batch(self, query_indices, candidate_indices):
        """
        Calculates pairwise L2 distances between batches using distance_l2_triton.
        ASSUMES distance_l2_triton exists and computes L2 distances.
        """
        # This requires a pairwise L2 function. If not available, this fails.
        # Using torch.cdist as a PyTorch alternative:
        if not query_indices or not candidate_indices:
            return torch.empty((len(query_indices), len(candidate_indices)), device=device)
        target_device = self.vectors.device

        valid_query_indices = [idx for idx in query_indices if idx < self.node_count and idx >= 0]
        valid_candidate_indices = [idx for idx in candidate_indices if idx < self.node_count and idx >= 0]
        if not valid_query_indices or not valid_candidate_indices:
             return torch.empty((len(valid_query_indices), len(valid_candidate_indices)), device=target_device)

        query_vectors = self.vectors[valid_query_indices]
        candidate_vectors = self.vectors[valid_candidate_indices]
        try:
            query_vec_prep, cand_vec_prep = _prepare_tensors(query_vectors, candidate_vectors, target_device=target_device)
            pairwise_l2_distances = torch.cdist(query_vec_prep, cand_vec_prep, p=2) # L2 distance
        except Exception as e:
            print(f"Error in _distance_batch using torch.cdist: {e}")
            return torch.empty((len(valid_query_indices), len(valid_candidate_indices)), device=target_device)

        return pairwise_l2_distances # Shape (len(valid_query), len(valid_candidate))

    def _select_neighbors_heuristic(self, query_vec, candidates, M_target):
        """Selects M_target neighbors (implementation using squared L2 internally)."""
        selected_neighbors = []
        # Candidates are (squared_dist, node_id)
        working_candidates_heap = [(dist, nid) for dist, nid in candidates]
        heapq.heapify(working_candidates_heap)
        discarded_candidates = set()

        while working_candidates_heap and len(selected_neighbors) < M_target:
            dist_best_sq, best_nid = heapq.heappop(working_candidates_heap)
            if best_nid in discarded_candidates: continue
            selected_neighbors.append(best_nid)

            remaining_candidates_info = {}
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
                dists_best_to_remaining = self._distance_batch([best_nid], remaining_nids) # Actual L2
                if dists_best_to_remaining.numel() > 0:
                    dists_best_to_remaining_sq = (dists_best_to_remaining**2).squeeze(0) # Squared L2

                    for i, r_nid in enumerate(remaining_nids):
                        dist_r_query_sq = remaining_candidates_info[r_nid] # Already squared L2
                        if i < len(dists_best_to_remaining_sq): # Bounds check
                           dist_r_best_sq = dists_best_to_remaining_sq[i].item()
                           if dist_r_best_sq < dist_r_query_sq:
                               discarded_candidates.add(r_nid)
        return selected_neighbors

    def add_point(self, point_vec):
        """Adds a single point to the graph."""
        target_device = self.vectors.device if self.node_count > 0 else device
        point_vec_prep, = _prepare_tensors(point_vec.flatten(), target_device=target_device)
        new_node_id = self.node_count

        if self.node_count == 0: self.vectors = point_vec_prep.unsqueeze(0)
        else: self.vectors = torch.cat((self.vectors, point_vec_prep.unsqueeze(0)), dim=0)
        self.node_count += 1
        node_level = self._get_level_for_new_node()
        self.level_assignments.append(node_level)

        while node_level >= len(self.graph): self.graph.append([])
        for lvl in range(len(self.graph)):
             while len(self.graph[lvl]) <= new_node_id: self.graph[lvl].append([])

        current_entry_point = self.entry_point
        current_max_level = self.max_level
        if current_entry_point == -1:
            self.entry_point = new_node_id; self.max_level = node_level; return new_node_id

        ep = [current_entry_point]
        for level in range(current_max_level, node_level, -1):
             if level >= len(self.graph) or not ep or ep[0] < 0 or ep[0] >= len(self.graph[level]): continue
             search_results = self._search_layer(point_vec_prep, ep, level, ef=1) # Uses squared L2
             if not search_results: break
             ep = [search_results[0][1]]

        for level in range(min(node_level, current_max_level), -1, -1):
             if level >= len(self.graph) or not ep or any(idx < 0 or idx >= len(self.graph[level]) for idx in ep): # Check all ep elements
                 if current_entry_point >=0 and current_entry_point < len(self.graph[level]):
                      ep = [current_entry_point]
                 else: continue # Cannot proceed

             neighbors_found_with_dist_sq = self._search_layer(point_vec_prep, ep, level, self.ef_construction) # Uses squared L2
             if not neighbors_found_with_dist_sq: continue
             selected_neighbor_ids = self._select_neighbors_heuristic(point_vec_prep, neighbors_found_with_dist_sq, self.M)
             self.graph[level][new_node_id] = selected_neighbor_ids

             for neighbor_id in selected_neighbor_ids:
                 if neighbor_id < 0 or neighbor_id >= len(self.graph[level]): continue # Ensure neighbor_id is valid
                 neighbor_connections = self.graph[level][neighbor_id]
                 if new_node_id not in neighbor_connections:
                     if len(neighbor_connections) < self.M:
                         neighbor_connections.append(new_node_id)
                     else:
                         # Pruning logic (uses squared L2 from _distance)
                         dist_new_sq, valid_new = self._distance(self.vectors[neighbor_id], [new_node_id])
                         if not valid_new or dist_new_sq.numel() == 0: continue
                         dist_new_sq = dist_new_sq[0].item()

                         current_neighbor_ids = list(neighbor_connections)
                         dists_to_current_sq, valid_curr_ids = self._distance(self.vectors[neighbor_id], current_neighbor_ids)

                         if dists_to_current_sq.numel() > 0:
                              furthest_dist_sq = -1.0; furthest_idx_in_list = -1
                              dist_map = {nid.item(): d.item() for nid, d in zip(valid_curr_ids, dists_to_current_sq)}
                              for list_idx, current_nid in enumerate(current_neighbor_ids):
                                   d_sq = dist_map.get(current_nid, float('inf'))
                                   if d_sq > furthest_dist_sq: furthest_dist_sq = d_sq; furthest_idx_in_list = list_idx
                              if furthest_idx_in_list != -1 and dist_new_sq < furthest_dist_sq:
                                   neighbor_connections[furthest_idx_in_list] = new_node_id

             ep = selected_neighbor_ids
             if not ep: ep = [nid for _, nid in neighbors_found_with_dist_sq[:1]]

        if node_level > self.max_level: self.max_level = node_level; self.entry_point = new_node_id
        return new_node_id

    def _search_layer(self, query_vec, entry_points, target_level, ef):
        """Performs greedy search on a single layer (returns squared L2 dists)."""
        if self.entry_point == -1: return []
        valid_entry_points = [ep for ep in entry_points if ep >= 0 and ep < self.node_count]
        if not valid_entry_points:
             if self.entry_point >= 0 and self.entry_point < self.node_count: valid_entry_points = [self.entry_point]
             else: return []

        initial_distances_sq, valid_indices_init = self._distance(query_vec, valid_entry_points) # Squared L2
        if valid_indices_init.numel() == 0: return []

        dist_map_init = {nid.item(): d.item() for nid, d in zip(valid_indices_init, initial_distances_sq)}
        candidate_heap = [(dist_map_init.get(ep, float('inf')), ep) for ep in valid_entry_points if ep in dist_map_init] # Only add valid EPs
        heapq.heapify(candidate_heap)
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
                 neighbor_distances_sq, valid_neighbor_indices = self._distance(query_vec, unvisited_neighbors) # Squared L2
                 if valid_neighbor_indices.numel() == 0: continue

                 dist_map_neighbors = {nid.item(): d.item() for nid, d in zip(valid_neighbor_indices, neighbor_distances_sq)}
                 for neighbor_id_tensor in valid_neighbor_indices:
                      neighbor_id = neighbor_id_tensor.item()
                      neighbor_dist_sq_val = dist_map_neighbors[neighbor_id]
                      furthest_dist_sq = -results_heap[0][0] if results_heap else float('inf')
                      if len(results_heap) < ef or neighbor_dist_sq_val < furthest_dist_sq:
                           heapq.heappush(results_heap, (-neighbor_dist_sq_val, neighbor_id))
                           if len(results_heap) > ef: heapq.heappop(results_heap)
                           heapq.heappush(candidate_heap, (neighbor_dist_sq_val, neighbor_id))

        final_results = sorted([(abs(neg_dist_sq), node_id) for neg_dist_sq, node_id in results_heap])
        return final_results # Returns (squared_dist, node_id)

    def search_knn(self, query_vec, k):
        """Searches for k nearest neighbors (returns squared L2 dists)."""
        if self.entry_point == -1: return []
        query_vec_prep, = _prepare_tensors(query_vec.flatten(), target_device=self.vectors.device)
        ep = [self.entry_point]
        current_max_level = self.max_level
        for level in range(current_max_level, 0, -1):
             if level >= len(self.graph) or not ep or ep[0] < 0 or ep[0] >= len(self.graph[level]):
                 if self.entry_point >= 0 and self.entry_point < len(self.graph[level]): ep = [self.entry_point]
                 else: break
                 if ep[0] < 0 or ep[0] >= len(self.graph[level]): break # Check again after fallback

             search_results = self._search_layer(query_vec_prep, ep, level, ef=1) # Squared L2
             if not search_results: break
             ep = [search_results[0][1]]
        if 0 >= len(self.graph) or not ep or ep[0] < 0 or ep[0] >= len(self.graph[0]):
             if self.entry_point != -1 and 0 < len(self.graph) and self.entry_point < len(self.graph[0]): ep = [self.entry_point]
             else: return []
        neighbors_found = self._search_layer(query_vec_prep, ep, 0, self.ef_search) # Squared L2
        return neighbors_found[:k] # Returns (squared_dist, node_id)


# --- our_ann function remains the same, calling this class ---
def our_ann(N_A, D, A, X, K, M=16, ef_construction=100, ef_search=50):
     # This function assumes A and X are PyTorch tensors
     # It calls SimpleHNSW_for_ANN which uses PyTorch/Triton internally
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

# ============================================================================
# Example Usage
# ============================================================================
if __name__ == "__main__":
    N_data = 5000
    Dim = 128
    K_clusters = 5

    print("="*40)
    print("Generating K-Means Test Data (CuPy)...")
    print("="*40)
    # Generate data directly as CuPy array
    A_data_cp = cp.random.randn(N_data, Dim, dtype=cp.float32)

    print("\n" + "="*40)
    print(f"Testing our_kmeans (Pure CuPy) (K={K_clusters})...")
    print("="*40)

    try:
        # Pass the CuPy array directly
        kmeans_centroids_cp, kmeans_assignments_cp = our_kmeans(
            N_data, Dim, A_data_cp, K_clusters, max_iters=20
        )
        print("KMeans centroids shape (CuPy):", kmeans_centroids_cp.shape)
        print("KMeans assignments shape (CuPy):", kmeans_assignments_cp.shape)

        # Verify counts using CuPy
        ids_cp, counts_cp = cp.unique(kmeans_assignments_cp.astype(cp.int32), return_counts=True) # unique needs int32 or int64 usually
        print("Cluster counts (CuPy):")
        # Convert to numpy for printing if needed
        ids_np = cp.asnumpy(ids_cp)
        counts_np = cp.asnumpy(counts_cp)
        for id_val, count_val in zip(ids_np, counts_np):
            print(f"  Cluster {id_val}: {count_val}")

    except ImportError:
         print("Error: cupyx not found. Cannot use cupyx.scatter_add.")
         print("Please install cupyx or modify the update step.")
    except cp.cuda.runtime.CUDARuntimeError as e:
         print(f"CUDA runtime error during K-Means: {e}")
    except Exception as e:
        print(f"Error during K-Means execution: {e}")
        import traceback
        traceback.print_exc()

    # --- ANN Example (Requires PyTorch Input) ---
    # If you want to test HNSW, you'll need PyTorch tensors
    print("\n" + "="*40)
    print("Generating ANN Test Data (PyTorch)...")
    print("="*40)
    N_queries = 100
    K_val = 10
    A_data_ann = torch.randn(N_data, Dim, dtype=torch.float32, device=device)
    X_queries_ann = torch.randn(N_queries, Dim, dtype=torch.float32, device=device)

    print("\n" + "="*40)
    print(f"Testing our_ann (HNSW - PyTorch/Triton) (K={K_val})...")
    print("="*40)
    try:
        # our_ann expects PyTorch tensors
        ann_indices, ann_dists_sq, build_t, search_t = our_ann(
            N_data, Dim, A_data_ann, X_queries_ann, K_val,
            M=32, ef_construction=200, ef_search=100
        )
        print("ANN results shape (Indices):", ann_indices.shape)
        print("ANN results shape (Distances - Squared L2):", ann_dists_sq.shape)
    except Exception as e:
        print(f"Error during ANN execution: {e}")
        import traceback
        traceback.print_exc()