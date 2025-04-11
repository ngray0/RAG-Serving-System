# task-1/task_Ann.py (Modified)
import torch
# Removed triton imports as they are no longer needed for k-means
import triton
import triton.language as tl
import math
import heapq # For HNSW priority queues
import random
import time
import cupy as cp # Ensure cupy is imported

# --- Device Setup ---
if not torch.cuda.is_available():
    print("CUDA not available, exiting.")
    exit()
# Check if CuPy recognizes the GPU
try:
    cp.cuda.Device(0).use()
    print(f"CuPy using GPU: {cp.cuda.Device(0)}")
except cp.cuda.runtime.CUDARuntimeError as e:
    print(f"CuPy CUDA Error: {e}")
    print("Falling back to CPU for PyTorch, but CuPy K-Means will fail.")
    device = torch.device("cpu")
# PyTorch device setup remains the same
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"PyTorch using device: {device}")


# --- Helper Functions ---
# (Keep _prepare_tensors and normalize_vectors as they might be used elsewhere)
def _prepare_tensors(*tensors, target_device = device):
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
    norms = torch.linalg.norm(vectors, dim=1, keepdim=True)
    return vectors / (norms + epsilon)

# --- Distance Functions (Keep any that might be used by other parts like ANN) ---
# (Keep distance_dot, distance_cosine if needed by ANN or other parts)
# (Keep L2 distance kernels if needed by ANN)

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
    pass # Keep kernel if needed elsewhere


# ============================================================================
# Task 2.1: K-Means Clustering (CuPy Implementation)
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
    # Clamp to avoid small negative values due to float precision
    return cp.maximum(0, dist_sq)

def our_kmeans(N_A, D, A, K, max_iters=100, tol=1e-4):
    """
    Performs K-means clustering on data A using CuPy for assignment
    and PyTorch scatter_add_ for the update step.

    Args:
        N_A (int): Number of data points.
        D (int): Dimensionality.
        A (torch.Tensor): Data points (N_A, D) on GPU (as PyTorch Tensor).
        K (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance for centroid movement convergence check.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - centroids (torch.Tensor): Final centroids (K, D).
            - assignments (torch.Tensor): Final cluster assignment (N_A,).
    """
    # Prepare input tensor (ensure it's on the right device and contiguous)
    A_prep, = _prepare_tensors(A, target_device=device)
    assert A_prep.shape[0] == N_A, "N_A doesn't match A.shape[0]"
    assert A_prep.shape[1] == D, "D doesn't match A.shape[1]"
    assert K > 0, "K must be positive"
    assert K <= N_A, "K cannot be larger than the number of data points"

    print(f"Running K-Means (Assignment: CuPy, Update: PyTorch): N={N_A}, D={D}, K={K}")
    start_time_total = time.time()

    # --- Initialization (using PyTorch on the target device) ---
    initial_indices = torch.randperm(N_A, device=device)[:K]
    centroids_torch = A_prep[initial_indices].clone()

    # --- Convert Initial Centroids and Data to CuPy ---
    # Use DLPack for efficient zero-copy transfer if possible
    try:
        A_cp = cp.from_dlpack(torch.to_dlpack(A_prep))
        centroids_cp = cp.from_dlpack(torch.to_dlpack(centroids_torch))
    except Exception as e:
        print(f"DLPack conversion failed ({e}), falling back to .numpy() conversion.")
        A_cp = cp.asarray(A_prep.cpu().numpy())
        centroids_cp = cp.asarray(centroids_torch.cpu().numpy())
        # If using numpy conversion, ensure data is moved back to GPU if needed by CuPy
        A_cp = cp.asarray(A_cp)
        centroids_cp = cp.asarray(centroids_cp)

    assignments_cp = cp.empty(N_A, dtype=cp.int32) # CuPy assignments

    for i in range(max_iters):
        iter_start_time = time.time()

        # --- 1. Assignment Step (Uses CuPy) ---
        # Calculate pairwise squared L2 distances
        all_dist_sq_cp = pairwise_l2_squared_cupy(A_cp, centroids_cp) # Shape (N_A, K)

        # Find the index of the closest centroid for each point
        assignments_cp = cp.argmin(all_dist_sq_cp, axis=1).astype(cp.int32) # Shape (N_A,)

        # Synchronize CuPy operations before potentially switching back to PyTorch
        cp.cuda.Stream.null.synchronize()
        assign_time = time.time() - iter_start_time

        # --- Convert Assignments back to PyTorch for Update Step ---
        # Use DLPack if possible
        try:
            # Ensure assignments_cp is int64 for scatter_add index
            assignments_torch = torch.from_dlpack(cp.to_dlpack(assignments_cp.astype(cp.int64)))
        except Exception as e:
             print(f"DLPack conversion failed ({e}), falling back to .numpy() conversion.")
             # Cast to int64 needed by scatter_add
             assignments_torch = torch.tensor(cp.asnumpy(assignments_cp), dtype=torch.int64, device=device)


        # --- 2. Update Step (Uses PyTorch scatter_add_) ---
        update_start_time = time.time()

        new_sums_torch = torch.zeros_like(centroids_torch) # Shape (K, D)
        cluster_counts_torch = torch.zeros(K, dtype=torch.float32, device=device) # Shape (K,)

        # Expand assignments to match the dimensions of A_prep for scatter_add_ on sums
        idx_expand = assignments_torch.unsqueeze(1).expand(N_A, D)

        # Add data points (A_prep is still the PyTorch tensor) to corresponding centroid sums
        new_sums_torch.scatter_add_(dim=0, index=idx_expand, src=A_prep)

        # Add 1 to counts for each data point's assigned cluster
        cluster_counts_torch.scatter_add_(dim=0, index=assignments_torch, src=torch.ones_like(assignments_torch, dtype=torch.float32))

        # Calculate new centroids, handle empty clusters
        final_counts_safe = cluster_counts_torch.clamp(min=1.0)
        new_centroids_torch = new_sums_torch / final_counts_safe.unsqueeze(1)

        # Keep old centroid if cluster becomes empty
        empty_cluster_mask = (cluster_counts_torch == 0)
        new_centroids_torch[empty_cluster_mask] = centroids_torch[empty_cluster_mask]

        # Synchronize PyTorch operations
        torch.cuda.synchronize()
        update_time = time.time() - update_start_time

        # --- Check Convergence ---
        # Calculate difference on PyTorch tensors
        centroid_diff = torch.norm(new_centroids_torch - centroids_torch)

        # Update centroids (both PyTorch and CuPy versions)
        centroids_torch = new_centroids_torch
        try:
            centroids_cp = cp.from_dlpack(torch.to_dlpack(centroids_torch))
        except Exception as e:
            print(f"DLPack conversion failed ({e}), falling back to .numpy() conversion.")
            centroids_cp = cp.asarray(centroids_torch.cpu().numpy())
            centroids_cp = cp.asarray(centroids_cp) # Ensure on GPU if numpy was used


        print(f"  Iter {i+1}/{max_iters} | Centroid Diff: {centroid_diff:.4f} | Assign Time (CuPy): {assign_time:.4f}s | Update Time (PyTorch): {update_time:.4f}s")

        if centroid_diff < tol:
            print(f"Converged after {i+1} iterations.")
            break

    if i == max_iters - 1:
        print(f"Reached max iterations ({max_iters}).")

    total_time = time.time() - start_time_total
    print(f"Total K-Means time: {total_time:.4f}s")

    # Return PyTorch tensors consistent with the original function signature
    # Final assignments need to be converted back to PyTorch if not already done in loop
    try:
        final_assignments_torch = torch.from_dlpack(cp.to_dlpack(assignments_cp.astype(cp.int64)))
    except Exception as e:
        print(f"DLPack conversion failed ({e}), falling back to .numpy() conversion.")
        final_assignments_torch = torch.tensor(cp.asnumpy(assignments_cp), dtype=torch.int64, device=device)

    return centroids_torch, final_assignments_torch

# ============================================================================
# Task 2.2: Approximate Nearest Neighbors (ANN - Placeholder/Keep Original)
# ============================================================================
# Keep the existing SimpleHNSW_for_ANN class and our_ann wrapper function
# if they are needed by the rest of the script (e.g., for recall comparison)
# Ensure they use the necessary kernels (like l2_dist_kernel_1_vs_M) which
# were kept above.

# (Paste the SimpleHNSW_for_ANN class and our_ann function here from task_Ann.py)
# Make sure distance functions used internally by HNSW (like _distance, _distance_batch)
# rely on the correct kernels or PyTorch/Triton implementations as needed.


# --- HNSW Class (Keep if needed, ensure internal distances are correct) ---
class SimpleHNSW_for_ANN:
    # Paste the full class definition from task_Ann.py here
    # Ensure the internal _distance and _distance_batch methods use the
    # appropriate distance calculations (likely the Triton kernels or PyTorch equivalents)
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
        # Use a reasonable block size, maybe link to a global default?
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
        # This requires a pairwise L2 function. If task_Ann.py defined one
        # (like `distance_l2_triton`), use it. Otherwise, implement one.
        # Let's assume distance_l2_triton exists from task_Ann.py:
        if not query_indices or not candidate_indices:
            return torch.empty((len(query_indices), len(candidate_indices)), device=device)
        target_device = self.vectors.device

        valid_query_indices = [idx for idx in query_indices if idx < self.node_count and idx >= 0]
        valid_candidate_indices = [idx for idx in candidate_indices if idx < self.node_count and idx >= 0]
        if not valid_query_indices or not valid_candidate_indices:
             return torch.empty((len(valid_query_indices), len(valid_candidate_indices)), device=target_device)

        query_vectors = self.vectors[valid_query_indices]
        candidate_vectors = self.vectors[valid_candidate_indices]

        # Assuming distance_l2_triton calculates pairwise L2 distance
        try:
             # Need a pairwise L2 distance function. If not available, this fails.
             # Using torch.cdist as a PyTorch alternative:
             query_vec_prep, cand_vec_prep = _prepare_tensors(query_vectors, candidate_vectors, target_device=target_device)
             pairwise_l2_distances = torch.cdist(query_vec_prep, cand_vec_prep, p=2) # L2 distance

             # Alternatively, if distance_l2_triton exists and returns L2 (not squared):
             # pairwise_l2_distances = distance_l2_triton(query_vectors, candidate_vectors)

             # If distance_l2_triton computes SQUARED L2, take sqrt:
             # pairwise_l2_dist_sq = distance_l2_squared_triton(query_vectors, candidate_vectors)
             # pairwise_l2_distances = torch.sqrt(pairwise_l2_dist_sq)

        except NameError:
            print("Error: Pairwise L2 distance function (e.g., distance_l2_triton) not found for HNSW _distance_batch.")
            # Fallback to basic PyTorch cdist (might be slower than optimized Triton)
            query_vec_prep, cand_vec_prep = _prepare_tensors(query_vectors, candidate_vectors, target_device=target_device)
            pairwise_l2_distances = torch.cdist(query_vec_prep, cand_vec_prep, p=2)

        except Exception as e:
            print(f"Error in _distance_batch: {e}")
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
             if level >= len(self.graph) or not ep or ep[0] >= len(self.graph[level]): continue
             search_results = self._search_layer(point_vec_prep, ep, level, ef=1) # Uses squared L2
             if not search_results: break
             ep = [search_results[0][1]]

        for level in range(min(node_level, current_max_level), -1, -1):
             if level >= len(self.graph) or not ep or any(idx >= len(self.graph[level]) for idx in ep):
                 if current_entry_point < len(self.graph[level]): ep = [current_entry_point]
                 else: continue

             neighbors_found_with_dist_sq = self._search_layer(point_vec_prep, ep, level, self.ef_construction) # Uses squared L2
             if not neighbors_found_with_dist_sq: continue
             selected_neighbor_ids = self._select_neighbors_heuristic(point_vec_prep, neighbors_found_with_dist_sq, self.M)
             self.graph[level][new_node_id] = selected_neighbor_ids

             for neighbor_id in selected_neighbor_ids:
                 if neighbor_id >= len(self.graph[level]): continue
                 neighbor_connections = self.graph[level][neighbor_id]
                 if new_node_id not in neighbor_connections:
                     if len(neighbor_connections) < self.M:
                         neighbor_connections.append(new_node_id)
                     else:
                         # Pruning logic (uses squared L2 from _distance)
                         dist_new_sq, valid_new = self._distance(self.vectors[neighbor_id], [new_node_id])
                         if not valid_new or dist_new_sq.numel() == 0: continue # Check if distance calculation succeeded
                         dist_new_sq = dist_new_sq[0].item()

                         current_neighbor_ids = list(neighbor_connections)
                         dists_to_current_sq, valid_curr_ids = self._distance(self.vectors[neighbor_id], current_neighbor_ids)

                         if dists_to_current_sq.numel() > 0:
                              furthest_dist_sq = -1.0; furthest_idx_in_list = -1
                              # Map valid distances back to original indices
                              dist_map = {nid.item(): d.item() for nid, d in zip(valid_curr_ids, dists_to_current_sq)}
                              for list_idx, current_nid in enumerate(current_neighbor_ids):
                                   # Use get with infinity if the neighbor was invalid for distance calc
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
        valid_entry_points = [ep for ep in entry_points if ep < self.node_count and ep >=0]
        if not valid_entry_points:
             if self.entry_point != -1 and self.entry_point < self.node_count: valid_entry_points = [self.entry_point]
             else: return []

        initial_distances_sq, valid_indices_init = self._distance(query_vec, valid_entry_points) # Squared L2
        if valid_indices_init.numel() == 0: return [] # Changed check from initial_distances_sq

        dist_map_init = {nid.item(): d.item() for nid, d in zip(valid_indices_init, initial_distances_sq)}
        candidate_heap = [(dist_map_init.get(ep, float('inf')), ep) for ep in valid_entry_points]
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
                 if valid_neighbor_indices.numel() == 0: continue # Changed check

                 dist_map_neighbors = {nid.item(): d.item() for nid, d in zip(valid_neighbor_indices, neighbor_distances_sq)}
                 for neighbor_id_tensor in valid_neighbor_indices: # Iterate through tensor
                      neighbor_id = neighbor_id_tensor.item() # Get Python int
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
             if level >= len(self.graph) or not ep or ep[0] < 0 or ep[0] >= len(self.graph[level]): # Added check ep[0] >= 0
                 # Fallback if ep becomes invalid
                 if self.entry_point >= 0 and self.entry_point < len(self.graph[level]):
                     ep = [self.entry_point]
                 else:
                     break # Cannot proceed
                 # Add a check here if ep is still invalid after fallback
                 if ep[0] < 0 or ep[0] >= len(self.graph[level]):
                     break

             search_results = self._search_layer(query_vec_prep, ep, level, ef=1) # Squared L2
             if not search_results: break
             ep = [search_results[0][1]]
        if 0 >= len(self.graph) or not ep or ep[0] < 0 or ep[0] >= len(self.graph[0]): # Added check ep[0] >= 0
             if self.entry_point != -1 and 0 < len(self.graph) and self.entry_point < len(self.graph[0]): ep = [self.entry_point]
             else: return []
        neighbors_found = self._search_layer(query_vec_prep, ep, 0, self.ef_search) # Squared L2
        return neighbors_found[:k] # Returns (squared_dist, node_id)


# --- our_ann function remains the same, calling this class ---
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

# ============================================================================
# Example Usage (Keep or comment out as needed)
# ============================================================================
if __name__ == "__main__":
    N_data = 5000
    N_queries = 100
    Dim = 128
    K_val = 10

    print("="*40)
    print("Generating K-Means Test Data...")
    print("="*40)
    A_data_kmeans = torch.randn(N_data, Dim, dtype=torch.float32, device=device)
    K_clusters = 5

    print("\n" + "="*40)
    print(f"Testing our_kmeans (CuPy Assign, PyTorch Update) (K={K_clusters})...")
    print("="*40)

    try:
        kmeans_centroids, kmeans_assignments = our_kmeans(N_data, Dim, A_data_kmeans, K_clusters, max_iters=20) # Limit iters for example
        print("KMeans centroids shape:", kmeans_centroids.shape)
        print("KMeans assignments shape:", kmeans_assignments.shape)
        ids, counts = torch.unique(kmeans_assignments, return_counts=True)
        print("Cluster counts:")
        for id_val, count_val in zip(ids.tolist(), counts.tolist()):
            print(f"  Cluster {id_val}: {count_val}")
    except Exception as e:
        print(f"Error during K-Means execution: {e}")
        import traceback
        traceback.print_exc()

    # --- You can add back the ANN tests here if needed ---
    # print("\n" + "="*40)
    # print("Generating ANN Test Data...")
    # print("="*40)
    # A_data_ann = torch.randn(N_data, Dim, dtype=torch.float32, device=device)
    # X_queries_ann = torch.randn(N_queries, Dim, dtype=torch.float32, device=device)
    #
    # print("\n" + "="*40)
    # print(f"Testing our_ann (HNSW, K={K_val})...")
    # print("="*40)
    # try:
    #     ann_indices, ann_dists, build_t, search_t = our_ann(N_data, Dim, A_data_ann, X_queries_ann, K_val,
    #                              M=32, ef_construction=200, ef_search=100)
    #     print("ANN results shape (Indices):", ann_indices.shape)
    #     print("ANN results shape (Distances - Squared L2):", ann_dists.shape)
    # except Exception as e:
    #      print(f"Error during ANN execution: {e}")
    #      import traceback
    #      traceback.print_exc()