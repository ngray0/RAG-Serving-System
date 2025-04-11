import torch
import triton
import triton.language as tl
import triton.testing # Needed for benchmarking utilities if used separately
import time
import traceback # For error printing

# --- Device Setup ---
if not torch.cuda.is_available():
    print("CUDA not available, exiting.")
    exit()
device = torch.device("cuda:0")
print(f"Using device: {device}")

# --- Default Block Size (Can be removed if autotuner tunes BLOCK_SIZE_D) ---
# DEFAULT_BLOCK_D = 128 # No longer strictly needed for the autotuned kernel

# --- Triton Kernels ---

# dot_kernel_pairwise remains the same as before
@triton.jit
def dot_kernel_pairwise(
    X_ptr, A_ptr, Out_ptr,
    Q, N, D,
    stride_xq, stride_xd,
    stride_an, stride_ad,
    stride_outq, stride_outn,
    BLOCK_SIZE_D: tl.constexpr,
):
    # ... (kernel code as before) ...
    pid_q = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    dot_prod = tl.zeros((), dtype=tl.float32)
    for d_start in range(0, D, BLOCK_SIZE_D):
        offs_d = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offs_d < D
        x_ptrs = X_ptr + pid_q * stride_xq + offs_d
        x_vals = tl.load(x_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        a_ptrs = A_ptr + pid_n * stride_an + offs_d
        a_vals = tl.load(a_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        dot_prod += tl.sum(x_vals * a_vals, axis=0)
    out_offset = pid_q * stride_outq + pid_n * stride_outn
    tl.store(Out_ptr + out_offset, dot_prod)

# Add autotuner to the assignment kernel
@triton.autotune(
    configs=[
        # Basic configs varying block sizes (powers of 2 are common)
        triton.Config({'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K_CHUNK': 32, 'BLOCK_SIZE_D': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K_CHUNK': 64, 'BLOCK_SIZE_D': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K_CHUNK': 32, 'BLOCK_SIZE_D': 256}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K_CHUNK': 64, 'BLOCK_SIZE_D': 128}, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K_CHUNK': 128, 'BLOCK_SIZE_D': 128}, num_warps=8),
        # Add more configs based on experimentation or GPU specs
    ],
    key=['N', 'D', 'K'], # Cache tuning result based on these dimensions
)
@triton.jit
def kmeans_assign_kernel(
    A_ptr,           # Pointer to data points (N, D) float32
    centroids_ptr,   # Pointer to centroids (K, D) float32
    assignments_ptr, # Pointer to output assignments (N,) int32
    N, D, K,
    stride_an, stride_ad,
    stride_ck, stride_cd,
    # These are now automatically set by the autotuner based on the chosen config
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K_CHUNK: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    """Assigns each point in A to the nearest centroid (Squared L2). Autotuned."""
    # ... (Kernel logic remains exactly the same as before) ...
    pid_n_base = tl.program_id(axis=0) * BLOCK_SIZE_N
    offs_n = pid_n_base + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N
    min_dist_sq = tl.full((BLOCK_SIZE_N,), float('inf'), dtype=tl.float32)
    best_assignment = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)
    points_block_ptr = A_ptr + offs_n[:, None] * stride_an
    for k_start in range(0, K, BLOCK_SIZE_K_CHUNK):
        k_end = tl.minimum(k_start + BLOCK_SIZE_K_CHUNK, K)
        for k_offset in range(BLOCK_SIZE_K_CHUNK):
            k_idx = k_start + k_offset
            if k_idx < k_end:
                current_dist_sq = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
                centroid_row_ptr = centroids_ptr + k_idx * stride_ck
                for d_start in range(0, D, BLOCK_SIZE_D):
                    offs_d = d_start + tl.arange(0, BLOCK_SIZE_D)
                    mask_d = offs_d < D
                    centroid_d_ptr = centroid_row_ptr + offs_d * stride_cd
                    centroid_vals = tl.load(centroid_d_ptr, mask=mask_d, other=0.0)
                    points_d_ptr = points_block_ptr + offs_d[None, :] * stride_ad
                    points_vals = tl.load(points_d_ptr, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
                    diff = points_vals - centroid_vals[None, :]
                    current_dist_sq += tl.sum(diff * diff, axis=1)
                is_closer = current_dist_sq < min_dist_sq
                min_dist_sq = tl.where(is_closer, current_dist_sq, min_dist_sq)
                best_assignment = tl.where(is_closer, k_idx, best_assignment)
    assignments_out_ptrs = assignments_ptr + offs_n
    tl.store(assignments_out_ptrs, best_assignment, mask=mask_n)


# --- Helper Functions (_prepare_tensors) ---
# Use the version that returns the single tensor if only one arg is passed
def _prepare_tensors(*tensors, target_device=device):
    prepared = []
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            try: t = torch.tensor(t, dtype=torch.float32, device=target_device)
            except Exception as e: raise TypeError(f"Failed conversion: {e}")
        if t.device != target_device: t = t.to(target_device)
        if t.dtype != torch.float32: t = t.to(dtype=torch.float32)
        if not t.is_contiguous(): t = t.contiguous()
        prepared.append(t)
    if len(prepared) == 1: return prepared[0] # Return single tensor directly
    else: return prepared # Return list/tuple for multiple tensors

# --- Distance Functions (distance_dot_tiled, distance_l2) ---
# Keep the corrected versions from previous steps (using positive dot product)
# Need DEFAULT_BLOCK_D for distance_dot_tiled if not autotuning it
DEFAULT_BLOCK_D = 128 # Define for distance_dot_tiled

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

def distance_l2(X, A): # Corrected version
    X_prep, A_prep = _prepare_tensors(X, A)
    Q, D = X_prep.shape; N, D_A = A_prep.shape
    if D != D_A: raise ValueError("Dimension mismatch")
    dot_products = distance_dot_tiled(X_prep, A_prep, prep=False) # Positive dot
    X_norm_sq = torch.sum(X_prep**2, axis=1, keepdims=True)
    A_norm_sq = torch.sum(A_prep**2, axis=1, keepdims=True)
    dist_sq = X_norm_sq + A_norm_sq.T - 2 * dot_products # Correct formula
    dist_sq.clamp_(min=0.0)
    return dist_sq

# --- Triton KMeans Implementation (Modified for Autotuner) ---
def our_kmeans(N_A, D, A, K, max_iters=100, tol=1e-4):
    """
    Performs K-means using Autotuned Triton kernel for assignment
    and PyTorch scatter_add_ for the update step. Uses L2 distance.
    """
    A_prep = _prepare_tensors(A) # Use corrected _prepare_tensors
    if A_prep.shape[0] != N_A or A_prep.shape[1] != D: N_A, D = A_prep.shape
    if not (K > 0): raise ValueError("K must be positive.")
    if K > N_A: K = N_A

    # print(f"Running K-Means (AUTOTUNED Triton Assign + PyTorch Update): N={N_A}, D={D}, K={K}")
    start_time_total = time.time()

    if K == 0: return torch.empty((0, D), dtype=torch.float32, device=device), \
                      torch.empty((N_A,), dtype=torch.int64, device=device)

    # --- Initialization ---
    initial_indices = torch.randperm(N_A, device=device)[:K]
    centroids = A_prep[initial_indices].clone()
    assignments = torch.empty(N_A, dtype=torch.int64, device=device)
    old_centroids = torch.empty_like(centroids)

    # --- Grid calculation for autotuned kernel ---
    # Grid now depends on BLOCK_SIZE_N chosen by autotuner, passed via meta
    grid_assign = lambda meta: (triton.cdiv(N_A, meta['BLOCK_SIZE_N']),)

    # --- Benchmarking variables ---
    assignment_times = []
    update_times = []

    for i in range(max_iters):
        t_iter_start = time.time()
        old_centroids = centroids.clone()

        # --- 1. Assignment Step (Autotuned Triton Kernel) ---
        t_assign_start = time.time()
        assignments_int32 = torch.empty(N_A, dtype=torch.int32, device=device)
        # Call the autotuned kernel - DO NOT pass BLOCK_SIZE_* args here
        kmeans_assign_kernel[grid_assign](
            A_prep, centroids, assignments_int32,
            N_A, D, K,
            A_prep.stride(0), 1,
            centroids.stride(0), 1,
            # Autotuner selects BLOCK_SIZE_N, BLOCK_SIZE_K_CHUNK, BLOCK_SIZE_D
        )
        assignments = assignments_int32.to(torch.int64)
        torch.cuda.synchronize(device=device)
        t_assign_end = time.time()
        assignment_times.append(t_assign_end - t_assign_start)

        # --- 2. Update Step (PyTorch scatter_add_) ---
        t_update_start = time.time()
        new_sums = torch.zeros_like(centroids)
        cluster_counts = torch.zeros(K, dtype=torch.float32, device=device)
        idx_expand = assignments.unsqueeze(1).expand(-1, D)
        new_sums.scatter_add_(dim=0, index=idx_expand, src=A_prep)
        cluster_counts.scatter_add_(dim=0, index=assignments, src=torch.ones_like(assignments, dtype=torch.float32))
        torch.cuda.synchronize(device=device)
        final_counts_safe = cluster_counts.clamp(min=1.0)
        new_centroids = new_sums / final_counts_safe.unsqueeze(1)
        empty_cluster_mask = (cluster_counts == 0)
        new_centroids[empty_cluster_mask] = old_centroids[empty_cluster_mask]
        if not torch.all(torch.isfinite(new_centroids)):
             new_centroids = torch.nan_to_num(new_centroids, nan=old_centroids[torch.isnan(new_centroids)])
        t_update_end = time.time()
        update_times.append(t_update_end - t_update_start)

        # --- Check Convergence ---
        centroid_diff = torch.linalg.norm(new_centroids - centroids)
        centroids = new_centroids

        # Optional: Print progress less often
        # if (i+1) % 10 == 0 or centroid_diff < tol or i == max_iters -1:
        #     print(f"  Iter {i+1}/{max_iters} | Centroid Diff: {centroid_diff:.4f} | Assign Time: {assignment_times[-1]:.4f}s | Update Time: {update_times[-1]:.4f}s")

        if centroid_diff < tol:
            # print(f"Converged after {i+1} iterations.")
            break

    # if i == max_iters - 1: print(f"Reached max iterations ({max_iters}).")
    total_time = time.time() - start_time_total
    # print(f"Total K-Means time: {total_time:.4f}s")

    # --- Basic Benchmarking Output ---
    if assignment_times: print(f"Avg Assign Step Time: {sum(assignment_times)/len(assignment_times):.6f}s")
    if update_times: print(f"Avg Update Step Time: {sum(update_times)/len(update_times):.6f}s")
    # Print the best config found by the autotuner for the assignment kernel
    print("Best config for kmeans_assign_kernel:", kmeans_assign_kernel.best_config)

    return centroids, assignments


# --- ANN Function (User Pseudocode, L2, Using Triton/PyTorch) ---
# Definition of our_ann_user_pseudocode_impl_l2_triton remains the same as before
# It uses distance_l2 internally, which now uses the updated dot kernel
def our_ann_user_pseudocode_impl_l2_triton(N_A, D, A, X, k_clusters, K1, K2,
                                           max_kmeans_iters=100, centroids=None):
    X_prep = _prepare_tensors(X) # Prepare queries
    Q = X_prep.shape[0]
    if X_prep.shape[1] != D: raise ValueError(f"Query dimension mismatch: X D={X_prep.shape[1]}, expected D={D}")
    build_time = 0.0; search_time = 0.0; actual_k_clusters = 0; centroids_used = None
    if centroids is None:
        build_start_time = time.time()
        A_prep = _prepare_tensors(A)
        if A_prep.shape[1] != D: raise ValueError("A data dimension mismatch.")
        centroids_used, _ = our_kmeans(N_A, D, A_prep, k_clusters, max_iters=max_kmeans_iters)
        torch.cuda.synchronize(device=device)
        build_time = time.time() - build_start_time
    else:
        centroids_used = _prepare_tensors(centroids)
        if centroids_used.ndim != 2 or centroids_used.shape[1] != D: raise ValueError(f"Provided centroids invalid shape/dim {centroids_used.shape}.")
    actual_k_clusters = centroids_used.shape[0]
    if actual_k_clusters == 0: # Error handling
        empty_indices = torch.full((Q, K2 if K2 > 0 else 1), -1, dtype=torch.int64, device=device)
        empty_dists = torch.full((Q, K2 if K2 > 0 else 1), float('inf'), dtype=torch.float32, device=device)
        return empty_indices, empty_dists, torch.empty((0,D), dtype=torch.float32, device=device), build_time, 0.0
    K1 = min(K1, actual_k_clusters); K2 = min(K2, K1)
    if K1 <= 0 or K2 <= 0: # Error handling
        empty_indices = torch.full((Q, K2 if K2 > 0 else 1), -1, dtype=torch.int64, device=device)
        empty_dists = torch.full((Q, K2 if K2 > 0 else 1), float('inf'), dtype=torch.float32, device=device)
        return empty_indices, empty_dists, centroids_used, build_time, 0.0

    search_start_time = time.time()
    all_query_centroid_dists_sq = distance_l2(X_prep, centroids_used) # Uses Triton dot
    topk1_dists_sq, topk1_indices = torch.topk(all_query_centroid_dists_sq, k=K1, dim=1, largest=False, sorted=True)
    topk2_subset_dists_sq = topk1_dists_sq[:, :K2]
    topk2_centroid_indices = topk1_indices[:, :K2]
    final_topk2_centroid_indices = topk2_centroid_indices
    final_topk2_centroid_distances_sq = topk2_subset_dists_sq
    torch.cuda.synchronize(device=device)
    search_time = time.time() - search_start_time
    return final_topk2_centroid_indices.to(torch.int64), final_topk2_centroid_distances_sq, centroids_used, build_time, search_time


# ============================================================================
# Main Execution Block (Triton/PyTorch ONLY, Looping Through Dimensions)
# ============================================================================
if __name__ == "__main__":
    # --- Fixed Parameters ---
    N_data = 100000 # Reduced N for faster testing with autotuning overhead
    N_queries_new = 10000
    num_clusters_for_kmeans = 1000 # Reduced K slightly
    K1_probe = 30
    K2_final = 10
    kmeans_max_iters = 50 # Fewer iters to speed up testing

    # --- Dimensions to Test ---
    dimensions_to_test = [16,16,32,2,2,4, 4, 64, 64,256, 256,1024, 1024] # Focus on larger dimensions

    print(f"--- Triton/PyTorch EXECUTION (AUTOTUNED KMEANS) ---")
    print(f"Fixed Params: N={N_data}, Q={N_queries_new}, k_clusters={num_clusters_for_kmeans}, K1={K1_probe}, K2={K2_final}")
    print(f"Testing Dimensions: {dimensions_to_test}")

    # --- Check Device ---
    # (Device check code as before) ...
    if not torch.cuda.is_available(): print("CUDA not available."); exit()
    device = torch.device("cuda:0"); print(f"Using device: {device}")

    # Loop through each dimension
    for Dim in dimensions_to_test:
        print("\n" + "#"*70)
        print(f"# Starting Test for Dimension D = {Dim}")
        print("#"*70)

        # --- Generate Base Data ---
        print("\n" + "="*40); print(f"Generating Base Data (D={Dim}, GPU)..."); print("="*40)
        try:
            A_data = torch.randn(N_data, Dim, dtype=torch.float32, device=device)
            torch.cuda.synchronize(device=device)
            print(f"Database shape (GPU): {A_data.shape}")
        # ... (Error handling for data generation) ...
        except RuntimeError as e: print(f"Error generating A_data: {e}"); torch.cuda.empty_cache(); continue
        except Exception as e: print(f"Error generating A_data: {e}"); continue

        # --- Build Index ONCE (AUTOTUNED GPU KMeans) ---
        print("\n" + "="*40); print(f"Building Centroids via Autotuned KMeans (D={Dim})..."); print("="*40)
        initial_centroids = None; actual_k_clusters = 0; build_time_actual = 0
        try:
            build_start_actual = time.time()
            # Call the AUTOTUNED Triton/PyTorch KMeans function
            initial_centroids, _ = our_kmeans(N_data, Dim, A_data, num_clusters_for_kmeans, max_iters=kmeans_max_iters)
            actual_k_clusters = initial_centroids.shape[0]
            build_time_actual = time.time() - build_start_actual
            if actual_k_clusters == 0: raise ValueError("KMeans returned 0 centroids.")
            print(f"Autotuned KMeans completed. Found {actual_k_clusters} centroids.")
            print(f"Actual Build Time (D={Dim}): {build_time_actual:.4f}s")
        # ... (Error handling for KMeans build) ...
        except RuntimeError as e: print(f"Error during KMeans build: {e}"); torch.cuda.empty_cache(); continue
        except Exception as e: print(f"Error during KMeans build: {e}"); traceback.print_exc(); torch.cuda.empty_cache(); continue


        # --- Generate NEW Queries ---
        print("\n" + "="*40); print(f"Generating {N_queries_new} NEW Test Queries (D={Dim}, GPU)..."); print("="*40)
        try:
            X_queries_new = torch.randn(N_queries_new, Dim, dtype=torch.float32, device=device)
            torch.cuda.synchronize(device=device)
            print(f"New query shape (GPU): {X_queries_new.shape}")
        # ... (Error handling for query generation) ...
        except RuntimeError as e: print(f"Error generating queries: {e}"); torch.cuda.empty_cache(); continue
        except Exception as e: print(f"Error generating queries: {e}"); continue


        # --- Run ANN Search ---
        print("\n" + "="*40); print(f"Testing Triton/PyTorch ANN Search (D={Dim})..."); print("="*40)
        k1_run = min(K1_probe, actual_k_clusters); k2_run = min(K2_final, k1_run)
        print(f"(Using {actual_k_clusters} centroids, K1={k1_run}, K2={k2_run}, L2 Distance, Triton Search)")
        print("="*40)
        ann_indices_centroids = None; ann_dists_sq_centroids = None; centroids_used = None; search_t = 0
        try:
            # Call the ANN function (it uses the optimized distance_l2 internally)
            ann_indices_centroids, ann_dists_sq_centroids, centroids_used, build_t_ignored, search_t = our_ann_user_pseudocode_impl_l2_triton(
                N_A=N_data, D=Dim, A=A_data, X=X_queries_new, k_clusters=actual_k_clusters,
                K1=k1_run, K2=k2_run, centroids=initial_centroids )
            print("Triton ANN results shape (Centroid Indices):", ann_indices_centroids.shape)
            print("Triton ANN results shape (**SQUARED** L2 Distances):", ann_dists_sq_centroids.shape)
            print(f"Triton ANN Search Time (D={Dim}): {search_t:.4f}s")
            if search_t > 0: print(f"-> Throughput: {N_queries_new / search_t:.2f} queries/sec (GPU)")
        # ... (Error handling for ANN search) ...
        except RuntimeError as e: print(f"Error during ANN execution: {e}"); traceback.print_exc(); ann_indices_centroids = None
        except Exception as e: print(f"Error during ANN execution: {e}"); traceback.print_exc(); ann_indices_centroids = None


        # --- Optional: Calculate Recall ---
        if ann_indices_centroids is not None and centroids_used is not None and centroids_used.shape[0] > 0:
            print("\n" + "="*40); print(f"Calculating Recall (D={Dim}, Triton/PyTorch L2)..."); print("="*40)
            K_recall = k2_run
            try:
                # Ground truth calculation using distance_l2
                start_gt = time.time()
                all_query_centroid_dists_sq_gt = distance_l2(X_queries_new, centroids_used)
                actual_num_centroids = centroids_used.shape[0]
                k_recall_adjusted = min(K_recall, actual_num_centroids)

                if k_recall_adjusted > 0:
                    true_knn_centroid_indices = torch.topk(all_query_centroid_dists_sq_gt, k=k_recall_adjusted, dim=1, largest=False, sorted=True)[1]
                else: true_knn_centroid_indices = torch.empty((N_queries_new, 0), dtype=torch.int64, device=device)
                torch.cuda.synchronize(device=device)
                # print(f"Ground truth calculation time: {time.time() - start_gt:.4f}s")

                # Compare results
                total_intersect_centroids = 0
                ann_indices_np = ann_indices_centroids[:, :k_recall_adjusted].cpu().numpy()
                true_indices_np = true_knn_centroid_indices.cpu().numpy()
                for i in range(N_queries_new):
                     approx_centroid_ids = set(idx for idx in ann_indices_np[i] if idx >= 0)
                     true_centroid_ids = set(true_indices_np[i])
                     total_intersect_centroids += len(approx_centroid_ids.intersection(true_centroid_ids))

                if N_queries_new > 0 and k_recall_adjusted > 0:
                    avg_recall_centroids = total_intersect_centroids / (N_queries_new * k_recall_adjusted)
                    print(f"\nAverage Recall @ {k_recall_adjusted} (vs Triton/Torch brute-force CENTROIDS w/ L2, D={Dim}): {avg_recall_centroids:.4f} ({avg_recall_centroids:.2%})")
                    # Use corrected commentary logic
                    epsilon = 1e-9
                    if abs(avg_recall_centroids - 1.0) < epsilon: print("Result: 100% recall indicates K1 was large enough...")
                    else: print(f"Result: Recall ({avg_recall_centroids:.4f}) < 100%...")
                else: print("\nCannot calculate recall.")

            # ... (Error handling for Recall) ...
            except RuntimeError as e: print(f"Error during recall calculation: {e}"); traceback.print_exc()
            except Exception as e: print(f"Error during recall calculation: {e}"); traceback.print_exc()
        else: print("\nSkipping recall calculation.")

        print(f"\n--- Finished Test for Dimension D = {Dim} ---")
        # Clean up GPU memory
        del A_data, X_queries_new, initial_centroids, centroids_used
        if 'ann_indices_centroids' in locals(): del ann_indices_centroids
        if 'ann_dists_sq_centroids' in locals(): del ann_dists_sq_centroids
        if 'all_query_centroid_dists_sq_gt' in locals(): del all_query_centroid_dists_sq_gt
        if 'true_knn_centroid_indices' in locals(): del true_knn_centroid_indices
        torch.cuda.empty_cache()

    print("\n--- ALL Triton/PyTorch DIMENSION TESTS FINISHED ---")

    # --- How to use Triton Benchmarking Separately ---
    print("\n" + "="*50)
    print("Benchmarking Information:")
    print("="*50)
    print("To benchmark the 'kmeans_assign_kernel' more formally, you would typically:")
    print("1. Create a separate Python script.")
    print("2. Import necessary modules (torch, triton, triton.testing).")
    print("3. Define or import the 'kmeans_assign_kernel' WITH the '@triton.autotune' decorator.")
    print("4. Create sample input tensors (A, centroids, assignments_out) with relevant shapes and dtypes.")
    print("5. Define the grid calculation logic.")
    print("6. Call 'triton.testing.do_bench(lambda: kmeans_assign_kernel[grid](...), quantiles=...)'")
    print("   - The lambda captures the kernel call with specific inputs.")
    print("   - 'quantiles' can provide median, 0.2, 0.8 percentile timings.")
    print("7. You can wrap this in loops to test different N, D, K values.")
    print("8. Use '@triton.testing.perf_report' to format results if desired.")
    print("Note: Running 'do_bench' measures execution time, separate from autotuning which finds the best config.")
    print("The autotuner runs its own short benchmarks internally when first called.")
    print("="*50)