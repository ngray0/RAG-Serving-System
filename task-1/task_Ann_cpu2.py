import numpy as np
import time
import traceback # For error printing

# ============================================================================
# CPU Helper Function (Pairwise Squared L2 Distance - Provided by User)
# ============================================================================

def pairwise_l2_squared_numpy(X_np, C_np):
    """
    Computes pairwise **squared** L2 distances using NumPy.
    X_np: (N, D) data points OR (Q, D) query points (NumPy array)
    C_np: (K, D) centroids OR (N, D) database points (NumPy array)
    Returns: (N|Q, K|N) NumPy array of SQUARED distances.
    """
    # Ensure inputs are NumPy arrays and float32
    if not isinstance(X_np, np.ndarray): X_np = np.asarray(X_np, dtype=np.float32)
    elif X_np.dtype != np.float32: X_np = X_np.astype(np.float32)
    if not isinstance(C_np, np.ndarray): C_np = np.asarray(C_np, dtype=np.float32)
    elif C_np.dtype != np.float32: C_np = C_np.astype(np.float32)

    if X_np.ndim == 1: X_np = X_np[np.newaxis, :] # Ensure X is 2D
    if C_np.ndim == 1: C_np = C_np[np.newaxis, :] # Ensure C is 2D

    # Handle empty inputs gracefully
    if X_np.shape[0] == 0 or C_np.shape[0] == 0:
        return np.empty((X_np.shape[0], C_np.shape[0]), dtype=np.float32)

    # Check dimension compatibility AFTER ensuring 2D and non-empty
    if X_np.shape[1] != C_np.shape[1]:
        raise ValueError(f"Dimension mismatch: X_np has D={X_np.shape[1]}, C_np has D={C_np.shape[1]}")

    # ||x - c||^2 = ||x||^2 - 2<x, c> + ||c||^2
    try:
        X_norm_sq = np.einsum('ij,ij->i', X_np, X_np)[:, np.newaxis] # Shape (N|Q, 1)
        C_norm_sq = np.einsum('ij,ij->i', C_np, C_np)[np.newaxis, :] # Shape (1, K|N)
        # Use np.dot for matrix multiplication
        dot_products = np.dot(X_np, C_np.T) # Shape (N|Q, K|N)

        # Broadcasting: (N|Q, 1) + (1, K|N) - 2*(N|Q, K|N) -> (N|Q, K|N)
        dist_sq = X_norm_sq + C_norm_sq - 2 * dot_products
        return np.maximum(0.0, dist_sq) # Clamp numerical negatives

    except MemoryError as e:
        print(f"MemoryError in pairwise_l2_squared_numpy: Shapes X={X_np.shape}, C={C_np.shape}")
        dot_prod_mem_gb = X_np.shape[0] * C_np.shape[0] * 4 / (1024**3) # GB
        print(f"Estimated memory for dot product matrix: {dot_prod_mem_gb:.2f} GB")
        raise e # Re-raise the exception
    except Exception as e:
        print(f"Error in pairwise_l2_squared_numpy: {e}")
        raise e


# ============================================================================
# CPU Brute-Force k-NN (NumPy version - WITH QUERY BATCHING)
# ============================================================================
def numpy_knn_bruteforce(N_A, D, A_np, X_np, K, batch_size_q=1024): # Add batch_size_q parameter
    """
    Finds the K nearest neighbors using brute-force NumPy distance calculation.
    Uses query batching to handle potentially large Q * N distance matrices in RAM.
    Returns original indices (int64) and SQUARED L2 distances (float32).
    Handles K > N_A by padding results.

    Args:
        N_A (int): Number of database points (inferred).
        D (int): Dimension (inferred).
        A_np (np.ndarray): Database vectors (N_A, D) NumPy array.
        X_np (np.ndarray): Query vectors (Q, D) NumPy array.
        K (int): Number of neighbors to find.
        batch_size_q (int): Number of queries to process in each batch.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - all_topk_indices_np (np.ndarray): Indices (Q, K), int64.
            - all_topk_distances_sq_np (np.ndarray): Squared distances (Q, K), float32.
    """
    # --- Input Validation ---
    if not isinstance(A_np, np.ndarray): A_np = np.asarray(A_np, dtype=np.float32)
    elif A_np.dtype != np.float32: A_np = A_np.astype(np.float32)
    if not isinstance(X_np, np.ndarray): X_np = np.asarray(X_np, dtype=np.float32)
    elif X_np.dtype != np.float32: X_np = X_np.astype(np.float32)

    if A_np.ndim != 2: raise ValueError(f"Database A_np must be 2D (N, D), got shape {A_np.shape}")
    if X_np.ndim != 2: raise ValueError(f"Queries X_np must be 2D (Q, D), got shape {X_np.shape}")

    actual_N_A, actual_D = A_np.shape
    Q, query_D = X_np.shape

    if query_D != actual_D: raise ValueError(f"Dimension mismatch: A_np D={actual_D}, X_np D={query_D}")

    # Use actual N_A
    N_A = actual_N_A

    # Handle empty database or query set
    if N_A == 0:
         print("Warning: CPU Brute force called with empty database A_np.")
         return np.full((Q, K), -1, dtype=np.int64), np.full((Q, K), np.inf, dtype=np.float32)
    if Q == 0:
         print("Warning: CPU Brute force called with empty query set X_np.")
         return np.empty((0, K), dtype=np.int64), np.empty((0, K), dtype=np.float32)

    if not K > 0: raise ValueError("K must be positive")

    # Adjust K if it's larger than the number of data points
    effective_K = min(K, N_A)
    if effective_K != K:
         print(f"Note: CPU Brute force K={K} requested > N_A={N_A}. Using K={effective_K}.")

    if effective_K == 0: # Handle K=0 or effective_K=0 case
         return np.empty((Q, 0), dtype=np.int64), np.empty((Q, 0), dtype=np.float32)

    print(f"Running k-NN Brute Force (NumPy CPU, Batched): Q={Q}, N={N_A}, D={actual_D}, K={effective_K}, BatchSize={batch_size_q}")
    start_time = time.time()

    # Pre-allocate result arrays
    all_topk_indices_np = np.full((Q, effective_K), -1, dtype=np.int64)
    all_topk_distances_sq_np = np.full((Q, effective_K), np.inf, dtype=np.float32)

    # --- Batch Processing Loop ---
    for q_start in range(0, Q, batch_size_q):
        q_end = min(q_start + batch_size_q, Q)
        batch_q_indices = slice(q_start, q_end) # Slice for indexing results
        X_batch_np = X_np[batch_q_indices]      # Current batch of queries
        current_batch_size = X_batch_np.shape[0]
        if current_batch_size == 0: continue

        # print(f"  Processing query batch {q_start}-{q_end-1}...") # Optional progress

        # Calculate SQUARED L2 distances for the current batch
        try:
            batch_distances_sq = pairwise_l2_squared_numpy(X_batch_np, A_np) # Shape (current_batch_size, N_A)
        except MemoryError as e:
            batch_mem_gb = current_batch_size * N_A * 4 / (1024**3)
            print(f"MemoryError during CPU Brute Force batch distance calculation:")
            print(f"  Batch Q={current_batch_size}, N={N_A}. Estimated matrix memory: {batch_mem_gb:.2f} GB")
            print(f"  Try reducing batch_size_q (current={batch_size_q}) or ensure sufficient RAM.")
            raise e # Re-raise
        except Exception as e:
            print(f"Error during CPU Brute Force batch distance calculation: {e}")
            raise e

        # Find top K for the current batch using NumPy partition/sort
        # Need kth = K-1 for argpartition (0-based)
        k_partition_idx = min(effective_K, N_A) - 1 # N_A because np allows K=N_A; -1 for 0-based index
        if k_partition_idx < 0: k_partition_idx = 0 # Handle N_A=0 or K=1 case

        batch_topk_indices = None
        batch_topk_distances_sq = None

        try:
            if effective_K >= N_A: # If K includes all points, just sort all
                batch_topk_indices = np.argsort(batch_distances_sq, axis=1)[:, :effective_K]
            else:
                 # Ensure N_A > 0 before calling argpartition
                 if N_A > 0:
                      # Partition to find indices of K smallest (unsorted within K)
                      topk_indices_unstructured = np.argpartition(batch_distances_sq, kth=k_partition_idx, axis=1)[:, :effective_K]
                      # Get the distances for these K indices
                      topk_distances_sq_unstructured = np.take_along_axis(batch_distances_sq, topk_indices_unstructured, axis=1)
                      # Sort within the K elements based on distance
                      sorted_order_in_k = np.argsort(topk_distances_sq_unstructured, axis=1)
                      # Apply sort order to get final indices and distances
                      batch_topk_indices = np.take_along_axis(topk_indices_unstructured, sorted_order_in_k, axis=1)
                 else: # Should not happen if N_A=0 check passed
                      batch_topk_indices = np.empty((current_batch_size, 0), dtype=np.int64)

            # Retrieve the final sorted distances AFTER getting the sorted indices
            if batch_topk_indices is not None and batch_topk_indices.size > 0:
                 batch_topk_distances_sq = np.take_along_axis(batch_distances_sq, batch_topk_indices, axis=1)
            elif effective_K == 0:
                 batch_topk_distances_sq = np.empty((current_batch_size, 0), dtype=np.float32)
            else: # Handle unexpected empty indices
                 print(f"Warning: batch_topk_indices empty/None in batch {q_start}-{q_end-1}")
                 batch_topk_distances_sq = np.full((current_batch_size, effective_K), np.inf, dtype=np.float32)
                 if batch_topk_indices is None: batch_topk_indices = np.full((current_batch_size, effective_K), -1, dtype=np.int64)

            # Store batch results in the main arrays
            all_topk_indices_np[batch_q_indices] = batch_topk_indices
            all_topk_distances_sq_np[batch_q_indices] = batch_topk_distances_sq

            # Optional: Clear intermediate batch results (less critical on CPU usually)
            del batch_distances_sq, batch_topk_indices, batch_topk_distances_sq
            if 'topk_indices_unstructured' in locals(): del topk_indices_unstructured
            if 'topk_distances_sq_unstructured' in locals(): del topk_distances_sq_unstructured
            if 'sorted_order_in_k' in locals(): del sorted_order_in_k

        except Exception as e:
            print(f"Error during CPU Brute Force batch top-K ({q_start}-{q_end-1}): {e}")
            raise e # Raise error to stop potentially incorrect recall calculation
    # --- End Batch Loop ---

    end_time = time.time()
    print(f"k-NN Brute Force (NumPy CPU, Batched) total computation time: {end_time - start_time:.4f} seconds")

    # Pad results if original K > effective_K (i.e., K > N_A)
    if K > effective_K:
        pad_width = K - effective_K
        # Ensure results are not None before padding
        if all_topk_indices_np is None: all_topk_indices_np = np.full((Q, effective_K), -1, dtype=np.int64)
        if all_topk_distances_sq_np is None: all_topk_distances_sq_np = np.full((Q, effective_K), np.inf, dtype=np.float32)

        indices_pad = np.full((Q, pad_width), -1, dtype=np.int64)
        dists_pad = np.full((Q, pad_width), np.inf, dtype=np.float32)
        all_topk_indices_np = np.hstack((all_topk_indices_np, indices_pad))
        all_topk_distances_sq_np = np.hstack((all_topk_distances_sq_np, dists_pad))

    # Ensure final return types are correct
    if all_topk_indices_np is None: all_topk_indices_np = np.full((Q, K), -1, dtype=np.int64)
    if all_topk_distances_sq_np is None: all_topk_distances_sq_np = np.full((Q, K), np.inf, dtype=np.float32)

    return all_topk_indices_np.astype(np.int64), all_topk_distances_sq_np.astype(np.float32)


# ============================================================================
# Example Usage (Optional - How you would call it)
# ============================================================================
if __name__ == "__main__":

    # Assuming you have already defined pairwise_l2_squared_numpy somewhere above

    print("\n--- Example CPU k-NN Brute Force Usage ---")
    # Create some dummy data
    N_cpu = 10000 # Smaller N for quick CPU example
    D_cpu = 64
    Q_cpu = 500
    K_cpu = 5

    A_data_cpu = np.random.rand(N_cpu, D_cpu).astype(np.float32)
    X_queries_cpu = np.random.rand(Q_cpu, D_cpu).astype(np.float32)

    print(f"Example Data: N={N_cpu}, D={D_cpu}, Q={Q_cpu}, K={K_cpu}")

    try:
        # Call the batched CPU k-NN function
        # Use a smaller batch size suitable for CPU testing if desired
        cpu_knn_indices, cpu_knn_dists_sq = numpy_knn_bruteforce(
            N_A=N_cpu, D=D_cpu, A_np=A_data_cpu, X_np=X_queries_cpu, K=K_cpu, batch_size_q=512
        )

        print("\nCPU Brute Force Results:")
        print(f"  Indices shape: {cpu_knn_indices.shape}") # Should be (Q_cpu, K_cpu)
        print(f"  Sq Distances shape: {cpu_knn_dists_sq.shape}")
        # print("  First 5 indices:", cpu_knn_indices[0,:5])
        # print("  First 5 distances:", cpu_knn_dists_sq[0,:5])

    except MemoryError:
        print("\nMemoryError encountered during CPU k-NN example.")
        print("Reduce N_cpu or Q_cpu, or ensure sufficient RAM.")
    except Exception as e:
        print(f"\nAn error occurred during CPU k-NN example: {e}")
        traceback.print_exc()