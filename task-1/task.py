import torch
import cupy as cp
import triton
import triton.language as tl
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
import time
import csv
import os


# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------
# Q1: How did you implement four distinct distance functions on the GPU?
# ------------------------------------------------------------------------------------------------

# Benchmark
def distance_cosine(X, Y):
    norm_X = cp.linalg.norm(X, axis=1, keepdims=True) 
    norm_Y = cp.linalg.norm(Y, axis=1, keepdims=True)
    cosine_similarity = cp.sum(X * Y, axis=1) / (norm_X.flatten() * norm_Y.flatten())
    cosine_distance = 1 - cosine_similarity
    return cosine_distance
def distance_l2(X, Y):
    squared_diff = cp.sum((X - Y) ** 2, axis=1)
    l2_distance = cp.sqrt(squared_diff)
    return l2_distance

def distance_dot(X, Y):
    dot_product = cp.sum(X * Y, axis=1)
    return dot_product

def distance_manhattan(X, Y):
    manhattan_distance = cp.sum(cp.abs(X - Y), axis=1)
    return manhattan_distance


# ------------------------------------------------------------------------------------------------
# Q2: What is the speed advantage of the GPU over the CPU version when the dimension is 2?
# Additionally, what is the speed advantage when the dimension is 2^15?
# ------------------------------------------------------------------------------------------------

# Benchmark 

def distance_cosine_cpu(X, Y):
    norm_X = np.linalg.norm(X, axis=1, keepdims=True)
    norm_Y = np.linalg.norm(Y, axis=1, keepdims=True)
    cosine_similarity = np.sum(X * Y, axis=1) / (norm_X.flatten() * norm_Y.flatten())
    return 1 - cosine_similarity

def distance_l2_cpu(X, Y):
    squared_diff = np.sum((X - Y) ** 2, axis=1)
    l2_distance = np.sqrt(squared_diff)
    return l2_distance

def distance_dot_cpu(X, Y):
    dot_product = np.sum(X * Y, axis=1)
    return dot_product

def distance_manhattan_cpu(X, Y):
    manhattan_distance = np.sum(np.abs(X - Y), axis=1)
    return manhattan_distance

# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def compute_all_distances(A, X, distance_fn):
    """
    Compute the distances between each row in array A and the vector X using the provided distance function.
    
    If X is a 1D vector, it is reshaped to 2D for broadcasting.
    
    Parameters:
        A (cupy.ndarray): 2D array of shape (N, D)
        X (cupy.ndarray): 1D or 2D array representing the query vector.
        distance_fn (function): A function that computes the distance given two arrays.
        
    Returns:
        cupy.ndarray: An array of distances computed between each row of A and X.
    """
    if X.ndim == 1:
        X = X[None, :]
    return distance_fn(A, cp.broadcast_to(X, A.shape))

# Benchmark
def our_knn(N, D, A, X, K):
    """
    Compute the K nearest neighbors indices for a query vector X from the dataset A using L2 distance.
    
    The function expects:
      - N: number of vectors in A.
      - D: dimension of each vector.
      - A: dataset array of shape (N, D) stored on the GPU.
      - X: query vector of dimension D.
      - K: number of nearest neighbors to retrieve.
    
    This function computes the distances between the query and each vector in A using L2 distance,
    then uses a partial sort (via cp.argpartition) to select the K smallest distances,
    and finally orders these K indices by their actual distances.
    
    Parameters:
        N (int): Number of vectors.
        D (int): Dimension of each vector.
        A (cupy.ndarray): 2D array of shape (N, D) containing the dataset vectors.
        X (cupy.ndarray): 1D array of shape (D,) or 2D array (1, D) representing the query vector.
        K (int): Number of nearest neighbors to retrieve.
        
    Returns:
        cupy.ndarray: 1D array containing the indices of the K nearest neighbors, ordered by distance.
    """
    distances = compute_all_distances(A, X, distance_l2)
    k_indices = cp.argpartition(distances, K)[:K]
    k_indices = k_indices[cp.argsort(distances[k_indices])]
    return k_indices

# CuPy + CUDA Streams
def our_knn_stream(N, D, A, X, K):
    """
    Optimized KNN using CUDA Streams in CuPy.
    This allows concurrent computation of multiple queries.

    N: Number of vectors in A
    D: Dimension of each vector
    A: Dataset array (CuPy array on GPU)
    X: Query vector (CuPy array on GPU)
    K: Number of nearest neighbors
    """
    B = X.shape[0] if X.ndim > 1 else 1  # Determine batch size
    streams = [cp.cuda.Stream() for _ in range(B)]
    results = [None] * B

    for i in range(B):
        with streams[i]:
            query = X[i] if X.ndim > 1 else X
            distances = cp.sqrt(cp.sum((A - query) ** 2, axis=1))
            k_indices = cp.argpartition(distances, K)[:K]
            results[i] = k_indices[cp.argsort(distances[k_indices])]

    for s in streams:
        s.synchronize()
    
    return results if B > 1 else results[0]

# Triton

@triton.jit
def knn_kernel(A_ptr, X_ptr, dist_ptr, N: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    Triton kernel to compute L2 distances between each row in A and X.
    Uses shared memory tiling to reduce redundant accesses.
    """
    row_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = row_idx < N
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for i in range(0, D, 128):
        x_tile = tl.load(X_ptr + i, mask=tl.arange(0, 128) < (D - i), other=0.0)
        a_tile = tl.load(A_ptr + row_idx[:, None] * D + i, mask=mask[:, None], other=0.0)
        acc += tl.sum((a_tile - x_tile) ** 2, axis=1)

    tl.store(dist_ptr + row_idx, tl.sqrt(acc), mask=mask)

def our_knn_triton(N, D, A, X, K):
    """
    Optimized KNN using Triton Kernel with Shared Memory Tiling.
    N: Number of vectors
    D: Dimension of each vector
    A: Dataset (CuPy array)
    X: Query (CuPy array)
    K: Number of nearest neighbors
    """
    distances = cp.empty((N,), dtype=cp.float32)
    grid = (N + 127) // 128

    knn_kernel[grid](A, X, distances, N, D, 128)
    cp.cuda.Device(0).synchronize()

    k_indices = cp.argpartition(distances, K)[:K]
    return k_indices[cp.argsort(distances[k_indices])]

# Hierachy Memory
def our_knn_hierachy(N, D, A, X, K):
    """
    Optimized KNN using Pinned Memory for Efficient Data Transfer.
    Pinned memory allows fast CPU-GPU transfers for batch operations.
    """
    if not isinstance(A, cp.ndarray):
        A = cp.asarray(A)  # Ensure A is on GPU
    if not isinstance(X, cp.ndarray):
        X = cp.asarray(X)  # Ensure X is on GPU

    distances = cp.empty((N,), dtype=cp.float32)

    # Pinned memory for efficient transfers
    X_pinned = cp.get_pinned_memory(X.nbytes)
    cp.cuda.runtime.memcpyAsync(X_pinned, X.data.ptr, X.nbytes, cp.cuda.runtime.memcpyHostToDevice)

    distances = cp.sqrt(cp.sum((A - X) ** 2, axis=1))
    k_indices = cp.argpartition(distances, K)[:K]

    return k_indices[cp.argsort(distances[k_indices])]

def generate_data(N, D, K, output_dir="test_data", seed=42):
    """
    Generate random dataset A, query vector X, and save them as .txt files.
    Also, create a JSON file referencing these files. Ensures reproducibility with a random seed.
    Does not regenerate data if files already exist.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define file paths
    A_file = os.path.join(output_dir, f"A_{N}_{D}.txt")
    X_file = os.path.join(output_dir, f"X_{D}.txt")
    json_file = os.path.join(output_dir, f"test_{N}_{D}_{K}.json")

    # Check if data already exists
    if os.path.exists(A_file) and os.path.exists(X_file) and os.path.exists(json_file):
        print(f"Data already exists. Skipping generation.")
        return
    
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Generate random dataset A (N x D)
    A = np.random.rand(N, D).astype(np.float32)
    np.savetxt(A_file, A)

    # Generate random query vector X (D,)
    X = np.random.rand(D).astype(np.float32)
    np.savetxt(X_file, X)

    # Create JSON data
    json_data = {
        "n": N,
        "d": D,
        "k": K,
        "a_file": A_file,
        "x_file": X_file
    }

    # Save JSON file
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"Generated JSON file: {json_file}")

def benchmark_knn_with_distances(test_file, filename="knn_benchmark_results.csv"):
    """
    Benchmark the performance of different KNN implementations with various distance functions.
    Uses the testdata_knn function to load data from a JSON file or generate random data.
    """
    results = []
    N, D, A, X, K = testdata_knn(test_file)
    print(f"\nBenchmarking for N={N}, D={D}, K={K}:")

    # Ensure data is on GPU
    A_gpu = cp.asarray(A)
    X_gpu = cp.asarray(X)

    # Benchmark each KNN implementation with each distance function
    for knn_name, knn_func in [
        ("CuPy", our_knn),
        ("CuPy Streams", our_knn_stream),
        ("Triton", our_knn_triton),
        ("Hierarchical Memory", our_knn_hierachy)
    ]:
        for distance_name, distance_fn in [
            ("Cosine", distance_cosine),
            ("L2", distance_l2),
            ("Dot", distance_dot),
            ("Manhattan", distance_manhattan)
        ]:
            # Modify the KNN function to use the specified distance function
            def knn_with_distance(N, D, A, X, K):
                distances = compute_all_distances(A, X, distance_fn)
                k_indices = cp.argpartition(distances, K)[:K]
                return k_indices[cp.argsort(distances[k_indices])]

            start = time.time()
            _ = knn_with_distance(N, D, A_gpu, X_gpu, K)
            cp.cuda.Stream.null.synchronize()
            elapsed_time = time.time() - start

            print(f"{knn_name} + {distance_name}: {elapsed_time:.6f}s")
            results.append([N, D, K, knn_name, distance_name, elapsed_time])

    # Write results to CSV
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)


def main():
    # Define test configurations
    test_configs = [
        {"N": 4000, "D": 100, "K": 10},
        {"N": 4000000, "D": 100, "K": 10},
        {"N": 4000, "D": 2**15, "K": 10},
        # {"N": 4000000, "D": 2**15, "K": 10}
    ]

    # Generate data for each configuration
    for config in test_configs:
        generate_data(config["N"], config["D"], config["K"])

    # Test configurations (JSON files or empty string for random data)
    test_files = [
        "test_data/test_4000_100_10.json",
        "test_data/test_4000000_100_10.json",
        "test_data/test_4000_32768_10.json",
        # "test_data/test_4000000_32768_10.json",
        # ""  # Random data
    ]

    # Write CSV header
    filename = "knn_benchmark_results.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["N", "D", "K", "KNN Implementation", "Distance Function", "Time (s)"])

    # Run benchmarks for each test file
    for test_file in test_files:
        benchmark_knn_with_distances(test_file, filename)

if __name__ == "__main__":
    main()



# ------------------------------------------------------------------------------------------------
# Your Task 2.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

def our_kmeans(N, D, A, K):
    pass

# ------------------------------------------------------------------------------------------------
# Your Task 2.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_ann(N, D, A, X, K):
    pass

# ------------------------------------------------------------------------------------------------
# Test your code here
# ------------------------------------------------------------------------------------------------

# Example
def test_kmeans():
    N, D, A, X, K = testdata_kmeans("test_file.json")
    kmeans_result = our_kmeans(N, D, A, K)
    print(kmeans_result)

def test_knn():
    N, D, A, X, K = testdata_knn("test_file.json")
    knn_result = our_knn(N, D, A, X, K)
    print(knn_result)
    
def test_ann():
    N, D, A, X, K = testdata_ann("test_file.json")
    ann_result = our_ann(N, D, A, X, K)
    print(ann_result)
    
def recall_rate(list1, list2):
    """
    Calculate the recall rate of two lists
    list1[K]: The top K nearest vectors ID
    list2[K]: The top K nearest vectors ID
    """
    return len(set(list1) & set(list2)) / len(list1)

