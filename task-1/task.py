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

def benchmark_distance(d, batch_sizes, filename="benchmark_results.csv"):
    results = []
    print(f"\nBenchmarking for dimension {d}:")
    
    for n in batch_sizes:
        X_cpu = np.random.rand(n, d).astype(np.float32)
        Y_cpu = np.random.rand(n, d).astype(np.float32)
        X_gpu = cp.array(X_cpu)
        Y_gpu = cp.array(Y_cpu)
        
        # CPU and GPU benchmarks for each distance function
        for distance_name, cpu_func, gpu_func in [
            ("Cosine", distance_cosine_cpu, distance_cosine),
            ("L2", distance_l2_cpu, distance_l2),
            ("Dot", distance_dot_cpu, distance_dot),
            ("Manhattan", distance_manhattan_cpu, distance_manhattan)
        ]:
            start = time.time()
            _ = cpu_func(X_cpu, Y_cpu)
            cpu_time = time.time() - start
            
            start = time.time()
            _ = gpu_func(X_gpu, Y_gpu)
            cp.cuda.Stream.null.synchronize()
            gpu_time = time.time() - start
            
            print(f"Batch size {n}, {distance_name}: CPU={cpu_time:.6f}s, GPU={gpu_time:.6f}s")
            results.append([d, n, distance_name, cpu_time, gpu_time])
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)

def main():
    batch_sizes = [1, 10, 100, 1000, 10000]  
    filename = "result/q2_benchmark_distance.csv"
    
    # Write CSV header
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Dimension", "Batch Size", "Distance Function", "CPU Time (s)", "GPU Time (s)"])
    
    for d in [2, 2**15]:  #
        benchmark_distance(d, batch_sizes, filename)

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

if __name__ == "__main__":
    # main()
    test_kmeans()
