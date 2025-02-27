import torch
import cupy as cp
import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann

import time

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
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_knn(N, D, A, X, K, distance_func, batch_size=1024, num_streams=4):
    """
    Parameters:
      N: Number of training examples.
      D: Dimensionality of the data.
      A: Training data, a (N, D) Cupy array.
      X: Query data, a (M, D) Cupy array.
      K: Number of nearest neighbors to return.
      distance_func: A callable that computes distances between a training set A and a single query vector.
                     It should accept two arguments: the training set (shape (N, D)) and a query (shape (1, D)),
                     and return a 1D Cupy array of distances of length N.
      batch_size: How many queries to process per batch.
      num_streams: Number of CUDA streams to use for overlapping computations.
    
    Returns:
      knn_indices: (M, K) array with indices of the K nearest training points for each query.
      knn_distances: (M, K) array with corresponding distances.
    """
    M = X.shape[0]
    knn_indices = cp.empty((M, K), dtype=cp.int32)
    knn_distances = cp.empty((M, K), dtype=A.dtype)
    
    streams = [cp.cuda.Stream() for _ in range(num_streams)]
    
    for batch_start in range(0, M, batch_size):
        batch_end = min(batch_start + batch_size, M)
        for i in range(batch_start, batch_end):
            stream = streams[i % num_streams]
            with stream:
                query = X[i:i+1, :]
                dists = distance_func(A, query)
                idx_part = cp.argpartition(dists, K)[:K]
                selected_dists = dists[idx_part]
                sorted_order = cp.argsort(selected_dists)
                sorted_idx = idx_part[sorted_order]
                sorted_dists = dists[sorted_idx]
                knn_indices[i] = sorted_idx
                knn_distances[i] = sorted_dists
    cp.cuda.Stream.null.synchronize()
    
    return knn_indices, knn_distances

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
'''
def test_kmeans():
    N, D, A, K = testdata_kmeans("test_file.json")
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
'''
def main():
    np.random.seed(0)
    #size = 1 << 20  # 1 million elements
    n = 1000
    d = 100
    # Transfer to GPU.
    A_cp = cp.random.rand(n, d).astype(cp.float32)
    B_cp = cp.random.rand(n, d).astype(cp.float32)
    
    start_time = time.time()
    l2 = distance_l2(A_cp, B_cp)
    end_time = time.time()
    print(f"L2 distance computation time: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    dot = distance_dot(A_cp, B_cp)
    end_time = time.time()
    print(f"Dot Product distance computation time: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    manhattan = distance_manhattan(A_cp, B_cp)
    end_time = time.time()
    print(f"Manhattan distance computation time: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    cosine = distance_cosine(A_cp, B_cp)
    end_time = time.time()
    print(f"Cosine distance computation time: {end_time - start_time:.4f} seconds")

    
    print("L2 (Euclidean) distance:", l2)
    print("Dot product:", dot)
    print("Manhattan (L1) distance:", manhattan)
    print("Cosine distance:", cosine)

    knn_idx, knn_dists = our_knn(A_cp.shape[0], A_cp.shape[1], A_cp, B_cp, 5, distance_l2)
    print(knn_idx, knn_dists)
    knn_idx, knn_dists = our_knn(A_cp.shape[0], A_cp.shape[1], A_cp, B_cp, 5, distance_cosine)
    print(knn_idx, knn_dists)
    knn_idx, knn_dists = our_knn(A_cp.shape[0], A_cp.shape[1], A_cp, B_cp, 5, distance_manhattan)
    print(knn_idx, knn_dists)
    knn_idx, knn_dists = our_knn(A_cp.shape[0], A_cp.shape[1], A_cp, B_cp, 5, distance_dot)
    print(knn_idx, knn_dists)


if __name__ == "__main__":
    main()
