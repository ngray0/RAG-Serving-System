import torch
import cupy as cp
import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann

import time

def distance_cosine(X, Y):
    # Compute norms for each row
    norm_X = cp.linalg.norm(X, axis=1, keepdims=True)
    norm_Y = cp.linalg.norm(Y, axis=1, keepdims=True)
    # Compute cosine similarity and convert to cosine distance
    cosine_similarity = cp.sum(X * Y, axis=1) / (norm_X.flatten() * norm_Y.flatten())
    return 1 - cosine_similarity

def distance_l2(X, Y):
    # Direct L2 norm computation
    return cp.linalg.norm(X - Y, axis=1)

def distance_dot(X, Y):
    # Compute row-wise dot product
    return cp.sum(X * Y, axis=1)

def distance_manhattan(X, Y):
    # Compute Manhattan (L1) distance row-wise
    return cp.sum(cp.abs(X - Y), axis=1)


# ------------------------------------------------------------------------------------------------
# Your Task 1.2 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here

def our_knn(N, D, A, X, K):
    pass

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


if __name__ == "__main__":
    main()
