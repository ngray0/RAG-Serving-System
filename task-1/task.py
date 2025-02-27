import torch
import cupy as cp
import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
from cupyx.jit import rawkernel
import time

def distance_l2(X, Y):
    X_norm = cp.sum(X * X, axis=1).reshape(-1, 1)
    Y_norm = cp.sum(Y * Y, axis=1).reshape(1, -1)
    dist_sq = cp.maximum(X_norm + Y_norm - 2 * cp.dot(X, Y.T), 0.0)
    return cp.sqrt(dist_sq)


def distance_cosine(X, Y):
    X_norm = X / cp.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = Y / cp.linalg.norm(Y, axis=1, keepdims=True)
    cosine_similarity = cp.dot(X_norm, Y_norm.T)
    return 1 - cosine_similarity

def distance_dot(X, Y):
    return cp.dot(X, Y.T)

def distance_manhattan(X, Y):
    return cp.sum(cp.abs(X[:, cp.newaxis, :] - Y[cp.newaxis, :, :]), axis=2)


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
    size = (1000, 100)
    # Create two random vectors.
    A_np = np.random.rand(*size).astype(np.float32)
    B_np = np.random.rand(*size).astype(np.float32)
    # Transfer to GPU.
    A_cp = cp.array(A_np)
    B_cp = cp.array(B_np)
    
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
