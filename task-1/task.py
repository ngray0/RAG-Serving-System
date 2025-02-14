import torch
import cupy as cp
import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
import time
# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass


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


def main():
    n = 1000
    d = 100
    X = cp.random.rand(n, d).astype(cp.float32)
    Y = cp.random.rand(n, d).astype(cp.float32)

    start_time = time.time()
    cosine_dist = distance_cosine(X, Y)
    end_time = time.time()
    print(f"Cosine distance computation time: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    l2_dist = distance_l2(X, Y)
    end_time = time.time()
    print(f"L2 distance computation time: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    dot_dist = distance_dot(X, Y)
    end_time = time.time()
    print(f"Dot product computation time: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    manhattan_dist = distance_manhattan(X, Y)
    end_time = time.time()
    print(f"Manhattan distance computation time: {end_time - start_time:.4f} seconds")

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
    main()
    # test_kmeans()
