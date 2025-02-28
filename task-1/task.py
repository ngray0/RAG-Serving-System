import torch
import cupy as cp
import triton
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

# Bench Mark
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
