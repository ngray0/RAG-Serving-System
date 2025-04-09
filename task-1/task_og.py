import torch
import cupy as cp
import triton
import triton.language as tl
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
import csv
import os
import math

# -------------------------------
# Distance Functions Implementation
# -------------------------------
def distance_cosine(X, Y):
    norm_X = cp.linalg.norm(X, axis=1) 
    norm_Y = cp.linalg.norm(Y, axis=1)
    cosine_similarity = cp.einsum('ij,ij->i', X, Y) / (norm_X * norm_Y)
    return 1 - cosine_similarity

def distance_l2(X, Y):
    return cp.linalg.norm(X - Y, axis=1)

def distance_dot(X, Y):
    return cp.einsum('ij,ij->i', X, Y)

def distance_manhattan(X, Y):
    return cp.sum(cp.abs(X - Y), axis=1)

# CPU versions (for benchmarking against GPU)
def distance_cosine_cpu(X, Y):
    norm_X = np.linalg.norm(X, axis=1)
    norm_Y = np.linalg.norm(Y, axis=1)
    cosine_similarity = np.einsum('ij,ij->i', X, Y) / (norm_X * norm_Y)
    return 1 - cosine_similarity

def distance_l2_cpu(X, Y):
    return np.linalg.norm(X - Y, axis=1)

def distance_dot_cpu(X, Y):
    return np.einsum('ij,ij->i', X, Y)

def distance_manhattan_cpu(X, Y):
    return np.sum(np.abs(X - Y), axis=1)

# Global variable for choosing the distance metric.
CURRENT_DISTANCE = "L2"

def compute_all_distances(A, X):
    """
    Compute all four distances and return as a dictionary.
    """
    if X.ndim == 1:
        X = X[None, :]
    X =  cp.broadcast_to(X, A.shape)
    return {
        "Cosine": distance_cosine(A, X),
        "L2": distance_l2(A, X),
        "Dot": distance_dot(A, X),
        "Manhattan": distance_manhattan(A, X)
    }

def our_knn(N, D, A, X, K):
    """
    Standard KNN using the distance selected in CURRENT_DISTANCE on the GPU (CuPy).
    """
    distances = compute_all_distances(A, X)[CURRENT_DISTANCE]
    k_indices = cp.argpartition(distances, K)[:K]
    k_indices = k_indices[cp.argsort(distances[k_indices])]
    return k_indices

def our_knn_stream(N, D, A, X, K):
    """
    KNN using CUDA Streams in CuPy for concurrent query processing.
    """
    B = X.shape[0] if X.ndim > 1 else 1  # Determine batch size.
    streams = [cp.cuda.Stream() for _ in range(B)]
    results = [None] * B

    for i in range(B):
        with streams[i]:
            query = X[i] if X.ndim > 1 else X
            distances = compute_all_distances(A, query)[CURRENT_DISTANCE]
            k_indices = cp.argpartition(distances, K)[:K]
            results[i] = k_indices[cp.argsort(distances[k_indices])]
    for s in streams:
        s.synchronize()
    
    return results if B > 1 else results[0]



def our_knn_hierachy(N, D, A, X, K):
    """
    KNN using hierarchical memory: Pinned memory enables fast CPU–GPU transfers.
    """
    if not isinstance(A, cp.ndarray):
        A = cp.asarray(A)
    if not isinstance(X, cp.ndarray):
        X = cp.asarray(X)
    
    distances = compute_all_distances(A, X)[CURRENT_DISTANCE]
    k_indices = cp.argpartition(distances, K)[:K]
    k_indices = k_indices[cp.argsort(distances[k_indices])]
    return k_indices

# -------------------------------
# Data Generation
# -------------------------------
def generate_data(N, D, K, output_dir="test_data", seed=42, chunk_size=10000):
    """
    Generate large datasets in chunks if necessary and save them as binary files (.npy).
    Includes a progress tracker to estimate generation time.
    """
    os.makedirs(output_dir, exist_ok=True)

    A_file = os.path.join(output_dir, f"A_{N}_{D}.npy")  # Use .npy for binary format
    X_file = os.path.join(output_dir, f"X_{D}.npy")
    json_file = os.path.join(output_dir, f"test_{N}_{D}_{K}.json")

    if os.path.exists(A_file) and os.path.exists(X_file) and os.path.exists(json_file):
        print(f"Data already exists. Skipping generation.")
        return

    np.random.seed(seed)

    # Check if dataset is large (for batch processing)
    large_dataset = (N * D > 1000000)  # Arbitrary threshold

    max_vram_gb = 40  # VRAM in GB
    vram_size = max_vram_gb * 1024 * 1024 * 1024  # Convert to bytes
    chunk_size = min(chunk_size, int(vram_size // (D * 4)))  # Calculate based on available VRAM

    if large_dataset:
        print(f"Generating LARGE dataset (N={N}, D={D}) in chunks...")

        start_time = time.time()
        batch_count = 0
        with open(A_file, "wb") as A_out:  # Use binary file for A
            for i in range(0, N, chunk_size):
                batch_size = min(chunk_size, N - i)
                batch_data = np.random.randn(batch_size, D)  # Generate random data
                # Save to binary format
                np.save(A_out, batch_data)  # Save as binary data
                batch_count += 1
                # Progress tracking
                percent_done = (i + batch_size) / N * 100
                elapsed = time.time() - start_time
                est_total_time = elapsed / (percent_done / 100) if percent_done > 0 else 0
                remaining_time = est_total_time - elapsed
                if batch_count % 10 == 0:  # Print progress every 10 batches
                    minutes, seconds = divmod(remaining_time, 60)
                    print(f"[{percent_done:.2f}%] - Estimated Time Left: {int(minutes)}m {int(seconds)}s", end="\r")
        print("\nDataset generation completed.")

        # Generate and save X
        X = np.random.randn(D)
        np.save(X_file, X)  # Save X as binary

    else:
        print(f"Generating SMALL dataset (N={N}, D={D}) in memory...")
        # Generate small dataset normally
        A = np.random.randn(N, D)
        X = np.random.randn(D)
        # Save dataset as binary
        np.save(A_file, A)
        np.save(X_file, X)

    # Save metadata as JSON
    json_data = {"n": N, "d": D, "k": K, "a_file": A_file, "x_file": X_file}
    with open(json_file, "w") as f:
        json.dump(json_data, f, indent=4)

    print(f"Generated JSON file: {json_file}")

# -------------------------------
# Benchmarking
# -------------------------------
def cpu_v_gpu(N=1000, D_range=(2, 2**15), num_trials=10, max_gpu_memory_fraction=0.9, csv_filename="dist_cpu_gpu_1000.csv"):
    dimensions = [2**i for i in range(int(np.log2(D_range[0])), int(np.log2(D_range[1])) + 1)]
    cpu_times_avg = []
    gpu_times_avg = []
    torch_times_avg = []
    threshold_dim = None
    np.random.seed(42)
    
    total_memory = torch.cuda.get_device_properties(0).total_memory
    max_chunk_memory = total_memory * max_gpu_memory_fraction
    
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Dimension", "CPU Time (s)", "GPU Time (s)", "PyTorch Time (s)"])
    
        for D in dimensions:
            cpu_times = []
            gpu_times = []
            torch_times = []
            chunk_size = N  # Default to full batch
            
            # Estimate max chunk size that fits in GPU memory
            estimated_chunk_size = int(max_chunk_memory / (D * 4))  # Float32 = 4 bytes per element
            if estimated_chunk_size < N:
                chunk_size = estimated_chunk_size
                print(f"D={D}: Splitting GPU computation into chunks of size {chunk_size}")
            
            for _ in range(num_trials):
                A = np.random.randn(N, D).astype(np.float32)
                X = np.random.randn(D).astype(np.float32)

                A_torch = torch.tensor(A).float().cuda()
                X_torch = torch.tensor(X).float().cuda()
                
                # CPU timing
                try:
                    start = time.perf_counter()
                    _ = distance_l2_cpu(A, X)
                    cpu_times.append(time.perf_counter() - start)
                except Exception:
                    cpu_times.append(None)
                
                # GPU timing (CuPy) with chunking
                try:
                    gpu_time_total = 0
                    for i in range(0, N, chunk_size):
                        A_gpu = cp.asarray(A[i:i+chunk_size])
                        X_gpu = cp.asarray(X)
                        start = time.perf_counter()
                        _ = distance_l2(A_gpu, X_gpu)
                        cp.cuda.Device(0).synchronize()
                        gpu_time_total += time.perf_counter() - start
                    gpu_times.append(gpu_time_total)
                except Exception:
                    gpu_times.append(None)
                
                # PyTorch timing with chunking
                try:
                    start = time.perf_counter()
                    _ = torch.norm(A_torch - X_torch, dim=1)
                    torch.cuda.synchronize()
                    torch_times.append(time.perf_counter() - start)
                except Exception:
                    torch_times.append(None)
                
            # Compute averages, ignoring None values
            cpu_avg = np.mean([t for t in cpu_times if t is not None]) if any(t is not None for t in cpu_times) else "N/A"
            gpu_avg = np.mean([t for t in gpu_times if t is not None]) if any(t is not None for t in gpu_times) else "N/A"
            torch_avg = np.mean([t for t in torch_times if t is not None]) if any(t is not None for t in torch_times) else "N/A"
            
            cpu_times_avg.append(cpu_avg)
            gpu_times_avg.append(gpu_avg)
            torch_times_avg.append(torch_avg)
            
            print(f"D={D}: CPU={cpu_avg:.6f}s, GPU={gpu_avg:.6f}s, PyTorch={torch_avg:.6f}s")
            
            # Find first dimension where GPU is faster than CPU
            if threshold_dim is None and isinstance(gpu_avg, float) and isinstance(cpu_avg, float) and gpu_avg < cpu_avg:
                threshold_dim = D
            
            # Write to CSV
            writer.writerow([D, cpu_avg, gpu_avg, torch_avg])
    
    print(f"Threshold dimension where GPU is faster: {threshold_dim}")
    return dimensions, cpu_times_avg, gpu_times_avg, torch_times_avg, threshold_dim

def benchmark_knn_with_distances(test_file, filename="result/knn_benchmark_results.csv", max_gpu_memory_fraction=0.9, chunk_size=100000, K=10):
    """
    Benchmark performance of different KNN implementations with each distance function.
    In total, 16 runs will be timed (4 KNN methods * 4 distance metrics).
    """
    results = []
    N, D, A, X, K = testdata_knn(test_file)  # testdata_knn must be defined elsewhere.
    print(f"\nBenchmarking for N={N}, D={D}, K={K}:")

    total_memory = torch.cuda.get_device_properties(0).total_memory
    max_memory = total_memory * max_gpu_memory_fraction
    available_memory = max_memory

    A_gpu = cp.asarray(A)
    X_gpu = cp.asarray(X)
    total_data_size = A_gpu.nbytes + X_gpu.nbytes

    if total_data_size <= available_memory:
        print("Data can fit in GPU memory. Processing all at once.")
        chunking_needed = False
    else:
        print(f"Data size exceeds available GPU memory ({total_data_size / 1e9:.2f} GB). Using chunking.")
        chunking_needed = True

    top_k_indices = {}

    # List of KNN implementations.
    knn_methods = [
        ("CuPy", our_knn),
        ("CuPy Streams", our_knn_stream),
        # ("Triton", our_knn_triton(N, D, A_gpu, X_gpu, K)),
        ("Hierarchical Memory", our_knn_hierachy)
    ]

    # List of distance metrics.
    distance_metrics = ["Cosine", "L2", "Dot", "Manhattan"]

    # For each KNN method, run all 4 distance functions.
    for knn_name, knn_func in knn_methods:
        for distance_name in distance_metrics:
            global CURRENT_DISTANCE
            CURRENT_DISTANCE = distance_name  # Set the distance metric.
            start = time.time()
            
            if chunking_needed:
                num_chunks = math.ceil(N / chunk_size)
                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min((chunk_idx + 1) * chunk_size, N)
                    A_chunk = A_gpu[start_idx:end_idx]
                    X_chunk = X_gpu[start_idx:end_idx]
                    k_indices = knn_func(end_idx - start_idx, D, A_chunk, X_chunk, K)
            else:
                k_indices = knn_func(N, D, A_gpu, X_gpu, K)
            
            cp.cuda.Stream.null.synchronize()
            elapsed_time = time.time() - start

            print(f"{knn_name} + {distance_name}: {elapsed_time:.6f}s")
            top_k_indices[(knn_name, distance_name)] = k_indices
            results.append([N, D, K, knn_name, distance_name, elapsed_time])
            
            # Compare current top K with previous distance functions for this KNN.
            for (prev_knn, prev_distance), prev_k_indices in top_k_indices.items():
                if prev_knn == knn_name and prev_distance != distance_name:
                    if np.array_equal(cp.asnumpy(k_indices), cp.asnumpy(prev_k_indices)):
                        print(f"Top K indices match for {distance_name} and {prev_distance} on {knn_name}")
                    else:
                        print(f"Top K indices DO NOT match for {distance_name} and {prev_distance} on {knn_name}")

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(results)

def main():
    test_configs = [
        {"N": 4000, "D": 100, "K": 10},
        {"N": 4000000, "D": 100, "K": 10},
        {"N": 4000, "D": 2**15, "K": 10},
        {"N": 4000000, "D": 2**15, "K": 10}
    ]

    for config in test_configs:
        generate_data(config["N"], config["D"], config["K"])

    test_files = [
        "test_data/test_4000_100_10.json",
        "test_data/test_4000000_100_10.json",
        "test_data/test_4000_32768_10.json",
        "test_data/test_4000000_32768_10.json"
    ]

    filename = "result/knn_benchmark_results.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["N", "D", "K", "KNN Implementation", "Distance Function", "Time (s)"])

    for test_file in test_files:
        benchmark_knn_with_distances(test_file, filename)
    
    # cpu_v_gpu()

if __name__ == "__main__":
    main()