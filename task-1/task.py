import torch
import cupy as cp
import triton
import numpy as np
import time
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
from cupyx.jit import rawkernel
import time
# ------------------------------------------------------------------------------------------------
# Your Task 1.1 code here
# ------------------------------------------------------------------------------------------------

# You can create any kernel here
# def distance_kernel(X, Y, D):
#     pass

# Define the CUDA kernel using Cupy's jit.rawkernel.
# The kernel code is specified in the function's docstring.

# ---------------------------
# 1. L2 Distance (Euclidean)
# ---------------------------
@rawkernel()
def l2_distance_kernel(A, B, output, N):
    """
    extern "C" __global__
    void l2_distance_kernel(const float *A, const float *B, float *output, int N) {
        extern __shared__ float sdata[];
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + tid;
        float sum = 0.0f;
        // Compute squared difference over grid-stride loop.
        for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
            float diff = A[i] - B[i];
            sum += diff * diff;
        }
        sdata[tid] = sum;
        __syncthreads();
        // Reduction in shared memory.
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            output[blockIdx.x] = sdata[0];
        }
    }
    """

def compute_l2_distance(A, B, threads_per_block=256):
    N = A.size
    grid_size = (N + threads_per_block - 1) // threads_per_block
    partial = cp.zeros(grid_size, dtype=cp.float32)
    shared_mem_bytes = threads_per_block * cp.dtype(cp.float32).itemsize
    l2_distance_kernel((grid_size,), (threads_per_block,),
                       (A, B, partial, np.int32(N)),
                       shared_mem=shared_mem_bytes)
    total = cp.sum(partial)
    # Euclidean distance is the square root of the summed squared differences.
    return cp.sqrt(total).get()


# ---------------------------
# 2. Dot Product
# ---------------------------
@rawkernel()
def dot_product_kernel(A, B, output, N):
    '''
    extern "C" __global__
    void dot_product_kernel(const float *A, const float *B, float *output, int N) {
        extern __shared__ float sdata[];
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + tid;
        float sum = 0.0f;
        for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
            sum += A[i] * B[i];
        }
        sdata[tid] = sum;
        __syncthreads();
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            output[blockIdx.x] = sdata[0];
        }
    }
    '''


def compute_dot_product(A, B, threads_per_block=256):
    N = A.size
    grid_size = (N + threads_per_block - 1) // threads_per_block
    partial = cp.zeros(grid_size, dtype=cp.float32)
    shared_mem_bytes = threads_per_block * cp.dtype(cp.float32).itemsize
    dot_product_kernel((grid_size,), (threads_per_block,),
                       (A, B, partial, np.int32(N)),
                       shared_mem=shared_mem_bytes)
    total = cp.sum(partial)
    return total.get()


# ---------------------------
# 3. Manhattan Distance (L1)
# ---------------------------
@rawkernel()
def manhattan_distance_kernel(A, B, output, N):
    """
    extern "C" __global__
    void manhattan_distance_kernel(const float *A, const float *B, float *output, int N) {
        extern __shared__ float sdata[];
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + tid;
        float sum = 0.0f;
        for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
            sum += fabsf(A[i] - B[i]);
        }
        sdata[tid] = sum;
        __syncthreads();
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            output[blockIdx.x] = sdata[0];
        }
    }
    """

def compute_manhattan_distance(A, B, threads_per_block=256):
    N = A.size
    grid_size = (N + threads_per_block - 1) // threads_per_block
    partial = cp.zeros(grid_size, dtype=cp.float32)
    shared_mem_bytes = threads_per_block * cp.dtype(cp.float32).itemsize
    manhattan_distance_kernel((grid_size,), (threads_per_block,),
                              (A, B, partial, np.int32(N)),
                              shared_mem=shared_mem_bytes)
    total = cp.sum(partial)
    return total.get()


# ---------------------------
# 4. Cosine Distance (for reference)
# ---------------------------
@rawkernel()
def cosine_distance_kernel(A, B, output, N):
    """
    extern "C" __global__
    void cosine_distance_kernel(const float *A, const float *B, float *output, int N) {
        // We compute three sums: dot, sumA^2, sumB^2.
        extern __shared__ float shared[];
        float *s_dot   = shared;
        float *s_normA = &shared[blockDim.x];
        float *s_normB = &shared[2 * blockDim.x];
        
        int tid = threadIdx.x;
        int idx = blockIdx.x * blockDim.x + tid;
        
        float dot = 0.0f, normA = 0.0f, normB = 0.0f;
        for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
            float a = A[i], b = B[i];
            dot   += a * b;
            normA += a * a;
            normB += b * b;
        }
        
        s_dot[tid]   = dot;
        s_normA[tid] = normA;
        s_normB[tid] = normB;
        __syncthreads();
        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                s_dot[tid]   += s_dot[tid + s];
                s_normA[tid] += s_normA[tid + s];
                s_normB[tid] += s_normB[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            output[blockIdx.x * 3 + 0] = s_dot[0];
            output[blockIdx.x * 3 + 1] = s_normA[0];
            output[blockIdx.x * 3 + 2] = s_normB[0];
        }
    }
    """

def compute_cosine_distance(A, B, threads_per_block=256):
    N = A.size
    grid_size = (N + threads_per_block - 1) // threads_per_block
    partial = cp.zeros(grid_size * 3, dtype=cp.float32)
    shared_mem_bytes = 3 * threads_per_block * cp.dtype(cp.float32).itemsize
    cosine_distance_kernel((grid_size,), (threads_per_block,),
                           (A, B, partial, np.int32(N)),
                           shared_mem=shared_mem_bytes)
    partial = partial.reshape((grid_size, 3))
    total_dot   = cp.sum(partial[:, 0])
    total_normA = cp.sum(partial[:, 1])
    total_normB = cp.sum(partial[:, 2])
    epsilon = 1e-6
    cosine_sim = total_dot / (cp.sqrt(total_normA) * cp.sqrt(total_normB) + epsilon)
    cosine_distance = 1.0 - cosine_sim
    return cosine_distance.get()


# ---------------------------
# Main Function to Test Metrics
# ---------------------------

def distance_cosine(X, Y):
    pass

def distance_l2(X, Y):
    pass

def distance_dot(X, Y):
    pass

def distance_manhattan(X, Y):
    pass

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
    l2 = compute_l2_distance(A_cp, B_cp)
    end_time = time.time()
    print(f"L2 distance computation time: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    dot = compute_dot_product(A_cp, B_cp)
    end_time = time.time()
    print(f"Dot Product distance computation time: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    manhattan = compute_manhattan_distance(A_cp, B_cp)
    end_time = time.time()
    print(f"Manhattan distance computation time: {end_time - start_time:.4f} seconds")

    start_time = time.time()
    cosine = compute_cosine_distance(A_cp, B_cp)
    end_time = time.time()
    print(f"Cosine distance computation time: {end_time - start_time:.4f} seconds")
    
    print("L2 (Euclidean) distance:", l2)
    print("Dot product:", dot)
    print("Manhattan (L1) distance:", manhattan)
    print("Cosine distance:", cosine)


if __name__ == "__main__":
    main()
