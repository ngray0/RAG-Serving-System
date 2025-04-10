import numpy as np
import torch
from typing import List
import logging
import triton
import triton.language as tl
import numpy as np
import time
import cupy as cp
import json
from test import testdata_kmeans, testdata_knn, testdata_ann
import csv
import os
import math
from typing import List, Union

if not torch.cuda.is_available():
    print("CUDA not available, exiting.")
    exit()
device = torch.device("cuda:0")
print(f"Using device: {device}")

@triton.jit
def dot_kernel_pairwise(
    X_ptr, A_ptr, Out_ptr,
    Q, N, D,
    stride_xq, stride_xd,
    stride_an, stride_ad,
    stride_outq, stride_outn,
    BLOCK_SIZE_D: tl.constexpr,
    ):
        """Calculates pairwise dot product: dot(X[q], A[n])"""
        pid_q = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)

        dot_prod = tl.zeros((), dtype=tl.float64)
        for d_start in range(0, D, BLOCK_SIZE_D):
            d_end = tl.minimum(d_start + BLOCK_SIZE_D, D)
            offs_d = d_start + tl.arange(0, BLOCK_SIZE_D)
            mask_d = offs_d < d_end

            x_ptrs = X_ptr + pid_q * stride_xq + offs_d * stride_xd
            x_vals = tl.load(x_ptrs, mask=mask_d, other=0.0)

            a_ptrs = A_ptr + pid_n * stride_an + offs_d * stride_ad
            a_vals = tl.load(a_ptrs, mask=mask_d, other=0.0)

            dot_prod += tl.sum(x_vals * a_vals, axis=0)

        out_offset = pid_q * stride_outq + pid_n * stride_outn
        tl.store(Out_ptr + out_offset, dot_prod)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimpleRetriever:
    """
    A basic retriever using dense embeddings and dot-product similarity
    """
    def __init__(self, doc_embeddings: np.ndarray, documents: List[str]):
        """
        Initializes the retriever with document embeddings and text.

        Args:
            doc_embeddings: A NumPy array of shape (num_docs, embedding_dim)
                            containing the precomputed document embeddings.
            documents: A list of strings, where each string is the text
                       content of a document, corresponding to the embeddings.
        """

        self.doc_embeddings = doc_embeddings
        self.documents = documents
        self.N_A = self.doc_embeddings.shape[0]
        self.D = self.doc_embeddings.shape[1]
        logging.info(f"Initialized SimpleRetriever with {len(self.documents)} documents.")
    def distance_dot(self, X, Y):
       """
    Calculates pairwise dot products between rows of X and rows of Y.

    Args:
        X: cupy array of shape (Q, D) - Query points.
        Y: cupy array of shape (N, D) - Reference data points.

    Returns:
        cupy array of shape (Q, N) where element (i, j) is dot(X[i], Y[j]).
        """
       X_cp = cp.asarray(X)
       Y_cp = cp.asarray(Y)
       print(f"Calculating pairwise dot products via matmul for shapes: {X_cp.shape} and {Y_cp.T.shape}")
       return X_cp @ Y_cp.T

  
    def retrieve(self,  X, K):
        """
    KNN using CUDA Streams in CuPy for concurrent query processing.
    MODIFIED: Uses distance_dot for similarity (finds K most similar neighbors).
        """
        if not isinstance(X, cp.ndarray):
            X = cp.asarray(X)


        if X.ndim > 0 and X.shape[-1] != self.D:
            raise ValueError(f"Query X feature dimension {X.shape[-1]} does not match D={self.D}")
  

        B = X.shape[0] if X.ndim > 1 else 1
        if B == 0: return []

        streams = [cp.cuda.Stream() for _ in range(B)]
        results = [None] * B # To store indices for each query

    # Determine the actual number of neighbors to find (cannot exceed dataset size)
        actual_k = min(K, self.N_A)
        if actual_k <= 0: # Handles K<=0 or N=0
            empty_result = cp.array([], dtype=cp.int64)
            return [empty_result] * B if B > 1 else empty_result

        for i in range(B):
        # Assign work to the stream
            with streams[i]:
                query = X[i] if X.ndim > 1 else X
                query_2d = query.reshape(1, self.D)

                raw_dot_products = self.distance_dot(query_2d, self.doc_embeddings)

                scores_1d = raw_dot_products[0]

                k_indices_unsorted = cp.argpartition(scores_1d, self.N_A - actual_k)[-actual_k:]

            # Get the actual scores for these K indices if needed (e.g., for sorting)
                k_scores_unsorted = scores_1d[k_indices_unsorted]

            # Sort these K scores in descending order to get the final ranking
            # argsort on negative scores gives descending order indices relative to the K subset
                sorted_order_in_k = cp.argsort(-k_scores_unsorted)

            # Apply the sort order to the indices
                top_k_indices = k_indices_unsorted[sorted_order_in_k]
                top_k_docs = self.documents[top_k_indices]

            # # Method 2: Sorting (simpler, maybe faster for small N)
            # sorted_indices_descending = cp.argsort(-scores_1d) # Sort all N scores descending
            # top_k_indices = sorted_indices_descending[:actual_k]

            # Store the final indices for this query
                results[i] = top_k_docs
            # --------------------------------------------------------------------

    # Wait for all streams to finish their work
        for s in streams:
            s.synchronize()

    # Return the list of results (or single result if B=1)
        return results if B > 1 else results[0]



class TritonKnnRetriever:
    """
    A basic retriever using dense embeddings and Triton for dot-product similarity.
    Initializes Triton kernel on instantiation.
    """
    def __init__(self, doc_embeddings: np.ndarray, documents: List[str]):
        """
        Initializes the retriever, moves embeddings to GPU, and warms up the kernel.

        Args:
            doc_embeddings: A NumPy array of shape (num_docs, embedding_dim)
                            containing the precomputed document embeddings (on CPU).
            documents: A list of strings, where each string is the text
                       content of a document, corresponding to the embeddings.
        """
        self.device = device # Store the determined device
        self.documents = documents # Store documents (assumed CPU list)
        

        if self.device.type == 'cpu':
             logging.warning("Running on CPU, Triton kernel will not be used effectively.")
             # Store embeddings as torch CPU tensor if needed for consistency
             self.doc_embeddings_torch = torch.tensor(doc_embeddings, dtype=torch.float32, device=self.device)
        else:
             # --- Convert Embeddings to GPU Tensor ---
             logging.info(f"Moving {doc_embeddings.shape} embeddings to {self.device}...")
             try:
                 # Ensure float32 for potential GPU operations
                 if doc_embeddings.dtype != np.float32:
                     logging.warning(f"Input doc_embeddings dtype is {doc_embeddings.dtype}, converting to float32.")
                     doc_embeddings = doc_embeddings.astype(np.float32)

                 # Transfer to GPU
                 self.doc_embeddings_torch = torch.from_numpy(doc_embeddings).to(self.device)
                 logging.info("Embeddings moved to GPU.")
             except Exception as e:
                 logging.error(f"Failed to move embeddings to GPU: {e}")
                 raise

        # Store dimensions
        self.N_A = self.doc_embeddings_torch.shape[0]
        self.D = self.doc_embeddings_torch.shape[1]

        # Store block size defaults (can be overridden if needed)
        self.DEFAULT_BLOCK_D = 512 # From original code

        logging.info(f"Initialized TritonKnnRetriever with {self.N_A} documents (Dim={self.D}).")

        # --- Warm-up Kernel ---
        if self.device.type == 'cuda':
            logging.info("Warming up Triton kernel...")
            try:
                # Create small dummy tensors on the GPU
                warmup_Q = 10
                warmup_N = min(16, self.N_A) # Use a small N, but not more than available
                warmup_D = self.D

                # Use float32, consistent with _prepare_tensors
                warmup_X = torch.randn(warmup_Q, warmup_D, dtype=torch.float32, device=self.device)
                # Use a slice of actual embeddings or create dummy A
                # Using dummy A avoids issues if N_A is very small
                warmup_A = torch.randn(warmup_N, warmup_D, dtype=torch.float32, device=self.device)
                # Alternatively use: warmup_A = self.doc_embeddings_torch[:warmup_N]

                # Call the method that uses the kernel
                # We need warmup_A_prep, warmup_X_prep if distance_dot expects prepared tensors
                # Let's assume distance_dot calls _prepare_tensors internally
                _ = self.distance_dot(warmup_X, warmup_A) # Discard result

                # --- Crucial: Synchronize GPU ---
                torch.cuda.synchronize()
                logging.info("Triton kernel warm-up complete.")

            except Exception as e:
                # Log warning but don't necessarily crash initialization
                logging.warning(f"Triton kernel warm-up failed: {e}. First call might be slow.")
        else:
             logging.info("Skipping Triton kernel warm-up on CPU.")


    def _prepare_tensors(self, *tensors): # Removed target_device, uses self.device
        """Ensure tensors are float32, contiguous, and on self.device."""
        prepared = []
        for t in tensors:
            # If input is numpy array, convert it directly
            if isinstance(t, np.ndarray):
                 # Ensure float32 during conversion
                 t_np_f32 = t.astype(np.float32) if t.dtype != np.float32 else t
                 t = torch.from_numpy(t_np_f32).to(self.device)
            elif not isinstance(t, torch.Tensor):
                # If not tensor or numpy, try converting directly (might fail)
                t = torch.tensor(t, dtype=torch.float32, device=self.device)

            # Ensure tensor is on the correct device
            if t.device != self.device:
                t = t.to(self.device)

            # Ensure float32 dtype
            if t.dtype != torch.float32:
                t = t.to(dtype=torch.float32)

            # Ensure contiguous
            if not t.is_contiguous():
                 t = t.contiguous()
            prepared.append(t)
        return prepared

    def distance_dot(self, X, A):
        """Computes pairwise dot product using Triton kernel."""
        # Ensure inputs are prepared GPU tensors
        X_prep, A_prep = self._prepare_tensors(X, A)

        Q, D_X = X_prep.shape
        N, D_A = A_prep.shape
        assert D_X == self.D, f"Query dimension mismatch: expected {self.D}, got {D_X}"
        assert D_A == self.D, f"Database dimension mismatch: expected {self.D}, got {D_A}"
        assert N == A_prep.shape[0] # N should match A's size

        # Output tensor, match kernel's compute type (float64)
        Out = torch.empty((Q, N), dtype=torch.float64, device=self.device)
        grid = (Q, N)

        # Check if on CPU - Triton kernel won't run
        if self.device.type == 'cpu':
            logging.warning("Cannot run Triton kernel on CPU. Returning dummy output.")
            # Fallback or error - returning dummy data here
            return torch.zeros((Q, N), dtype=torch.float64, device=self.device)

        dot_kernel_pairwise[grid](
            X_prep, A_prep, Out,
            Q, N, self.D, # Use self.D
            X_prep.stride(0), X_prep.stride(1),
            A_prep.stride(0), A_prep.stride(1),
            Out.stride(0), Out.stride(1),
            BLOCK_SIZE_D=self.DEFAULT_BLOCK_D
        )
        return -Out

    def retrieve(self, X: Union[np.ndarray, torch.Tensor], K: int) -> Union[np.ndarray, List]:
        """
         Finds the K nearest neighbors for query vector(s) X.
         Accepts X as 1D (D,) or 2D (Q, D) NumPy array or Torch Tensor.

         Args:
             X: Query vector (D,) or vectors (Q, D) as NumPy array or Torch Tensor.
             K (int): Number of neighbors to find.

         Returns:
             Retrieved documents. Format depends on input shape and self.documents type.
        """
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X) 
        elif isinstance(X, torch.Tensor):
            X_tensor = X
        else:
            try:
                 # Attempt conversion for list-like inputs etc.
                 X_tensor = torch.tensor(X, dtype=torch.float32)
            except Exception as e:
                 raise TypeError(f"Input X must be NumPy array or Torch Tensor, got {type(X)}. Conversion failed: {e}")

        input_was_1d = False
        if X_tensor.ndim == 1:
            input_was_1d = True
            if X_tensor.shape[0] != self.D: # Check dimension before unsqueeze
                 raise ValueError(f"Input query dimension is {X_tensor.shape[0]}, expected {self.D}")
            X_tensor = X_tensor.unsqueeze(0) # Add batch dimension -> (1, D)
        elif X_tensor.ndim == 2:
             if X_tensor.shape[1] != self.D: # Check dimension
                 raise ValueError(f"Input query dimension is {X_tensor.shape[1]}, expected {self.D}")
        else:
            raise ValueError(f"Query tensor X must be 1D or 2D, got {X_tensor.ndim} dimensions.")
        # Now X_tensor is guaranteed to be 2D: (Q, D) and on its original device

        # Prepare tensors (moves X to self.device, ensures float32/contiguous)
        # A is already prepared and stored as self.doc_embeddings_torch
        A_prep = self.doc_embeddings_torch
        _, X_prep = self._prepare_tensors(A_prep, X_tensor) # Only need to prepare X

        Q = X_prep.shape[0]

        # Assertions (using instance attributes N_A, D)
        assert A_prep.shape[0] == self.N_A, f"Internal N_A mismatch: {self.N_A} vs {A_prep.shape[0]}"
        assert A_prep.shape[1] == self.D, f"Internal D mismatch: {self.D} vs {A_prep.shape[1]}"
        assert X_prep.shape[1] == self.D, f"Query D mismatch after prepare: {self.D} vs {X_prep.shape[1]}"
        assert K > 0, "K must be positive"
        assert K <= self.N_A, f"K ({K}) cannot be larger than the number of documents ({self.N_A})"

        start_time = time.time()

        # Pass the prepared tensors directly
        all_distances = self.distance_dot(X_prep, A_prep) # Shape (Q, N_A)

        # Find the top K smallest distances (or largest similarities)
        # Assuming lower distance value (higher dot product if using raw dot) is better
        # If using raw dot product, change largest=True
        topk_distances, topk_indices = torch.topk(all_distances, k=K, dim=1, largest=False)

        # 3. Transfer indices to CPU for document lookup
        indices_np = topk_indices.cpu().numpy() # Shape (Q, K)

        gpu_end_time = time.time()

        retrieved_docs_batched = None
        try:
            if isinstance(self.documents, np.ndarray):
                retrieved_docs_batched = self.documents[indices_np]
            elif isinstance(self.documents, list):
                retrieved_docs_batched = [
                    [self.documents[int(idx)] for idx in row]
                    for row in indices_np
                ]
            else:
                 print(f"Warning: self.documents unhandled type: {type(self.documents)}. Attempting list retrieval.")
                 retrieved_docs_batched = [
                     [self.documents[int(idx)] for idx in row]
                     for row in indices_np
                 ]
        except IndexError:
             print(f"Error: Index out of bounds. Max index: {indices_np.max()}, Docs size: {len(self.documents)}")
             raise
        except Exception as e:
             print(f"Error during document retrieval: {e}")
             raise

        cpu_end_time = time.time()
        #print(f"k-NN GPU+Transfer Time: {gpu_end_time - start_time:.6f}s | CPU Lookup Time: {cpu_end_time - gpu_end_time:.6f}s")

        if input_was_1d:
            if isinstance(retrieved_docs_batched, np.ndarray):
                return retrieved_docs_batched.squeeze(0)
            elif isinstance(retrieved_docs_batched, list) and len(retrieved_docs_batched) == 1:
                return retrieved_docs_batched[0]
            else:
                return retrieved_docs_batched
        else:
            return retrieved_docs_batched

'''
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Example Data
    num_docs = 1000
    embedding_dim = 768
    example_docs = [f"This is document {i}" for i in range(num_docs)]
    example_embeddings_np = np.random.rand(num_docs, embedding_dim).astype(np.float32)

    print("Initializing retriever (includes moving embeddings to GPU and warm-up)...")
    start_init = time.time()
    retriever = TritonKnnRetriever(doc_embeddings=example_embeddings_np, documents=example_docs)
    end_init = time.time()
    print(f"Initialization took {end_init - start_init:.4f} seconds.")

    # Example Query
    num_queries = 5
    k = 3
    example_query_np = np.random.rand(num_queries, embedding_dim).astype(np.float32)
    example_single_query_np = np.random.rand(embedding_dim).astype(np.float32)


    # Test retrieve with batch
    print("\nTesting retrieve with batch...")
    results_batch = retriever.retrieve(X=example_query_np, K=k)
    print(f"Retrieved batch type: {type(results_batch)}")
    # Add shape print if numpy
    if isinstance(results_batch, np.ndarray): print(f"Retrieved batch shape: {results_batch.shape}")
    elif isinstance(results_batch, list): print(f"Retrieved batch lengths: {len(results_batch)} x {len(results_batch[0]) if results_batch else 0}")
    # print("Sample batch results:", results_batch[0])


    # Test retrieve with single query (as numpy)
    print("\nTesting retrieve with single NumPy query...")
    results_single_np = retriever.retrieve(X=example_single_query_np, K=k)
    print(f"Retrieved single type: {type(results_single_np)}")
    if isinstance(results_single_np, np.ndarray): print(f"Retrieved single shape: {results_single_np.shape}")
    elif isinstance(results_single_np, list): print(f"Retrieved single length: {len(results_single_np)}")
    # print("Sample single results:", results_single_np)

    # Test retrieve with single query (as torch tensor)
    print("\nTesting retrieve with single Torch query...")
    example_single_query_torch = torch.from_numpy(example_single_query_np).to(device) # Put on correct device
    results_single_torch = retriever.retrieve(X=example_single_query_torch, K=k)
    print(f"Retrieved single type: {type(results_single_torch)}")
    if isinstance(results_single_torch, np.ndarray): print(f"Retrieved single shape: {results_single_torch.shape}")
    elif isinstance(results_single_torch, list): print(f"Retrieved single length: {len(results_single_torch)}")
    # print("Sample single results:", results_single_torch)



'''
    