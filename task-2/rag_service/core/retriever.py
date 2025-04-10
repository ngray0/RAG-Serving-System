
import numpy as np
import torch
from typing import List, Union
import logging
import triton
import triton.language as tl
import time
import json
# from test import testdata_kmeans, testdata_knn, testdata_ann # Assuming this exists if needed
import csv
import os
import math

# --- Device Setup ---
if not torch.cuda.is_available():
    print("CUDA not available, using CPU. Triton kernel will not be used.")
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")
    print(f"Using device: {device}")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Triton Kernel (Unchanged) ---
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

        dot_prod = tl.zeros((), dtype=tl.float64) # Compute in float64 for precision
        for d_start in range(0, D, BLOCK_SIZE_D):
            d_end = tl.minimum(d_start + BLOCK_SIZE_D, D)
            offs_d = d_start + tl.arange(0, BLOCK_SIZE_D)
            mask_d = offs_d < d_end

            # Load X values (queries) - ensure they are treated as float32 for calculation
            x_ptrs = X_ptr + pid_q * stride_xq + offs_d * stride_xd
            x_vals = tl.load(x_ptrs, mask=mask_d, other=0.0).to(tl.float32)

            # Load A values (documents) - ensure they are treated as float32 for calculation
            a_ptrs = A_ptr + pid_n * stride_an + offs_d * stride_ad
            a_vals = tl.load(a_ptrs, mask=mask_d, other=0.0).to(tl.float32)

            # Perform dot product calculation in float32, accumulate in float64
            dot_prod += tl.sum((x_vals * a_vals).to(tl.float64), axis=0)

        out_offset = pid_q * stride_outq + pid_n * stride_outn
        tl.store(Out_ptr + out_offset, dot_prod)


# --- SimpleRetriever (for reference) ---
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
        if doc_embeddings.dtype != np.float32:
             logging.warning(f"SimpleRetriever received doc_embeddings with dtype {doc_embeddings.dtype}. Converting to float32 for dot product.")
             doc_embeddings = doc_embeddings.astype(np.float32)
        self.doc_embeddings = doc_embeddings
        self.documents = documents
        logging.info(f"Initialized SimpleRetriever with {len(self.documents)} documents.")

    def retrieve(self, query_emb: np.ndarray, k: int = 2) -> List[str]:
        """
        Retrieve top-k documents for a single query embedding using dot-product.

        Args:
            query_emb: A 1D NumPy array representing the query embedding (shape: embedding_dim).
            k: The number of top documents to retrieve.

        Returns:
            A list containing the text of the top-k documents. Returns empty list on error.
        """
        if k <= 0:
             logging.warning(f"Requested k={k} is non-positive. Returning empty list.")
             return []
        if k > self.doc_embeddings.shape[0]:
            logging.warning(f"Requested k={k} is larger than the number of documents ({self.doc_embeddings.shape[0]}). Returning all documents.")
            k = self.doc_embeddings.shape[0]

        try:
            # Ensure query is float32 and 1D
            if query_emb.dtype != np.float32:
                query_emb = query_emb.astype(np.float32)
            if query_emb.ndim != 1:
                 # Attempt to flatten if shape is (1, dim)
                 if query_emb.ndim == 2 and query_emb.shape[0] == 1:
                     query_emb = query_emb.flatten()
                 else:
                     logging.error(f"SimpleRetriever.retrieve expects a 1D query embedding, got shape {query_emb.shape}")
                     return []

            if query_emb.shape[0] != self.doc_embeddings.shape[1]:
                logging.error(f"Query embedding dimension ({query_emb.shape[0]}) does not match document embedding dimension ({self.doc_embeddings.shape[1]})")
                return []

            sims = self.doc_embeddings @ query_emb # Dot product
            # No need to ravel if query_emb is 1D

            # Get indices of top k similarities (argsort sorts ascending, so take last k and reverse)
            # Using argpartition is slightly more efficient for large N
            # top_k_indices = np.argsort(sims)[-k:][::-1]
            top_k_indices = np.argpartition(sims, -k)[-k:]
            # Sort the top-k indices by similarity score
            top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])[::-1]]


            retrieved_docs = [self.documents[i] for i in top_k_indices]
            return retrieved_docs

        except Exception as e:
            logging.error(f"Unexpected error during SimpleRetriever.retrieve: {e}", exc_info=True)
            return []


    def batch_retrieve(self, query_embeddings: np.ndarray, ks: List[int]) -> List[List[str]]:
        """
        Retrieve top-k documents for a batch of query embeddings by calling
        the single 'retrieve' method iteratively.

        Args:
            query_embeddings: A NumPy array of query embeddings
                              (shape: batch_size, embedding_dim).
            ks: A list of integers, where each element is the 'k' value
                for the corresponding query in the batch.

        Returns:
            A list of lists, where each inner list contains the text of the
            top-k documents for the corresponding query.
        """
        if query_embeddings.ndim != 2:
             logging.error(f"SimpleRetriever.batch_retrieve expects a 2D query_embeddings array, got {query_embeddings.ndim} dimensions.")
             # Attempt to provide empty lists matching ks length
             return [[] for _ in ks]
        if query_embeddings.shape[0] != len(ks):
            logging.error(f"Batch retrieve size mismatch: {query_embeddings.shape[0]} embeddings vs {len(ks)} k values.")
            # Attempt to provide empty lists matching ks length
            return [[] for _ in ks]
        if query_embeddings.shape[1] != self.doc_embeddings.shape[1]:
            logging.error(f"Batch query embedding dimension ({query_embeddings.shape[1]}) does not match document embedding dimension ({self.doc_embeddings.shape[1]})")
            return [[] for _ in ks]


        batch_results = []
        for i, query_emb in enumerate(query_embeddings):
            k = ks[i]
            # retrieve expects a 1D array
            top_docs = self.retrieve(query_emb, k) # query_emb is already 1D here
            batch_results.append(top_docs)

        return batch_results


# --- Reworked TritonKnnRetriever ---
class TritonKnnRetriever:
    """
    A retriever using dense embeddings and Triton for dot-product similarity,
    mimicking the interface of SimpleRetriever but using batch processing.
    """
    def __init__(self, doc_embeddings: np.ndarray, documents: List[str]):
        """
        Initializes the retriever, moves embeddings to GPU (if available), and warms up the kernel.

        Args:
            doc_embeddings: A NumPy array of shape (num_docs, embedding_dim)
                            containing the precomputed document embeddings (on CPU).
            documents: A list of strings, where each string is the text
                       content of a document, corresponding to the embeddings.
        """
        self.device = device # Store the determined device
        self.documents = documents # Store documents (assumed CPU list)

        # --- Convert Embeddings to GPU Tensor ---
        logging.info(f"Preparing {doc_embeddings.shape} embeddings for device {self.device}...")
        try:
            # Ensure float32 for GPU operations and kernel compatibility
            if doc_embeddings.dtype != np.float32:
                logging.warning(f"Input doc_embeddings dtype is {doc_embeddings.dtype}, converting to float32.")
                doc_embeddings = doc_embeddings.astype(np.float32)

            # Transfer to target device (GPU or CPU)
            self.doc_embeddings_torch = torch.from_numpy(doc_embeddings).to(self.device)
            # Ensure contiguity after potential conversion/transfer
            if not self.doc_embeddings_torch.is_contiguous():
                 self.doc_embeddings_torch = self.doc_embeddings_torch.contiguous()
            logging.info(f"Embeddings prepared on {self.device}.")

        except Exception as e:
            logging.error(f"Failed to prepare embeddings for device {self.device}: {e}")
            raise

        # Store dimensions
        self.N_A = self.doc_embeddings_torch.shape[0] # Number of documents
        self.D = self.doc_embeddings_torch.shape[1]   # Embedding dimension

        # Store block size defaults
        self.DEFAULT_BLOCK_D = 512 # Default block size for Triton kernel dimension splitting

        logging.info(f"Initialized TritonKnnRetriever with {self.N_A} documents (Dim={self.D}) on device {self.device}.")

        # --- Warm-up Kernel (only if on CUDA) ---
        if self.device.type == 'cuda':
            logging.info("Warming up Triton kernel...")
            try:
                # Create small dummy tensors on the GPU
                warmup_Q = min(10, self.N_A) # Number of dummy queries
                warmup_K = min(5, self.N_A)  # Number of neighbors for warmup

                # Use float32, consistent with _prepare_tensors and internal logic
                warmup_X = torch.randn(warmup_Q, self.D, dtype=torch.float32, device=self.device)

                # Call the internal retrieval method which uses the kernel
                _ = self._retrieve_internal(warmup_X, warmup_K) # Discard result

                # --- Crucial: Synchronize GPU ---
                torch.cuda.synchronize()
                logging.info("Triton kernel warm-up complete.")

            except Exception as e:
                # Log warning but don't necessarily crash initialization
                logging.warning(f"Triton kernel warm-up failed: {e}. First call might be slow.", exc_info=True)
        else:
            logging.info("Skipping Triton kernel warm-up on CPU.")


    def _prepare_tensors(self, *tensors):
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
                try:
                    t = torch.tensor(t, dtype=torch.float32, device=self.device)
                except Exception as e:
                     logging.error(f"Failed to convert input of type {type(t)} to a Tensor: {e}")
                     raise TypeError(f"Input must be NumPy array or Torch Tensor, got {type(t)}.")

            # Ensure tensor is on the correct device
            if t.device != self.device:
                t = t.to(self.device)

            # Ensure float32 dtype (redundant if converted from float32 numpy, but safe)
            if t.dtype != torch.float32:
                t = t.to(dtype=torch.float32)

            # Ensure contiguous
            if not t.is_contiguous():
                t = t.contiguous()
            prepared.append(t)
        # Return tuple unpacking if multiple tensors, else single tensor
        return prepared[0] if len(prepared) == 1 else tuple(prepared)


    def distance_dot_tiled(self, X, A, N_TILE=50000, prep = True): # Tile size, adjust if needed
        """
    Computes pairwise dot product using Triton kernel, tiled over A
    to avoid exceeding GPU grid dimension limits.

    Args:
        X (torch.Tensor): Query vectors (Q, D) on GPU.
        A (torch.Tensor): Database vectors (N, D) on GPU.
        N_TILE (int): The maximum number of rows of A to process in one kernel launch.

    Returns:
        torch.Tensor: Output tensor of dot products (Q, N) on GPU.
        """
        if prep == True:
            X_prep, A_prep = self._prepare_tensors(X, A) # Ensure tensors are ready
        else:
            X_prep, A_prep = X, A
        Q, D = X_prep.shape
        N, D_A = A_prep.shape
        assert D == D_A, f"Dimension mismatch: X({D}) vs A({D_A})"

    # Output tensor remains the full size
        Out = torch.empty((Q, N), dtype=torch.float32, device=device)

        print(f"Tiling dot product calculation with N_TILE={N_TILE}")

        for n_start in range(0, N, N_TILE):
            n_end = min(n_start + N_TILE, N)
            N_chunk = n_end - n_start # Size of the current chunk of A
            A_chunk = A_prep[n_start:n_end, :] # Shape (N_chunk, D)
        # Slice the relevant part of Out for this tile
            Out_chunk = Out[:, n_start:n_end]   # Shape (Q, N_chunk)

            grid = (Q, N_chunk)
            if grid[0] == 0 or grid[1] == 0: continue 

        dot_kernel_pairwise[grid](
            X_prep, A_chunk, Out_chunk,       # Data pointers for the chunks
            Q, N_chunk, D,                    # Dimensions for the chunk
            X_prep.stride(0), X_prep.stride(1),
            A_chunk.stride(0), A_chunk.stride(1), # Strides of A_chunk
            Out_chunk.stride(0), Out_chunk.stride(1),# Strides of Out_chunk
            BLOCK_SIZE_D=self.DEFAULT_BLOCK_D          # Kernel block size constant
        )
        # Potentially add torch.cuda.synchronize() here if debugging tile-by-tile issues

        return -Out
    

    def distance_cosine_tiled(self, X, A, epsilon=1e-8, **kwargs):
        """
    Computes pairwise Cosine distances using the tiled dot product kernel
    and PyTorch operations for norms.
        """
        target_device = X.device
        X_prep, A_prep = self._prepare_tensors(X, A)
        Q, D = X_prep.shape
        N, D_A = A_prep.shape
        assert D == D_A, f"Dimension mismatch: X({D}) vs A({D_A})"
       # print(f"Calculating pairwise Cosine (Triton Dot + PyTorch Norm) for shapes: {X_prep.shape} and {A_prep.shape}") # Optional verbose

        dot_products = self.distance_dot_tiled(X_prep, A_prep, **kwargs) # (Q, N)
        X_norm = torch.linalg.norm(X_prep, axis=1, keepdims=True) # (Q, 1)
        A_norm = torch.linalg.norm(A_prep, axis=1, keepdims=True) # (N, 1)
        norm_product = X_norm * A_norm.T # (Q, N)
        cosine_similarity = dot_products / (norm_product + epsilon)
        cosine_similarity.clamp_(min=-1.0, max=1.0)
        return cosine_similarity


    def _retrieve_internal(self, X_batch: torch.Tensor, K: int) -> List[List[str]]:
        """
        Internal method to find K nearest neighbors for a batch of query vectors X_batch.
        Always expects a 2D Tensor for X_batch and returns a list of lists.
        Uses Triton kernel via _distance_dot.

        Args:
            X_batch (torch.Tensor): Prepared query vectors (Q, D) on self.device.
            K (int): Number of neighbors to find for EACH query.

        Returns:
            List[List[str]]: A list where each inner list contains the text of the
                             top-K documents for the corresponding query.
        """
        Q = X_batch.shape[0] # Batch size

        # --- Basic Input Validation ---
        if not isinstance(X_batch, torch.Tensor) or X_batch.ndim != 2:
             raise ValueError(f"_retrieve_internal expects a 2D Torch Tensor, got {type(X_batch)} with ndim {X_batch.ndim}")
        if X_batch.shape[1] != self.D:
            raise ValueError(f"Input query dimension is {X_batch.shape[1]}, expected {self.D}")
        if K <= 0:
            logging.warning(f"Requested K={K} is non-positive. Returning empty lists.")
            return [[] for _ in range(Q)]
        if K > self.N_A:
            logging.warning(f"Requested K={K} is larger than the number of documents ({self.N_A}). Clamping K to {self.N_A}.")
            K = self.N_A # Clamp K

        A_prep = self.doc_embeddings_torch # Already prepared during __init__

        # 1. Compute all pairwise dot products (Scores) using Triton/Fallback
        # Result is Float64 tensor on self.device, shape (Q, N_A)
        all_scores = self.distance_cosine_tiled(X_batch, A_prep)

        # 2. Find the top K indices based on scores
        # Use largest=True because higher dot product means more similar
        # topk returns values and indices
        # Ensure scores are float32 for topk if needed (though float64 should work)
        topk_scores, topk_indices = torch.topk(all_scores, k=K, dim=1, largest=True) # Shape (Q, K)

        # 3. Transfer indices to CPU for document lookup
        indices_np = topk_indices.cpu().numpy() # Shape (Q, K)

        # 4. Retrieve documents using CPU indices
        retrieved_docs_batched = []
        try:
            # Handle if self.documents is a list (most common)
            if isinstance(self.documents, list):
                retrieved_docs_batched = [
                    [self.documents[int(idx)] for idx in row_indices]
                    for row_indices in indices_np # Iterate through each query's results
                ]
            # Handle if self.documents is somehow a NumPy array (less common for text)
            elif isinstance(self.documents, np.ndarray):
                 # Fancy indexing with the 2D indices_np array
                 retrieved_docs_batched = self.documents[indices_np].tolist() # Convert rows to lists
            else:
                # Fallback attempt assuming list-like access
                logging.warning(f"self.documents has unhandled type: {type(self.documents)}. Attempting list-based retrieval.")
                retrieved_docs_batched = [
                    [self.documents[int(idx)] for idx in row_indices]
                    for row_indices in indices_np
                ]

        except IndexError as e:
            logging.error(f"Index out of bounds during document retrieval. Max index requested: {indices_np.max()}, Documents available: {len(self.documents)}. Error: {e}", exc_info=True)
            # Return empty lists for all queries in the batch on error
            return [[] for _ in range(Q)]
        except Exception as e:
            logging.error(f"Error during document lookup: {e}", exc_info=True)
             # Return empty lists for all queries in the batch on error
            return [[] for _ in range(Q)]

        return retrieved_docs_batched # List of lists, shape (Q, K)

    # --- SimpleRetriever Interface Methods ---

    def retrieve(self, query_emb: np.ndarray, k: int = 2) -> List[str]:
        """
        Retrieve top-k documents for a single query embedding.

        Args:
            query_emb: A 1D NumPy array representing the query embedding (shape: embedding_dim).
            k: The number of top documents to retrieve.

        Returns:
            A list containing the text of the top-k documents. Returns empty list on error.
        """
        start_time = time.time()
        # --- Input Validation ---
        if not isinstance(query_emb, np.ndarray):
             logging.error(f"TritonKnnRetriever.retrieve expects a NumPy array, got {type(query_emb)}")
             return []
        if query_emb.ndim == 2 and query_emb.shape[0] == 1:
             # Allow (1, dim) input and flatten it
             query_emb = query_emb.flatten()
             logging.debug("Flattened input query_emb with shape (1, dim) to 1D.")
        elif query_emb.ndim != 1:
             logging.error(f"TritonKnnRetriever.retrieve expects a 1D query embedding or (1, dim), got shape {query_emb.shape}")
             return []
        if query_emb.shape[0] != self.D:
            logging.error(f"Query embedding dimension ({query_emb.shape[0]}) does not match document embedding dimension ({self.D})")
            return []
        if k <= 0:
             logging.warning(f"Requested k={k} is non-positive. Returning empty list.")
             return []

        try:
            # 1. Prepare the single query as a batch of size 1
            # _prepare_tensors handles numpy -> tensor, device, dtype, contiguous
            query_tensor_batch = self._prepare_tensors(query_emb.reshape(1, -1)) # Shape (1, D)

            # 2. Call the internal batch retrieval method
            # It returns a list containing one list: [[doc1, doc2, ...]]
            results_batch = self._retrieve_internal(query_tensor_batch, k)

            # 3. Extract the single result list
            if results_batch and len(results_batch) == 1:
                final_result = results_batch[0]
            else:
                 # This case should ideally not happen if _retrieve_internal works correctly
                 logging.error(f"Internal retrieval for single query returned unexpected format: {results_batch}")
                 final_result = []

            end_time = time.time()
            logging.debug(f"TritonKnnRetriever.retrieve took {end_time - start_time:.6f}s")
            return final_result

        except Exception as e:
            logging.error(f"Unexpected error during TritonKnnRetriever.retrieve: {e}", exc_info=True)
            return []


    def batch_retrieve(self, query_embeddings: np.ndarray, ks: List[int]) -> List[List[str]]:
        """
        Retrieve top-k documents for a batch of query embeddings using efficient
        batch processing via Triton kernel. Handles variable k per query.

        Args:
            query_embeddings: A NumPy array of query embeddings
                              (shape: batch_size, embedding_dim).
            ks: A list of integers, where each element is the 'k' value
                for the corresponding query in the batch.

        Returns:
            A list of lists, where each inner list contains the text of the
            top-k documents for the corresponding query.
        """
        start_time = time.time()
        batch_size = query_embeddings.shape[0]

        # --- Input Validation ---
        if not isinstance(query_embeddings, np.ndarray):
             logging.error(f"TritonKnnRetriever.batch_retrieve expects a NumPy array, got {type(query_embeddings)}")
             return [[] for _ in ks] # Match expected output structure on error
        if query_embeddings.ndim != 2:
             logging.error(f"TritonKnnRetriever.batch_retrieve expects a 2D query_embeddings array, got {query_embeddings.ndim} dimensions.")
             return [[] for _ in ks]
        if batch_size == 0:
             logging.info("Received empty batch of queries.")
             return []
        if not isinstance(ks, list) or batch_size != len(ks):
            logging.error(f"Batch retrieve size mismatch: {batch_size} embeddings vs {len(ks)} k values.")
            return [[] for _ in range(batch_size)] # Match query batch size on error
        if query_embeddings.shape[1] != self.D:
            logging.error(f"Batch query embedding dimension ({query_embeddings.shape[1]}) does not match document embedding dimension ({self.D})")
            return [[] for _ in ks]
        if not all(isinstance(k, int) for k in ks):
             logging.error(f"List ks must contain only integers.")
             return [[] for _ in ks]
        if any(k <= 0 for k in ks):
             logging.warning(f"List ks contains non-positive values. Results for these queries will be empty lists.")
             # We proceed but results for k<=0 will be truncated to empty

        try:
            # 1. Determine the maximum k needed for the batch calculation
            max_k = 0
            valid_ks = [k for k in ks if k > 0] # Filter out invalid k values for max calculation
            if valid_ks:
                max_k = max(valid_ks)
            else:
                logging.warning("All k values in ks are non-positive. Returning list of empty lists.")
                return [[] for _ in ks]

            # Clamp max_k to the number of documents
            if max_k > self.N_A:
                logging.warning(f"Maximum requested k ({max_k}) is larger than the number of documents ({self.N_A}). Clamping to {self.N_A}.")
                max_k = self.N_A

            # 2. Prepare the batch of queries
            query_embeddings_batch = self._prepare_tensors(query_embeddings) # Shape (Q, D)

            # 3. Call the internal batch retrieval method ONCE with max_k
            # This gets the top max_k results for ALL queries efficiently
            results_max_k = self._retrieve_internal(query_embeddings_batch, max_k) # List[List[str]]

            # 4. Post-process: Slice results for each query to its specific k
            final_results = []
            if len(results_max_k) != batch_size:
                 # This indicates a problem in _retrieve_internal
                 logging.error(f"Internal retrieval returned {len(results_max_k)} results for a batch size of {batch_size}. Mismatch detected.")
                 # Fallback: return empty lists matching ks length
                 return [[] for _ in ks]

            for i in range(batch_size):
                k_i = ks[i]
                if k_i <= 0:
                    final_results.append([]) # Append empty list for k<=0
                else:
                    # Slice the results for query i down to k_i
                    # Ensure slicing doesn't go beyond max_k (or N_A if clamped)
                    actual_k_i = min(k_i, max_k) # Use the possibly clamped max_k as upper bound
                    final_results.append(results_max_k[i][:actual_k_i])

            end_time = time.time()
            logging.debug(f"TritonKnnRetriever.batch_retrieve took {end_time - start_time:.6f}s")
            return final_results

        except Exception as e:
            logging.error(f"Unexpected error during TritonKnnRetriever.batch_retrieve: {e}", exc_info=True)
            return [[] for _ in ks] # Return empty lists matching ks length on error

'''
# --- Example Usage (Optional - uncomment to run) ---
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO) # Ensure logging is configured
    logging.getLogger().setLevel(logging.INFO) # Set root logger level if needed

    # Example Data
    num_docs = 5000
    embedding_dim = 768 # Example dimension
    # Ensure documents list matches num_docs
    example_docs = [f"This is document content {i}" for i in range(num_docs)]
    # Use float32 as standard for embeddings
    example_embeddings_np = np.random.rand(num_docs, embedding_dim).astype(np.float32)

    print("\n--- Initializing SimpleRetriever ---")
    simple_retriever = SimpleRetriever(doc_embeddings=example_embeddings_np.copy(), documents=example_docs) # Use copy for safety

    print("\n--- Initializing TritonKnnRetriever ---")
    print("(Includes moving embeddings to device and potential warm-up)...")
    start_init = time.time()
    triton_retriever = TritonKnnRetriever(doc_embeddings=example_embeddings_np, documents=example_docs)
    end_init = time.time()
    print(f"TritonKnnRetriever Initialization took {end_init - start_init:.4f} seconds.")

    # Example Queries
    num_queries = 10
    example_queries_np = np.random.rand(num_queries, embedding_dim).astype(np.float32)
    example_single_query_np = np.random.rand(embedding_dim).astype(np.float32)
    example_ks = [np.random.randint(1, 11) for _ in range(num_queries)] # Variable k, e.g., 1 to 10
    single_k = 5

    # --- Test Single Retrieve ---
    print(f"\n--- Testing retrieve (single query, k={single_k}) ---")

    start_simple_single = time.time()
    results_simple_single = simple_retriever.retrieve(example_single_query_np, k=single_k)
    end_simple_single = time.time()
    print(f"SimpleRetriever Single Time: {end_simple_single - start_simple_single:.6f}s")
    print(f"SimpleRetriever Single Result Type: {type(results_simple_single)}, Length: {len(results_simple_single)}")
    # print("SimpleRetriever Sample:", results_simple_single[0] if results_simple_single else "N/A")

    start_triton_single = time.time()
    results_triton_single = triton_retriever.retrieve(example_single_query_np, k=single_k)
    end_triton_single = time.time()
    print(f"TritonRetriever Single Time: {end_triton_single - start_triton_single:.6f}s")
    print(f"TritonRetriever Single Result Type: {type(results_triton_single)}, Length: {len(results_triton_single)}")
    # print("TritonRetriever Sample:", results_triton_single[0] if results_triton_single else "N/A")

    # Optional: Check if results match (allow for floating point differences in order)
    # print("Single Results Match (set comparison):", set(results_simple_single) == set(results_triton_single))

    # --- Test Batch Retrieve ---
    print(f"\n--- Testing batch_retrieve (batch_size={num_queries}, variable k={example_ks[:3]}...) ---")

    start_simple_batch = time.time()
    results_simple_batch = simple_retriever.batch_retrieve(example_queries_np, ks=example_ks)
    end_simple_batch = time.time()
    print(f"SimpleRetriever Batch Time: {end_simple_batch - start_simple_batch:.6f}s")
    print(f"SimpleRetriever Batch Result Type: {type(results_simple_batch)}, Length: {len(results_simple_batch)}")
    print(f"SimpleRetriever Batch Inner Lengths (first 3): {[len(r) for r in results_simple_batch[:3]]}")
    # print("SimpleRetriever Sample:", results_simple_batch[0][0] if results_simple_batch and results_simple_batch[0] else "N/A")


    start_triton_batch = time.time()
    results_triton_batch = triton_retriever.batch_retrieve(example_queries_np, ks=example_ks)
    end_triton_batch = time.time()
    print(f"TritonRetriever Batch Time: {end_triton_batch - start_triton_batch:.6f}s")
    print(f"TritonRetriever Batch Result Type: {type(results_triton_batch)}, Length: {len(results_triton_batch)}")
    print(f"TritonRetriever Batch Inner Lengths (first 3): {[len(r) for r in results_triton_batch[:3]]}")
    # print("TritonRetriever Sample:", results_triton_batch[0][0] if results_triton_batch and results_triton_batch[0] else "N/A")

    # Optional: Check if batch results match
    # match = True
    # if len(results_simple_batch) == len(results_triton_batch):
    #     for i in range(len(results_simple_batch)):
    #         if set(results_simple_batch[i]) != set(results_triton_batch[i]):
    #             match = False
    #             print(f"Mismatch at index {i}")
    #             # print(" Simple:", results_simple_batch[i])
    #             # print(" Triton:", results_triton_batch[i])
    #             break
    # else:
    #     match = False
    # print("Batch Results Match (set comparison per query):", match)

    # Verify k values were respected
    print("Batch Results Lengths Match Requested Ks:", all(len(results_triton_batch[i]) == example_ks[i] for i in range(num_queries) if example_ks[i]>0))
'''