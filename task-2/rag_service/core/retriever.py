import numpy as np
from typing import List
import logging

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
        logging.info(f"Initialized SimpleRetriever with {len(self.documents)} documents.")

    def retrieve(self, query_emb: np.ndarray, k: int = 2) -> List[str]:
        """
        Retrieve top-k documents for a single query embedding using the
        specified dot-product logic.

        Args:
            query_emb: A NumPy array representing the query embedding
                             (shape: embedding_dim or potentially 1, embedding_dim).
            k: The number of top documents to retrieve.

        Returns:
            A list containing the text of the top-k documents. Returns empty list on error.
        """
        try:
            query_emb_flat = query_emb.flatten()
            sims = self.doc_embeddings @ query_emb_flat
            sims = sims.ravel()
            top_k_indices = np.argsort(sims)[::-1][:k]

            retrieved_docs = [self.documents[i] for i in top_k_indices]
            return retrieved_docs

        except Exception as e:
            logging.error(f"Unexpected error during retrieve method: {e}", exc_info=True)
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
        if query_embeddings.shape[0] != len(ks):
            logging.error(f"Batch retrieve size mismatch: {query_embeddings.shape[0]} embeddings vs {len(ks)} k values.")
            return [[] for _ in range(len(ks))]

        batch_results = []
        for i, query_emb in enumerate(query_embeddings):
            k = ks[i]
            top_docs = self.retrieve(query_emb, k)
            batch_results.append(top_docs)

        return batch_results