import threading
import time
import torch
import numpy as np

from rag_service.queue.request_queue import RequestQueue

class BatchProcessor(threading.Thread):
    """Background thread that processes batches of RAG requests"""
    def __init__(self, request_queue, embedding_model, llm_model, doc_embeddings, documents):
        super().__init__(daemon=True)
        self.request_queue = request_queue
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.doc_embeddings = doc_embeddings
        self.documents = documents
        self.running = True
    
    def run(self):
        """Main processing loop"""
        while self.running:
            # Get a batch of requests
            batch = self.request_queue.get_batch()
            
            if not batch:
                # No requests, sleep briefly and try again
                time.sleep(0.1)
                continue
            
            # Process the batch
            self._process_batch(batch)
    
    def _process_batch(self, batch):
        """Process a batch of requests through the RAG pipeline"""
        try:
            # Extract queries and request IDs
            queries = [req["query"] for req in batch]
            request_ids = [req["id"] for req in batch]
            ks = [req["k"] for req in batch]
            
            # 1. Batch embedding generation
            with torch.no_grad():
                inputs = [f"passage: {query}" for query in queries]
                encoded = self.embedding_model["tokenizer"](inputs, padding=True, truncation=True, return_tensors="pt")
                outputs = self.embedding_model["model"](**encoded)
                query_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            # 2. Batch retrieval
            all_contexts = []
            for i, query_emb in enumerate(query_embeddings):
                # Calculate similarities
                sims = self.doc_embeddings @ query_emb.T
                # Get top-k documents REPLACE ME WITH BETTER TOP K ALGORITHM
                top_k_indices = np.argsort(sims.ravel())[::-1][:ks[i]]
                contexts = [self.documents[idx] for idx in top_k_indices]
                all_contexts.append("\n".join(contexts))
            
            # 3. Batch LLM generation
            prompts = [f"Question: {q}\nContext:\n{c}\nAnswer:" for q, c in zip(queries, all_contexts)]
            outputs = self.llm_model(prompts, max_new_tokens=25, do_sample=True)
            
            # 4. Store results
            for i, request_id in enumerate(request_ids):
                result = outputs[i]["generated_text"]
                self.request_queue.store_result(request_id, result)
                
        except Exception as e:
            print(f"Error processing batch: {e}")