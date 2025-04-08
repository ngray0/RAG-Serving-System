import threading
import time
import torch
import numpy as np
import logging 
from typing import List
from datasets import Dataset

from .request_queue import RequestQueue
from .retriever import SimpleRetriever 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BatchProcessor(threading.Thread):
    """Background thread that processes batches of RAG requests"""
    def __init__(self,
                 request_queue: RequestQueue,
                 embedding_model: dict,
                 llm_model, 
                 retriever: SimpleRetriever, 
                 device: str = 'cuda'): 
        super().__init__(daemon=True) 
        self.request_queue = request_queue
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.retriever = retriever 
        self.device = torch.device(device) 
        self.running = True

        # Move models to the cuda deviced
        try:
            self.embedding_model["model"].to(self.device)
            logging.info(f"Models assigned to device: {self.device}")
        except Exception as e:
            logging.error(f"Error moving models to device {self.device}: {e}")

        logging.info("BatchProcessor initialized.")

    def stop(self):
        """Signals the processing thread to stop."""
        logging.info("Stopping BatchProcessor...")
        self.running = False

    def run(self):
        """Main processing loop"""
        logging.info("BatchProcessor started.")
        while self.running:
            batch = []
            try:

                batch = self.request_queue.get_batch()

                if not batch:
                    # No requests, sleep adn try again
                    time.sleep(0.05) 
                    continue

                logging.info(f"Processing batch of size {len(batch)}...")
                start_time = time.time()

                self._process_batch(batch)
                end_time = time.time()
                logging.info(f"Batch processed in {end_time - start_time:.4f} seconds.")

            except Exception as e:
                logging.error(f"Critical error in BatchProcessor run loop: {e}", exc_info=True)

        logging.info("BatchProcessor stopped.")


    def _process_batch(self, batch):
        """Process a batch of requests through the RAG pipeline"""
        request_ids = [req["id"] for req in batch] 
        try:
            queries = [req["query"] for req in batch]
            ks = [req["k"] for req in batch]

            # 1. Batch embedding generation
            with torch.no_grad():
                inputs = [f"query: {query}" for query in queries] 
                encoded = self.embedding_model["tokenizer"](
                    inputs, padding=True, truncation=True, return_tensors="pt"
                )
                # Move tensors to the designated device
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                outputs = self.embedding_model["model"](**encoded)

                query_embeddings = outputs.last_hidden_state.mean(dim=1)
                query_embeddings_np = query_embeddings.cpu().numpy()

            # 2. Batch retrieval using the retriever instance
            all_retrieved_docs: List[List[str]] = self.retriever.batch_retrieve(
                query_embeddings_np, ks
            )

            # Format contexts for the LLM
            all_contexts = ["\n---\n".join(docs) for docs in all_retrieved_docs] # Use a separator

            # 3 Batch LLM generation
            prompts = [f"Context:\n{c}\n\nQuestion: {q}\n\nAnswer:" for q, c in zip(queries, all_contexts)]
            dataset = Dataset.from_dict({"text": prompts}) 
            outputs = self.llm_model(dataset["text"], max_new_tokens=10, do_sample=True, batch_size=len(prompts))

            # 4 stores results
            for i, request_id in enumerate(request_ids):
                try:
                    output = outputs[i]
                    full_text = str(output)
                        
                    self.request_queue.store_result(request_id, {"result": full_text})
                except Exception as e:
                    logging.error(f"Error processing result for request {request_id}: {e}")
                    self.request_queue.store_result(request_id, {"error": str(e)})

        except Exception as e:
            logging.error(f"Error processing batch (requests: {request_ids}): {e}")
                
            for request_id in request_ids:
                self.request_queue.store_result(request_id, {
                    "error": str(e),
                    "status": "failed"
                })