import queue
import threading
import time
import uuid
from typing import Dict, Any, Optional

class RequestQueue:
    """Thread-safe queue for RAG requests with result tracking"""
    def __init__(self, max_batch_size=8, max_wait_time=0.5, polling_interval=0.1):
        self.queue = queue.Queue()
        self.results = {}
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.polling_interval = polling_interval
        self.lock = threading.Lock()
        
    
    def add_request(self, query: str, k: int = 2) -> str:
        """Add a request to the queue and return its ID"""
        request_id = str(uuid.uuid4())
        self.queue.put({"id": request_id, "query": query, "k": k})
        return request_id
    
    def get_batch(self) -> list:
        """Get a batch of requests from the queue"""
        batch = []
        start_time = time.time()
        
        while len(batch) < self.max_batch_size:
            try:
                # Calculate remaining wait time
                elapsed = time.time() - start_time
                if elapsed >= self.max_wait_time and batch:
                    break  
                
                # Try to get a request with timeout
                request = self.queue.get(timeout=max(0.1, self.max_wait_time - elapsed))
                batch.append(request)
                self.queue.task_done()
            except queue.Empty:
                break  # Queue is empty or timeout
        
        return batch
    
    def store_result(self, request_id: str, result: Any):
        """Store the result for a completed request"""
        with self.lock:
            self.results[request_id] = result
    
    def get_result(self, request_id: str, timeout: float = 30) -> Optional[Any]:
        """Wait for and return the result for a specific request ID"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            with self.lock:
                if request_id in self.results:
                    result = self.results[request_id]
                    del self.results[request_id]  # Clean up
                    return result
            time.sleep(self.polling_interval)  # Check again after a short delay
        return None  # Timeout