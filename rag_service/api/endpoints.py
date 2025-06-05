from fastapi import FastAPI
from typing import Union
from prometheus_client import Gauge, generate_latest
from fastapi.responses import Response
import time
import json

from rag_service.api.models import QueryRequest, QueryResponse
from rag_service.core.request_queue import RequestQueue, RedisRequestQueue

QUEUE_SIZE = Gauge('rag_queue_size', 'Number of requests in queue')
QUEUE_WAIT_TIME = Gauge('rag_queue_wait_time', 'Average wait time in queue (seconds)')

def create_api(request_queue: Union[RedisRequestQueue, RequestQueue]):
    """Create the FastAPI application with endpoints"""
    app = FastAPI()
    
    @app.post("/rag")
    async def rag_endpoint(payload: QueryRequest):
        # Add request to the queue
        request_id = request_queue.add_request(payload.query, payload.k)
        
        # Return a request ID immediately for polling
        return {
            "request_id": request_id,
            "status": "processing"
        }

    @app.get("/rag/result/{request_id}")
    async def get_result(request_id: str):
        # Check for the result
        result = request_queue.get_result(request_id, timeout=0.1)
        
        if result is None:
            return {"status": "processing"}
        
        return {
            "status": "complete",
            "result": result
        }
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}

    
    app.get("/metrics")
    async def metrics():
        # Update metrics before returning them
        if hasattr(request_queue, 'redis'):
            # For Redis queue
            queue_size = request_queue.redis.llen(request_queue.queue_key)
            
            # Calculate wait time (simplified approach)
            wait_time = 0
            if queue_size > 0:
                try:
                    oldest_item = request_queue.redis.lindex(request_queue.queue_key, 0)
                    if oldest_item:
                        item_data = json.loads(oldest_item)
                        if "timestamp" in item_data:
                            wait_time = time.time() - item_data["timestamp"]
                except Exception as e:
                    print(f"Error calculating wait time: {e}")
        else:
            # For in-memory queue
            queue_size = request_queue.queue.qsize()
            wait_time = 0  
        
        QUEUE_SIZE.set(queue_size)
        QUEUE_WAIT_TIME.set(wait_time)
        
        return Response(generate_latest(), media_type="text/plain")
    
    return app


    