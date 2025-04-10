from fastapi import FastAPI, HTTPException
from typing import Dict, Any, Union

from rag_service.api.models import QueryRequest, QueryResponse
from rag_service.core.request_queue import RequestQueue, RedisRequestQueue

def create_api(request_queue: Union[RequestQueue, RedisRequestQueue]):
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
    
    return app


    