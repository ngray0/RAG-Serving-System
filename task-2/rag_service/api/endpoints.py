from fastapi import FastAPI, HTTPException
from typing import Dict, Any

from rag_service.api.models import QueryRequest, QueryResponse
from rag_service.queue.request_queue import RequestQueue

def create_api(request_queue: RequestQueue):
    """Create the FastAPI application with endpoints"""
    app = FastAPI()
    
    @app.post("/rag")
    async def rag_endpoint(payload: QueryRequest):
        # Add request to the queue
        request_id = request_queue.add_request(payload.query, payload.k)
        
        # Wait for the result
        result = request_queue.get_result(request_id)
        
        if result is None:
            raise HTTPException(status_code=408, detail="Request timed out")
        
        return {
            "query": payload.query,
            "result": result
        }
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy"}
    
    return app