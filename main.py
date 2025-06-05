import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
import uvicorn
import json
import os

from rag_service.config import Settings
from rag_service.core.request_queue import RequestQueue
from rag_service.core.request_queue import RedisRequestQueue
from rag_service.core.batch_processor import BatchProcessor
from rag_service.core.retriever import CupyRetriever
from rag_service.api.endpoints import create_api

def main():
    # Load configuration
    settings = Settings()
    
    # Load documents and embeddings
    with open(settings.document_text_file, 'r') as f:
        documents = json.load(f)
    
    doc_embeddings = np.load(settings.document_embeddings_file)
    
    # Load models
    embed_tokenizer = AutoTokenizer.from_pretrained(settings.embed_model_name)
    embed_model = AutoModel.from_pretrained(settings.embed_model_name)
    embedding_model = {"tokenizer": embed_tokenizer, "model": embed_model}
    
    llm_tokenizer = AutoTokenizer.from_pretrained(settings.llm_model_name)
    llm_tokenizer.padding_side = 'left'
    llm_model = pipeline("text-generation", model=settings.llm_model_name, tokenizer=llm_tokenizer)
    
    # Create request queue - use Redis if REDIS_URL is set
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        print(f"Using Redis queue with URL: {redis_url}")
        request_queue = RedisRequestQueue(
            redis_url=redis_url,
            max_batch_size=settings.max_batch_size,
            max_wait_time=settings.max_wait_time
        )
    else:
        print("Using in-memory queue")
        request_queue = RequestQueue(
            max_batch_size=settings.max_batch_size,
            max_wait_time=settings.max_wait_time,
            polling_interval=settings.polling_interval
        )

    # retriever object
    retriever = CupyRetriever(
        doc_embeddings=doc_embeddings,
        documents=documents
    )
    
    # Start batch processor
    processor = BatchProcessor(
        request_queue=request_queue,
        embedding_model=embedding_model,
        llm_model=llm_model,
        retriever=retriever,
        device=settings.device
    )
    
    processor.start()
    
    # Create API
    app = create_api(request_queue)
    
    # Run server
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port
    )

if __name__ == "__main__":
    main()