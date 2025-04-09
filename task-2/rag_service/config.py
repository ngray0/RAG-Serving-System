from pydantic_settings import BaseSettings
import torch

class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: int = 8
    max_wait_time: float = 1 # seconds
    polling_interval = 0.05
    document_text_file: str = "data/short_facts_contexts.json"
    document_embeddings_file: str = "data/short_facts_embeddings.npy"
    document_queries_file: str = "data/short_facts_queries.json"
    embed_model_name: str = "intfloat/multilingual-e5-large-instruct" 
    llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct" 
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
