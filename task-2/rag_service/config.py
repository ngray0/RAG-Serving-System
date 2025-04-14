from pydantic_settings import BaseSettings
import torch
import os

class Settings(BaseSettings):
    host: str = os.environ.get("HOST", "0.0.0.0")
    port: int = int(os.environ.get("PORT", "8000"))
    max_batch_size: int = int(os.environ.get("MAX_BATCH_SIZE", "32"))
    max_wait_time: float = float(os.environ.get("MAX_WAIT_TIME", "1.00"))  # seconds
    polling_interval: float = float(os.environ.get("POLLING_INTERVAL", "0.3")) # seconds
    document_text_file: str = os.environ.get("DOCUMENT_TEXT_FILE", "data/short_facts_contexts.json")
    document_embeddings_file: str = os.environ.get("DOCUMENT_EMBEDDINGS_FILE", "data/short_facts_embeddings.npy")
    document_queries_file: str = os.environ.get("DOCUMENT_QUERIES_FILE", "data/short_facts_queries.json")
    embed_model_name: str = os.environ.get("EMBED_MODEL_NAME", "intfloat/multilingual-e5-large-instruct")
    llm_model_name: str = os.environ.get("LLM_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
    device: str = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
