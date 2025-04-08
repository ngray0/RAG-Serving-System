from pydantic import BaseModel
from typing import Optional

class QueryRequest(BaseModel):
    query: str
    k: int = 2

class QueryResponse(BaseModel):
    query: str
    result: str