from typing import List, Protocol, Any, Dict, Optional
from pydantic import BaseModel

class VectorDocument(BaseModel):
    content: str
    metadata: Dict[str, Any] = {}
    embedding: Optional[List[float]] = None
    id: Optional[str] = None

class VectorStore(Protocol):
    """
    Abstract interface for Vector Store operations.
    Allows switching between Postgres (pgvector), Qdrant, Pinecone, etc.
    """
    
    async def connect(self) -> None:
        ...
        
    async def disconnect(self) -> None:
        ...

    async def add_documents(self, documents: List[VectorDocument]) -> None:
        ...
    
    async def search(self, query_vector: List[float], limit: int = 5, filters: Dict = None) -> List[VectorDocument]:
        ...
