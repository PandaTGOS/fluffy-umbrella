from typing import List
from langchain_ollama import OllamaEmbeddings
from backend.core.config import settings
from backend.infrastructure.vector.postgres import PostgresVectorStore
from backend.infrastructure.vector.interface import VectorStore
from backend.core.logger import get_logger

logger = get_logger(__name__)

class RAGService:
    def __init__(self, vector_store: VectorStore, embeddings: OllamaEmbeddings):
        self.vector_store = vector_store
        self.embeddings = embeddings

    async def ingest_text(self, text: str, metadata: dict):
        """
        Embed and ingest text into vector store.
        """
        vector = await self.embeddings.aembed_query(text)
        
        from backend.infrastructure.vector.interface import VectorDocument
        doc = VectorDocument(
            content=text,
            metadata=metadata,
            embedding=vector
        )
        
        await self.vector_store.add_documents([doc])

    async def retrieve(self, query: str, k: int = 5) -> str:
        """
        Retrieve context for a query. Returns formatted string.
        """
        query_vector = await self.embeddings.aembed_query(query)
        
        results = await self.vector_store.search(query_vector, limit=k)
        
        if not results:
            return ""
            
        formatted_context = "\n\n".join([doc.content for doc in results])
        return formatted_context
