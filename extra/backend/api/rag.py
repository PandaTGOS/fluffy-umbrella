from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from backend.domain.rag.ingestion import IngestionService
from backend.domain.rag.service import RAGService
from backend.core.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)

class IngestRequest(BaseModel):
    directory_path: str

@router.post("/ingest")
async def ingest_documents(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Trigger async ingestion of a directory.
    """
    service = IngestionService()
    
    # Run in background to not block API
    background_tasks.add_task(service.ingest_directory, request.directory_path)
    
    return {"status": "Ingestion started", "directory": request.directory_path}

@router.post("/query")
async def query_rag(query: str, k: int = 5):
    """
    Direct RAG query test endpoint.
    """
    from backend.infrastructure.vector.postgres import PostgresVectorStore
    from langchain_ollama import OllamaEmbeddings
    from backend.core.config import settings

    store = PostgresVectorStore()
    embeddings = OllamaEmbeddings(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.DEFAULT_EMBEDDING_MODEL
    )

    service = RAGService(vector_store=store, embeddings=embeddings)
    context = await service.retrieve(query, k)
    return {"query": query, "context": context}
