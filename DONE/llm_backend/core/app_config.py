from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, List

class LLMConfig(BaseModel):
    provider: str
    model: str
    system_prompt: str

class EmbeddingConfig(BaseModel):
    provider: str
    model: str

class VectorStoreConfig(BaseModel):
    type: Literal["in_memory", "pgvector"]
    collection: str
    db_url: Optional[str] = None
    table_name: Optional[str] = None

class RerankerConfig(BaseModel):
    provider: Literal["none", "cohere", "colbert"] = "none"
    model: Optional[str] = None
    top_n: int = 5

class RAGConfig(BaseModel):
    data_path: str
    chunk_sizes: List[int] = Field(default=[500], alias="chunk_size") 
    top_k: int = 5
    use_hybrid_search: bool = False
    reranker: Optional[RerankerConfig] = None
    qa_template: str = (
        "Use the context below to answer the question.\n\n{context}\n\nQuestion: {query}"
    )

class AppConfig(BaseModel):
    app_name: str
    app_type: Literal["chat", "rag"]

    llm: LLMConfig

    # RAG-only (optional at schema level)
    embedding: Optional[EmbeddingConfig] = None
    vector_store: Optional[VectorStoreConfig] = None
    rag: Optional[RAGConfig] = None