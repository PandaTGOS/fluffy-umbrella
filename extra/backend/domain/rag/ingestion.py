import os
from typing import List, Dict, Any
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from backend.domain.rag.service import RAGService
from backend.core.logger import get_logger

logger = get_logger(__name__)

class IngestionService:
    def __init__(self):
        from backend.infrastructure.vector.postgres import PostgresVectorStore
        from langchain_ollama import OllamaEmbeddings
        from backend.core.config import settings

        store = PostgresVectorStore()
        embeddings = OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.DEFAULT_EMBEDDING_MODEL
        )
        self.rag_service = RAGService(vector_store=store, embeddings=embeddings)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True
        )

    def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Replicates legacy metadata extraction:
        - language (en/sv based on path)
        - category (parent dir name)
        - filename
        """
        path_lower = file_path.lower()
        return {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "category": os.path.basename(os.path.dirname(file_path)),
            "language": "en" if "english" in path_lower else "sv"
        }

    async def ingest_directory(self, directory_path: str):
        """
        Recursively ingest all .txt/.md files in directory.
        """
        logger.info(f"Starting ingestion for directory: {directory_path}")
        
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")

        documents: List[Document] = []

        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.endswith((".txt", ".md", ".py")): # Added .py for code RAG
                    file_path = os.path.join(root, file)
                    try:
                        loader = TextLoader(file_path, encoding='utf-8')
                        loaded_docs = loader.load()
                        
                        # Enrich metadata
                        metadata = self._extract_metadata(file_path)
                        for doc in loaded_docs:
                            doc.metadata.update(metadata)
                            
                        documents.extend(loaded_docs)
                    except Exception as e:
                        logger.warning(f"Failed to load {file_path}: {e}")

        if not documents:
            logger.warning("No documents found to ingest.")
            return

        # Chunking
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} files")

        # Ingest into Vector Store
        # We need to adapt RAGService to accept Batch Documents for efficiency
        # For now, we loop (optimizable later) 
        # TODO: Update RAGService to have `add_documents` method taking list suitable for RAG
        
        # Mapping to VectorDocument
        from backend.infrastructure.vector.interface import VectorDocument
        
        vector_docs = []
        # We process in batches to avoid overwhelming embedding API
        batch_size = 50 
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_texts = [d.page_content for d in batch]
            
            # Embed batch
            try:
                embeddings = await self.rag_service.embeddings.aembed_documents(batch_texts)
                
                for j, doc_chunk in enumerate(batch):
                    vector_docs.append(VectorDocument(
                        content=doc_chunk.page_content,
                        metadata=doc_chunk.metadata,
                        embedding=embeddings[j]
                    ))
            except Exception as e:
                logger.error(f"Failed to embed batch {i}: {e}")
                continue

        if vector_docs:
            await self.rag_service.vector_store.add_documents(vector_docs)
            logger.info(f"Successfully ingested {len(vector_docs)} vectors")

