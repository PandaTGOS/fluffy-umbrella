from typing import List, Optional
from llmkit.retrievers import VectorRetriever
from .ingestion.pipeline import IngestionPipeline
from .ingestion.loaders import DirectoryLoader, TextLoader, MarkdownLoader
from ..interfaces import FileLoader


class DirectoryKnowledgeBase:
    def __init__(
        self,
        loader: DirectoryLoader,
        ingestion_pipeline: IngestionPipeline,
        retriever: VectorRetriever
    ):
        self.loader = loader
        self.pipeline = ingestion_pipeline
        self.retriever = retriever

    @classmethod
    def from_path(
        cls,
        path: str,
        *,
        loaders: Optional[List[FileLoader]] = None,
        chunker,
        embedding_client,
        vector_store,
        enable_bm25: bool = False
    ):
        """
        Seamlessly creates a RAG-ready Knowledge Base from a directory.
        """
        # 1. Setup Loaders (Use defaults if none provided)
        if loaders is None:
            loaders = [TextLoader(), MarkdownLoader()]
            
        loader = DirectoryLoader(loaders)
        
        # 2. Load Documents
        print(f"Loading knowledge from: {path}")
        docs = loader.load(path)
        print(f"Found {len(docs)} documents")

        # 3. Ingest
        ingestion = IngestionPipeline(
            chunker=chunker,
            embedding_client=embedding_client,
            vector_store=vector_store
        )
        # Modified IngestionPipeline returns list of chunk metadatas
        chunk_metadatas = ingestion.ingest(docs)

        # 4. Create Retriever
        vector_retriever = VectorRetriever(embedding_client, vector_store)
        
        bm25_retriever = None
        if enable_bm25 and chunk_metadatas:
            print("[INFO] Building BM25 Index...")
            from ..interfaces import Document
            from llmkit.retrievers import BM25Retriever
            
            # Convert metadatas back to Documents for BM25
            bm25_docs = []
            for m in chunk_metadatas:
                # Ensure we have content
                if "content" in m:
                    bm25_docs.append(Document(
                        id=m.get("id"),
                        content=m.get("content"),
                        metadata=m
                    ))
            
            if bm25_docs:
                bm25_retriever = BM25Retriever(bm25_docs)
                print("[INFO] BM25 Index ready.")

        kb = cls(loader, ingestion, vector_retriever)
        if bm25_retriever:
            kb.bm25_retriever = bm25_retriever
            
        return kb

    def as_retriever(self):
        if hasattr(self, "bm25_retriever") and self.bm25_retriever:
            from llmkit.retrievers import HybridRetriever
            return HybridRetriever(
                vector_retriever=self.retriever,
                bm25_retriever=self.bm25_retriever
            )
        return self.retriever