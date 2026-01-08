from typing import List, Optional
from brainbox.core.knowledge.retrievers.vector import VectorRetriever
from brainbox.core.knowledge.ingestion.loaders.directory import DirectoryLoader
from brainbox.core.knowledge.ingestion.loaders.base import FileLoader
from brainbox.core.knowledge.ingestion.pipeline import IngestionPipeline

# Default Loaders
from brainbox.core.knowledge.ingestion.loaders.text import TextLoader
from brainbox.core.knowledge.ingestion.loaders.markdown import MarkdownLoader

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
        vector_store
    ):
        """
        Seamlessly creates a RAG-ready Knowledge Base from a directory.
        """
        # 1. Setup Loaders (Use defaults if none provided)
        if loaders is None:
            loaders = [TextLoader(), MarkdownLoader()]
            
        loader = DirectoryLoader(loaders)
        
        # 2. Load Documents
        print(f"🌟 Loading knowledge from: {path}")
        docs = loader.load(path)
        print(f"📄 Found {len(docs)} documents.")

        # 3. Ingest
        ingestion = IngestionPipeline(
            chunker=chunker,
            embedding_client=embedding_client,
            vector_store=vector_store
        )
        ingestion.ingest(docs)

        # 4. Create Retriever
        retriever = VectorRetriever(embedding_client, vector_store)

        return cls(loader, ingestion, retriever)

    def as_retriever(self):
        return self.retriever
