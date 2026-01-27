
import sys
import os
import logging
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(os.getcwd())

from llm_backend.config import load_app_configs
from llm_backend.core.adapters.pg_vector import PGVectorStore
from llm_backend.core.adapters.openai_embedding import OpenAIEmbeddingProvider
from llm_backend.core.services.document_loader import load_documents
from llm_backend.core.services.chunker import build_chunks

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IngestionService")

def ingest_app(cfg):
    logger.info(f"Starting ingestion for app: {cfg.app_name}")

    if not cfg.rag or not cfg.embedding or not cfg.vector_store:
        logger.warning(f"Skipping {cfg.app_name}: Missing RAG/Vector configuration.")
        return

    # 1. Configure Embeddings
    embed_provider = OpenAIEmbeddingProvider(cfg.embedding.model)
    embed_provider.configure()

    # 2. Load Documents
    logger.info(f"Loading documents from: {cfg.rag.data_path}")
    documents = load_documents(cfg.rag.data_path)
    logger.info(f"Loaded {len(documents)} documents.")

    # 3. Chunk Documents
    logger.info("Chunking documents...")
    nodes, leaf_nodes = build_chunks(
        documents,
        chunk_sizes=cfg.rag.chunk_sizes
    )
    logger.info(f"Created {len(nodes)} nodes and {len(leaf_nodes)} leaf nodes.")

    # 4. Build Vector Store (Upsert)
    if cfg.vector_store.type == "pgvector":
        store = PGVectorStore(
            db_url=cfg.vector_store.db_url,
            table_name=cfg.vector_store.table_name,
            collection_name=cfg.vector_store.collection
        )
        logger.info(f"Indexing to PGVector table: {cfg.vector_store.table_name}")
        store.build(nodes, leaf_nodes)
    else:
        logger.warning(f"Skipping ingestion for non-persistent store: {cfg.vector_store.type}")

    logger.info(f"Completed ingestion for {cfg.app_name}")


def main():
    load_dotenv()
    
    configs = load_app_configs("apps.yaml")
    
    for cfg in configs:
        if cfg.app_type == "rag":
            try:
                ingest_app(cfg)
            except Exception as e:
                logger.error(f"Failed to ingest {cfg.app_name}: {e}")

if __name__ == "__main__":
    main()
