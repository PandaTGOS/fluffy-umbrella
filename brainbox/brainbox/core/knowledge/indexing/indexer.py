from typing import List
from brainbox.core.knowledge.chunking.base import Chunker
from brainbox.core.embeddings.base import EmbeddingClient
from brainbox.core.vectorstore.base import VectorStore
from brainbox.core.knowledge.documents import Document

def index_documents(
    documents: List[Document],
    chunker: Chunker,
    embedding_client: EmbeddingClient,
    vector_store: VectorStore
):
    """
    Orchestrates the indexing process: Chunk -> Embed -> Store.
    """
    texts = []
    metadatas = []

    for doc in documents:
        # 1. Chunking
        chunks = chunker.chunk(doc.content)
        
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            # Create richer metadata linking back to parent doc
            metadatas.append({
                "doc_id": doc.id,
                "chunk_id": i,
                "content": chunk,  # Store content in metadata for retrieval reconstruction
                **(doc.metadata or {}) # Merge original metadata
            })

    if not texts:
        return

    # 2. Embedding
    vectors = embedding_client.embed(texts)
    
    # 3. Storage
    vector_store.add(vectors, metadatas)
    print(f"[INDEXER] Indexed {len(vectors)} chunks from {len(documents)} documents.")
