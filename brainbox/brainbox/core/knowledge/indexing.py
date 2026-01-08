from typing import List
from brainbox.core.knowledge.documents import Document
from brainbox.core.embeddings.base import EmbeddingClient
from brainbox.core.vectorstore.base import VectorStore

def index_documents(
    documents: List[Document],
    embedding_client: EmbeddingClient,
    vector_store: VectorStore,
):
    if not documents:
        return

    # Extract texts to embed
    texts = [doc.content for doc in documents]
    
    # Generate embeddings
    vectors = embedding_client.embed(texts)
    
    # Prepare metadata
    metadatas = []
    for doc in documents:
        # Base metadata includes ID and Content as per requirements
        meta = {
            "id": doc.id,
            "content": doc.content,
        }
        # Merge with existing document metadata if any
        if doc.metadata:
            meta.update(doc.metadata)
        metadatas.append(meta)
        
    # Add to store
    vector_store.add(vectors, metadatas)
