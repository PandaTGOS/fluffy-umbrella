from typing import List, Optional
from brainbox.core.knowledge import Document
# Helper types (using existing interfaces or defining typed protocols)
# Assuming Chunker, EmbeddingClient, VectorStore follow the core interfaces

class IngestionPipeline:
    def __init__(
        self,
        chunker,
        embedding_client,
        vector_store
    ):
        self.chunker = chunker
        self.embedding_client = embedding_client
        self.vector_store = vector_store

    def ingest(self, documents: List[Document]):
        if not documents:
            print("[INFO] No documents to ingest.")
            return

        chunks = []
        metadatas = []
        
        print(f"[INGEST] Processing {len(documents)} documents...")

        for doc in documents:
            doc_chunks = self.chunker.chunk(doc.content)
            for chunk_text in doc_chunks:
                chunks.append(chunk_text)
                # Combine doc metadata with chunk text for storage
                meta = doc.metadata.copy() if doc.metadata else {}
                meta.update({
                    "id": doc.id,
                    "content": chunk_text, 
                    "parent_id": doc.id
                })
                metadatas.append(meta)

        if not chunks:
            print("[INFO] No chunks generated.")
            return

        print(f"[INGEST] Embedding {len(chunks)} chunks...")
        vectors = self.embedding_client.embed(chunks)
        
        print(f"[INGEST] Storing in Vector Store...")
        self.vector_store.add(vectors, metadatas)
        print("[INGEST] Complete.")
