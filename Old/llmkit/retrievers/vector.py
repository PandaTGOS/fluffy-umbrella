from ..interfaces import Document, RetrievalResult, Retriever, EmbeddingClient, VectorStore

class VectorRetriever(Retriever):
    def __init__(self, embedding_client: EmbeddingClient, vector_store: VectorStore):
        self.embedding_client = embedding_client
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 3) -> RetrievalResult:
        # 1. Embed the query
        # embed expects a list of texts and returns a list of embeddings
        embeddings = self.embedding_client.embed([query])
        if not embeddings:
            return []
        query_vector = embeddings[0]

        # 2. Search the vector store
        results = self.vector_store.search(query_vector, k=k)

        # 3. Convert results to Documents
        documents = []
        for result in results:
            meta = result.get("metadata", {})
            score = result.get("score")
            
            # Extract core fields if present in metadata, otherwise defaults
            doc_id = meta.get("id", "unknown")
            content = meta.get("content", "")
            
            # Create Document
            doc = Document(
                id=doc_id,
                content=content,
                metadata=meta,
                score=score
            )
            documents.append(doc)
            
        return RetrievalResult(
            documents=documents,
            signals={"retriever": "VectorRetriever", "count": len(documents), "index_type": "flat"}
        )