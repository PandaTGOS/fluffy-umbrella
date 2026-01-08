from typing import List, Optional, Any
from brainbox.core.vectorstore.base import BaseVectorStore
from brainbox.core.knowledge.documents import Document
from brainbox.core.knowledge.retrieval_result import RetrievalResult

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("Warning: 'chromadb' not found. Using in-memory fallback for ChromaVectorStore.")


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, collection_name: str = "brainbox_store", persist_directory: str = "./chroma_db"):
        self.use_mock = not CHROMA_AVAILABLE
        if not self.use_mock:
            self.client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.client.get_or_create_collection(name=collection_name)
        else:
            self.mock_store = {} # id -> doc

    def add(self, documents: List[Document]):
        """
        Add documents to the vector store.
        """
        if not documents:
            return

        if not self.use_mock:
            ids = [doc.id for doc in documents]
            documents_text = [doc.content for doc in documents]
            metadatas = [doc.metadata if doc.metadata else {} for doc in documents]
            
            self.collection.upsert(
                ids=ids,
                documents=documents_text,
                metadatas=metadatas
            )
        else:
            for doc in documents:
                self.mock_store[doc.id] = doc

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        """
        Retrieve documents similar to the query.
        """
        if not self.use_mock:
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )

            ids = results["ids"][0]
            distances = results["distances"][0]
            metadatas = results["metadatas"][0]
            documents_text = results["documents"][0]

            retrieved_docs = []
            for i, doc_id in enumerate(ids):
                score = 1.0 / (1.0 + distances[i]) 
                doc = Document(
                    id=doc_id,
                    content=documents_text[i],
                    metadata=metadatas[i],
                    score=score
                )
                retrieved_docs.append(doc)
        else:
            # Simple mock retrieval: text contains query word?
            # Or just return all because it's a small demo
            retrieved_docs = []
            for doc in self.mock_store.values():
                # Mock similarity
                if query.lower() in doc.content.lower():
                     doc.score = 0.9
                     retrieved_docs.append(doc)
            
            # Sort and slice
            retrieved_docs.sort(key=lambda x: x.score, reverse=True)
            retrieved_docs = retrieved_docs[:k]

        return RetrievalResult(
            documents=retrieved_docs,
            signals={"retriever": "ChromaVectorStore", "count": len(retrieved_docs), "mocked": self.use_mock}
        )

    def delete(self, ids: List[str]):
        """
        Delete documents by ID.
        """
        if not self.use_mock:
            self.collection.delete(ids=ids)
        else:
            for i in ids:
                self.mock_store.pop(i, None)
