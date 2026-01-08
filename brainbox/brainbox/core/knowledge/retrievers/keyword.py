from typing import List
from .base import BaseRetriever
from ..documents import Document
from ..retrieval_result import RetrievalResult

class KeywordRetriever(BaseRetriever):
    def __init__(self, documents: List[Document]):
        self.documents = documents

    def retrieve(self, query: str, k: int = 3) -> RetrievalResult:
        query_terms = set(query.lower().split())  # Lowercase and split the query into terms

        # Calculate the score for each document based on keyword overlap
        scored_docs = []
        for doc in self.documents:
            doc_terms = set(doc.content.lower().split())  # Lowercase and split document into terms
            overlap = len(query_terms & doc_terms)  # Count the overlapping words
            if overlap > 0:
                doc.score = overlap
                scored_docs.append((doc, overlap))

        # Sort documents by score (overlap) in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top-k documents with the highest scores
        top_k_docs = [doc for doc, score in scored_docs[:k]]

        return RetrievalResult(
            documents=top_k_docs,
            signals={"retriever": "KeywordRetriever", "count": len(top_k_docs)}
        )