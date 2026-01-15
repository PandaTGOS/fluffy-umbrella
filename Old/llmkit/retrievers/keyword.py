from typing import List
from ..interfaces import Retriever, Document, RetrievalResult

class KeywordRetriever(Retriever):
    def __init__(self, documents: List[Document]):
        self.documents = documents

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        query_terms = set(query.lower().split())

        # Calculate the score for each document based on keyword overlap
        scored_docs = []
        for doc in self.documents:
            doc_terms = set(doc.content.lower().split())
            overlap = len(query_terms & doc_terms)
            if overlap > 0:
                doc.score = overlap
                scored_docs.append(doc)

        # Sort documents by score (overlap) in descending order
        scored_docs.sort(key=lambda x: x.score, reverse=True)
        top_k_docs=scored_docs[:k]

        # Return top-k documents with the highest scores
        # top_k_docs = [doc for doc, score in scored_docs[:k]]

        return RetrievalResult(
            documents=top_k_docs,
            signals={"retriever": "KeywordRetriever", "count": len(top_k_docs)}
        )