from typing import List
from brainbox.core.knowledge.rerankers.base import Reranker
from brainbox.core.knowledge.documents import Document

class LLMReranker(Reranker):
    def __init__(self, client):
        self.client = client

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        # Simple heuristic for now:
        # boost docs containing key terms
        query_terms = set(query.lower().split())

        for doc in documents:
            # Simple content overlap check
            doc_terms = set(doc.content.lower().split())
            overlap = len(query_terms & doc_terms)
            # Boost score
            doc.score += overlap * 0.1

        # Re-sort desc
        documents.sort(key=lambda d: d.score, reverse=True)
        return documents
