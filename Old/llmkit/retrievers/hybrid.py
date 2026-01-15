from typing import List, Dict, Any
from ..interfaces import Retriever, RetrievalResult, Document

class HybridRetriever(Retriever):
    def __init__(self, vector_retriever: Retriever, bm25_retriever: Retriever, alpha: float = 0.5):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha # Weight for vector search (0.0 to 1.0). 1.0 = Vector only, 0.0 = BM25 only.

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        # Retrieve more candidates for fusion
        candidate_k = k * 2 
        
        vector_res = self.vector_retriever.retrieve(query, k=candidate_k)
        bm25_res = self.bm25_retriever.retrieve(query, k=candidate_k)
        
        # Combine using Reciprocal Rank Fusion (RRF) or Weighted Score
        # Since scores are on different scales (Cosine 0-1, BM25 0-inf), RRF is safer.
        
        return self._rrf_fusion(vector_res.documents, bm25_res.documents, k)

    def _rrf_fusion(self, vector_docs: List[Document], bm25_docs: List[Document], k: int, c: int = 60) -> RetrievalResult:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        
        # Helper to get unique ID. If ID is missing, usage hash of content?
        # Assuming ID exists.
        
        for rank, doc in enumerate(vector_docs):
            doc_id = doc.id or str(hash(doc.content))
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (c + rank + 1))

        for rank, doc in enumerate(bm25_docs):
            doc_id = doc.id or str(hash(doc.content))
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (c + rank + 1))
            
        # Sort
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        top_ids = sorted_ids[:k]
        final_docs = []
        for doc_id in top_ids:
            doc = doc_map[doc_id]
            doc.score = scores[doc_id] # Update score to RRF score
            final_docs.append(doc)
            
        return RetrievalResult(
            documents=final_docs,
            signals={"retriever": "HybridRetriever-RRF", "count": len(final_docs)}
        )
