from typing import List, Dict, Any, Optional
from collections import defaultdict
from brainbox.core.knowledge.retrievers import BaseRetriever
from brainbox.core.knowledge.documents import Document
from brainbox.core.knowledge.retrieval_result import RetrievalResult
from brainbox.core.knowledge.rerankers.base import Reranker

class CompositeRetriever(BaseRetriever):
    def __init__(
        self, 
        retrievers: List[BaseRetriever], 
        reranker: Optional[Reranker] = None,
        overfetch: int = 20
    ):
        self.retrievers = retrievers
        self.reranker = reranker
        self.overfetch = overfetch

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        import time
        start_time = time.time()
        
        # 1. Fan-out retrieval (high recall)
        all_docs: List[Document] = []
        signals = {
            "retrievers_used": [],
            "raw_counts": {},
            "components": {}
        }
        
        for retriever in self.retrievers:
            result = retriever.retrieve(query, k=self.overfetch)
            docs = result.documents
            name = retriever.__class__.__name__
            
            signals["retrievers_used"].append(name)
            signals["raw_counts"][name] = len(docs)
            signals["components"][name] = result.signals
            
            for d in docs:
                # Ensure metadata exists
                if d.metadata is None:
                    d.metadata = {}
                
                # Tag provenance
                d.metadata["retriever"] = name
                d.metadata["raw_score"] = d.score
            all_docs.extend(docs)

        if not all_docs:
            signals["retrieval_latency_ms"] = (time.time() - start_time) * 1000
            return RetrievalResult(documents=[], signals=signals)

        # 2. De-duplicate by doc.id (keep best raw score)
        best_by_id: Dict[str, Document] = {}
        for doc in all_docs:
            if doc.id not in best_by_id or doc.score > best_by_id[doc.id].score:
                best_by_id[doc.id] = doc

        unique_docs = list(best_by_id.values())
        signals["deduped_count"] = len(unique_docs)

        # 🔥 RERANK HERE (Before Normalization)
        if self.reranker:
            # Reranker takes the unique pool and re-scores/re-orders them
            unique_docs = self.reranker.rerank(query, unique_docs)

        # 3. Normalize scores (Min-Max) 
        # Reranked scores might need normalization too to fit [0,1] expectation for downstream logic
        scores = [d.score for d in unique_docs]
        if not scores:
             signals["retrieval_latency_ms"] = (time.time() - start_time) * 1000
             return RetrievalResult(documents=[], signals=signals)
             
        min_s, max_s = min(scores), max(scores)
        
        signals["avg_score"] = sum(scores) / len(scores)

        for d in unique_docs:
            if max_s > min_s:
                d.score = (d.score - min_s) / (max_s - min_s)
            else:
                d.score = 1.0

        # 4. Sort by normalized score (or reranker score if normalization skipped)
        unique_docs.sort(key=lambda d: d.score, reverse=True)

        # 5. Return top-k
        final_docs = unique_docs[:k]
        
        signals["retrieval_latency_ms"] = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            documents=final_docs,
            signals=signals
        )
