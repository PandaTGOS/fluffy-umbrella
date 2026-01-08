from typing import List, Any
from brainbox.core.knowledge.retrievers.base import BaseRetriever
from brainbox.core.knowledge.retrieval_result import RetrievalResult
from brainbox.core.knowledge.indexing.inverted_index import InvertedIndex

class InvertedIndexRetriever(BaseRetriever):
    """
    Retriever backed by an InvertedIndex.
    """
    def __init__(self, index: InvertedIndex):
        self.index = index

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        all_matches = self.index.retrieve(query)
        
        # Simple scoring: Term Frequency in query matched in doc? 
        # Actually base InvertedIndex implementation returns OR matches.
        # Let's refine scoring here or in the Index.
        # For now, just trust the index results and maybe truncate to k.
        # A better implementation would have TF-IDF logic.
        
        # Let's do a simple count of query terms present in doc for scoring
        query_tokens = self.index._tokenize(query)
        scored_docs = []
        for doc in all_matches:
            doc_tokens = self.index._tokenize(doc.content)
            doc_tokens_set = set(doc_tokens)
            score = sum(1 for q_term in query_tokens if q_term in doc_tokens_set)
            
            doc.score = score
            scored_docs.append(doc)
            
        scored_docs.sort(key=lambda x: x.score, reverse=True)
        top_k = scored_docs[:k]
        
        return RetrievalResult(
            documents=top_k,
            signals={"retriever": "InvertedIndexRetriever", "count": len(top_k)}
        )
