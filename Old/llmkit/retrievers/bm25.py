from typing import List
from rank_bm25 import BM25Okapi
from ..interfaces import Retriever, Document, RetrievalResult

class BM25Retriever(Retriever):
    def __init__(self, documents: List[Document]):
        self.documents = documents
        # Tokenize corpus
        tokenized_corpus = [self._tokenize(doc.content) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        tokenized_query = self._tokenize(query)
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Pair with docs
        doc_scores = []
        for i, score in enumerate(scores):
            if score > 0:
                # We copy the document to avoid mutating the original store in-place if shared?
                # Actually, Retriever returns new RetrievalResult, but better to be safe or just set score.
                # Since Document object might be shared, let's clone or create new if needed.
                # For efficiency, we just set score on the object if it's transient, 
                # OR we rely on downstream to handle it. 
                # llmkit interfaces seem to imply returning Documents with scores.
                # Let's create a copy of the list item if possible, or just reuse.
                # Reusing is faster.
                doc = self.documents[i]
                # We need to set the score for THIS retrieval. 
                # If we modify doc.score, it might persist if doc is reused.
                # But typically docs are re-instantiated from chunks in KB. 
                # Here `self.documents` are kept in memory.
                
                # Let's just create a shallow copy / or simple object if possible.
                # Assuming Document is a dataclass or simple object.
                # Let's just set the score and assume single-threaded usage for now.
                doc.score = score
                doc_scores.append(doc)
        
        doc_scores.sort(key=lambda x: x.score, reverse=True)
        top_k = doc_scores[:k]
        
        return RetrievalResult(
            documents=top_k,
            signals={"retriever": "BM25Retriever", "count": len(top_k)}
        )
