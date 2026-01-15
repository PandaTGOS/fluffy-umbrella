from collections import defaultdict
from ..interfaces import Reranker


class DocumentReranker(Reranker):
    def __init__(self, max_docs: int = 3, chunks_per_doc: int = 2):
        self.max_docs = max_docs
        self.chunks_per_doc = chunks_per_doc

    def rerank(self, results, query: str = None):
        """
        Group chunks by document_id, rank documents,
        then select top chunks per document.
        """
        grouped = defaultdict(list)

        for r in results:
            doc_id = r["metadata"].get("document_id")
            if not doc_id:
                continue
            grouped[doc_id].append(r)

        ranked_docs = sorted(
            grouped.items(),
            key=lambda x: max(c["score"] for c in x[1]),
            reverse=True
        )

        final_results = []

        for _, chunks in ranked_docs[:self.max_docs]:
            top_chunks = sorted(
                chunks,
                key=lambda c: c["score"],
                reverse=True
            )[:self.chunks_per_doc]

            final_results.extend(top_chunks)

        return final_results
