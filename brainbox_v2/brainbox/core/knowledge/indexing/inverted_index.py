from typing import Dict, List, Set, Any
import re
from brainbox.core.knowledge.documents import Document

class InvertedIndex:
    """
    A simple in-memory inverted index.
    Maps terms -> list of document IDs.
    """
    def __init__(self):
        self.index: Dict[str, Set[str]] = {}
        self.doc_store: Dict[str, Document] = {}

    def _tokenize(self, text: str) -> List[str]:
        # Simple tokenization: lowercase and alphanumeric only
        return re.findall(r'\b\w+\b', text.lower())

    def add(self, document: Document):
        """
        Add a document to the inverted index.
        """
        self.doc_store[document.id] = document
        tokens = self._tokenize(document.content)
        
        for token in set(tokens): # Use set to avoid duplicates per doc
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(document.id)

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve documents containing terms from the query.
        Returns OR logic (any term matches).
        """
        query_tokens = self._tokenize(query)
        matching_doc_ids = set()
        
        for token in query_tokens:
            if token in self.index:
                matching_doc_ids.update(self.index[token])
        
        return [self.doc_store[doc_id] for doc_id in matching_doc_ids]

    def delete(self, doc_id: str):
        """
        Remove a document from the index.
        Note: This is expensive in a simple inverted index as we have to scan keys.
        """
        if doc_id not in self.doc_store:
            return
        
        del self.doc_store[doc_id]
        
        # Prune index
        for term, doc_ids in self.index.items():
            if doc_id in doc_ids:
                doc_ids.discard(doc_id)
        
        # Cleanup empty terms (optional)
        self.index = {k: v for k, v in self.index.items() if v}
