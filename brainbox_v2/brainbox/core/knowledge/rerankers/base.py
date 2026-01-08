from abc import ABC, abstractmethod
from typing import List
from brainbox.core.knowledge.documents import Document

class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        pass
