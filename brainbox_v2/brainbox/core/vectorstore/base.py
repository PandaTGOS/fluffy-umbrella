from abc import ABC, abstractmethod
from typing import List, Any
from brainbox.core.knowledge.documents import Document
from brainbox.core.knowledge.retrieval_result import RetrievalResult

class VectorStore(ABC):
    @abstractmethod
    def add(self, vectors: List[List[float]], metadatas: List[dict]) -> None:
        pass

    @abstractmethod
    def search(self, query_vector: List[float], k: int) -> List[dict]:
        pass

class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, documents: List[Document]):
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        pass

    @abstractmethod
    def delete(self, ids: List[str]):
        pass
