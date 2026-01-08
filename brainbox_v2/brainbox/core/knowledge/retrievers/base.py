from abc import ABC, abstractmethod
from typing import List
from ..documents import Document
from ..retrieval_result import RetrievalResult

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        pass
