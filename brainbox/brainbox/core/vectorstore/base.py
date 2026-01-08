from abc import ABC, abstractmethod
from typing import List

class VectorStore(ABC):
    @abstractmethod
    def add(self, vectors: List[List[float]], metadatas: List[dict]) -> None:
        pass

    @abstractmethod
    def search(self, query_vector: List[float], k: int) -> List[dict]:
        pass
