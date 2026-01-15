from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict

from ..interfaces import Document, RetrievalResult


@dataclass
class ChunkSpec:
    size: int = 512          # tokens or chars (depending on chunker)
    overlap: int = 64
    strategy: str = "fixed"  # fixed | semantic | sentence

class Chunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        """Splits text into chunks."""
        pass


class EmbeddingClient(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        pass


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


class Reranker(ABC):
    @abstractmethod
    def rerank(self, results: List[Dict], query: str = None) -> List[Dict]:
        """
        Input: list of retrieval results
        Output: reranked list
        """
        pass