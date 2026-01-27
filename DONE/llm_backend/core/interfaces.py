from abc import ABC, abstractmethod
from typing import Tuple, Protocol, Dict, Any

class LLMProvider(ABC):
    @abstractmethod
    def configure(self) -> None:
        ...


class EmbeddingProvider(ABC):
    @abstractmethod
    def configure(self) -> None:
        ...


class VectorStore(ABC):
    @abstractmethod
    def build(self, nodes, leaf_nodes) -> Tuple[object, object]:
        ...


class DocumentLoader(ABC):
    @abstractmethod
    def load(self):
        ...


class ChatApplication(Protocol):
    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        ...