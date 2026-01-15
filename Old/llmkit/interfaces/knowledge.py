from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Document:
    id: str
    content: str
    metadata: Dict[str, str]
    score: float = 0.0


@dataclass
class Context:
    documents: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    documents: List[Document]
    signals: Dict[str, Any]   # scores, diagnostics, retriever stats


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> RetrievalResult:
        pass