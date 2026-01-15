from abc import ABC, abstractmethod
from typing import List
from ..interfaces import Document

class FileLoader(ABC):
    @abstractmethod
    def can_load(self, path: str) -> bool:
        pass

    @abstractmethod
    def load(self, path: str) -> List[Document]:
        pass