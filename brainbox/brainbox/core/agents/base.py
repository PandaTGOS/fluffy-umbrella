from abc import ABC, abstractmethod
from typing import Dict, Any
from brainbox.core.state.rag_state import RAGState

class Agent(ABC):
    name: str

    @abstractmethod
    def run(self, state: RAGState) -> Dict[str, Any]:
        """
        Execute the agent's logic.
        Returns a dictionary of updates to be merged into the state.
        """
        pass
