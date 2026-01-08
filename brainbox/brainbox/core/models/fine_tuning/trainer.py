from abc import ABC, abstractmethod
from .dataset import FineTuneDataset

class FineTuner(ABC):
    @abstractmethod
    def train(
        self,
        base_model: str,
        dataset: FineTuneDataset,
        output_name: str
    ) -> str:
        """
        Returns a model reference or adapter ID
        """
        pass
