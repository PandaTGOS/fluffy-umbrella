from typing import List
import ollama
from brainbox.core.embeddings.base import EmbeddingClient

class OllamaEmbeddingClient(EmbeddingClient):
    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            # Ollama python lib 'embeddings' call:
            response = ollama.embeddings(model=self.model, prompt=text)
            embeddings.append(response["embedding"])
        return embeddings
