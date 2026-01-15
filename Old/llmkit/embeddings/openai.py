import httpx
from typing import List
from openai import OpenAI
from ..interfaces import EmbeddingClient

class OpenAIEmbeddingClient(EmbeddingClient):
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.client = OpenAI(http_client=httpx.Client(verify=False))

    def embed(self, texts: List[str]) -> List[List[float]]:
        # OpenAI allows batching; send all texts in one call
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        # Extract embeddings in order
        return [item.embedding for item in response.data]
