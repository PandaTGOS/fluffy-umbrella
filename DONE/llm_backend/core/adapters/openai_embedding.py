import httpx
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings

from ..interfaces import EmbeddingProvider

class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model):
        self.model = model

    def configure(self):
        Settings.embed_model = OpenAIEmbedding(
            model=self.model,
            http_client=httpx.Client(verify=False)
        )
