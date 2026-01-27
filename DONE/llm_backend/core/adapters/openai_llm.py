import httpx
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

from ..interfaces import LLMProvider

class OpenAILLMProvider(LLMProvider):
    def __init__(self, model, system_prompt):
        self.model = model
        self.system_prompt = system_prompt

    def configure(self):
        Settings.llm = OpenAI(
            model=self.model,
            system_prompt=self.system_prompt,
            # max_tokens=256,
            temperature=0.1,
            http_client=httpx.Client(verify=False)
        )
