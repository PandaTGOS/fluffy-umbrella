from typing import Optional
from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel
# from langchain_openai import ChatOpenAI 

from backend.core.config import settings

class LLMFactory:
    @staticmethod
    def get_chat_model(
        provider: str = settings.DEFAULT_LLM_PROVIDER, 
        model_name: str = settings.DEFAULT_MODEL_NAME,
        temperature: float = 0.7
    ) -> BaseChatModel:
        
        if provider == "ollama":
            return ChatOllama(
                base_url=settings.OLLAMA_BASE_URL,
                model=model_name,
                temperature=temperature
            )
        elif provider == "openai":
            # return ChatOpenAI(model=model_name, temperature=temperature, api_key=settings.OPENAI_API_KEY)
            raise NotImplementedError("OpenAI provider not yet enabled in factory")
        else:
            raise ValueError(f"Unknown provider: {provider}")
