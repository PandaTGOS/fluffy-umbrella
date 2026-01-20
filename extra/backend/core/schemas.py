from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "qwen:1.8b"
    temperature: float = 0.7
    base_url: Optional[str] = None
    api_key: Optional[str] = None

class RAGConfig(BaseModel):
    enabled: bool = False
    collection_name: Optional[str] = None
    k: int = 5

class AppConfig(BaseModel):
    id: str
    name: str 
    description: Optional[str] = None
    system_prompt: str = "You are a helpful assistant."
    llm: LLMConfig = Field(default_factory=LLMConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    tools: List[str] = Field(default_factory=list)
