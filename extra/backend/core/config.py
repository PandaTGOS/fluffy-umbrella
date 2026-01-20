from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os

class Settings(BaseSettings):
    # App
    PROJECT_NAME: str = "Enterprise LLM Backend"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "local" # local, staging, production
    
    # LLM Defaults
    DEFAULT_LLM_PROVIDER: str = "ollama"
    DEFAULT_MODEL_NAME: str = "qwen:1.8b"
    DEFAULT_EMBEDDING_MODEL: str = "nomic-embed-text:latest"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # Validation: OpenAI
    OPENAI_API_KEY: Optional[str] = None
    
    # Database
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_DB: str = "llm_backend"
    POSTGRES_PORT: int = 5432
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore"
    )

settings = Settings()
