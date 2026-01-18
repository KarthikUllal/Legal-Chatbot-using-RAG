# backend/app/config.py
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    DATA_DIR: str = "./datas"
    CHROMA_DIR: str = "./chroma_db"
    EMBED_MODEL: str= "models/embedding-001"

    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    BATCH_SIZE: int = 32
    ALLOW_ORIGINS: list = ["*"]
    OLLAMA_MODEL: str ="llama3.2:3b"
    FORCE_LOCAL: bool = False # make sure to change it to True when you dont use openRouter api key

    # NVIDIA Configuration
    NVIDIA_API_KEY: str = os.getenv("NVIDIA_API_KEY", "")
    NVIDIA_MODEL: str = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
    USE_NVIDIA: bool = os.getenv("USE_NVIDIA", "true").lower() == "true"

    #OpenRouter Configuration
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL: str = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free")
    USE_OPENROUTER: bool = os.getenv("USE_OPENROUTER", "false").lower() == "true"

    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    USE_WEB_FALLBACK: bool = os.getenv("USE_WEB_FALLBACK", "true").lower() == "true"  # Default to enabled

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
