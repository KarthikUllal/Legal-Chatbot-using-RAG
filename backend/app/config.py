# backend/app/config.py
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    DATA_DIR: str = "./datas"
    CHROMA_DIR: str = "./chroma_db"
    EMBED_MODEL: str= "models/embedding-001"
    LLM_PROVIDER: str = "gemini"  
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")     # fill via env
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120
    BATCH_SIZE: int = 32
    ALLOW_ORIGINS: list = ["*"]
    OLLAMA_MODEL: str ="llama3.2:3b"
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY")
    FORCE_LOCAL: bool = True
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
