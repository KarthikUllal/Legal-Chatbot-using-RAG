# backend/app/config.py
from pydantic import BaseSettings
import os

class Settings(BaseSettings):
    DATA_DIR: str = "./datas"
    CHROMA_DIR: str = "./chroma_db"
    EMBED_MODEL: str = "models/embedding-gecko-001"
    LLM_PROVIDER: str = "gemini"   # "gemini" | "local" | "openai" | "none"
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")     # fill via env
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 200
    BATCH_SIZE: int = 64
    ALLOW_ORIGINS: list = ["*"]

    class Config:
        env_file = ".env"

settings = Settings()
