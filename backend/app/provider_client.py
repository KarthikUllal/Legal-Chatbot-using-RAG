# backend/app/provider_client.py
from typing import List, Optional
import logging
import os
from .config import settings

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ProviderClient:
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        raise NotImplementedError


class LocalProvider(ProviderClient):
    def __init__(self, model_name: Optional[str] = None):
        default_local = "all-MiniLM-L6-v2"
        cfg_name = model_name or getattr(settings, "EMBED_MODEL", None) or default_local

        if isinstance(cfg_name, str) and cfg_name.startswith("models/"):
            logger.info(f"Falling back to local model: {default_local}")
            cfg_name = default_local

        try:
            # Try new import path first (LangChain >= 0.1.0)
            from langchain_huggingface import HuggingFaceEmbeddings

            self.embeddings = HuggingFaceEmbeddings(model_name=cfg_name)
        except ImportError:
            try:
                # Fallback to old import path (LangChain < 0.1.0)
                from langchain.embeddings import HuggingFaceEmbeddings

                self.embeddings = HuggingFaceEmbeddings(model_name=cfg_name)
            except ImportError:
                raise ImportError(
                    "Install: pip install langchain sentence-transformers"
                )

        self.model_name = cfg_name
        logger.info(f"Loading local model with LangChain: {self.model_name}")
        logger.info("Local model loaded successfully with LangChain!")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Generating embeddings for {len(texts)} texts using LangChain")
        embs = self.embeddings.embed_documents(texts)
        logger.info(f"Generated {len(embs)} embeddings with {len(embs[0])} dimensions")
        return embs

    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        truncated_prompt = prompt[:200] + "..." if len(prompt) > 200 else prompt
        return f"[LOCAL MODEL] This is a simulated response to: '{truncated_prompt}'. For actual text generation, you would need to load a local LLM."


class GeminiClient(ProviderClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        emb_model: Optional[str] = None,
        gen_model: Optional[str] = None,
    ):
        try:
            import google.generativeai as genai

            # Try new import path first
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
            except ImportError:
                from langchain.embeddings import GoogleGenerativeAIEmbeddings
        except ImportError:
            raise ImportError(
                "Install: pip install google-generativeai langchain-google-genai"
            )

        self.genai = genai
        self.api_key = (
            api_key
            or getattr(settings, "GEMINI_API_KEY", None)
            or os.getenv("GEMINI_API_KEY")
        )
        if not self.api_key:
            raise ValueError("Gemini API key required")

        self.genai.configure(api_key=self.api_key)
        self.emb_model = emb_model or getattr(
            settings, "EMBED_MODEL", "models/embedding-001"
        )
        self.gen_model = gen_model or getattr(settings, "GEN_MODEL", "gemini-pro")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.emb_model, google_api_key=self.api_key
        )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Generating embeddings for {len(texts)} texts using Gemini")
        embs = self.embeddings.embed_documents(texts)
        logger.info(f"Generated {len(embs)} embeddings with {len(embs[0])} dimensions")
        return embs

    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        model = self.genai.GenerativeModel(self.gen_model)
        response = model.generate_content(prompt)
        return response.text


class OpenAIClient(ProviderClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        emb_model: Optional[str] = None,
        gen_model: Optional[str] = None,
    ):
        try:
            from openai import OpenAI

            # Try new import path first
            try:
                from langchain_openai import OpenAIEmbeddings
            except ImportError:
                from langchain.embeddings import OpenAIEmbeddings
        except ImportError:
            raise ImportError("Install: pip install openai langchain-openai")

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.emb_model = emb_model or "text-embedding-3-small"
        self.gen_model = gen_model or "gpt-3.5-turbo"

        self.embeddings = OpenAIEmbeddings(
            model=self.emb_model, openai_api_key=api_key or os.getenv("OPENAI_API_KEY")
        )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Generating embeddings for {len(texts)} texts using OpenAI")
        embs = self.embeddings.embed_documents(texts)
        logger.info(f"Generated {len(embs)} embeddings with {len(embs[0])} dimensions")
        return embs

    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.gen_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class HuggingFaceClient(ProviderClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        emb_model: Optional[str] = None,
        gen_model: Optional[str] = None,
    ):
        try:
            # Try new import path first
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
            except ImportError:
                from langchain.embeddings import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError("Install: pip install langchain-huggingface")

        self.api_key = api_key or os.getenv("HUGGINGFACE_TOKEN")
        self.emb_model = emb_model or "sentence-transformers/all-MiniLM-L6-v2"
        self.gen_model = gen_model or "google/flan-t5-large"

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.emb_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": False},
        )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Generating embeddings for {len(texts)} texts using HuggingFace")
        embs = self.embeddings.embed_documents(texts)
        logger.info(f"Generated {len(embs)} embeddings with {len(embs[0])} dimensions")
        return embs

    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        import requests

        response = requests.post(
            f"https://api-inference.huggingface.co/models/{self.gen_model}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"inputs": prompt, "parameters": {"max_length": max_tokens}},
        )
        return response.json()[0]["generated_text"]


class DeepSeekClient(ProviderClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        emb_model: Optional[str] = None,
        gen_model: Optional[str] = None,
    ):
        try:
            from openai import OpenAI

            # Try new import path first
            try:
                from langchain_openai import OpenAIEmbeddings
            except ImportError:
                from langchain.embeddings import OpenAIEmbeddings
        except ImportError:
            raise ImportError("Install: pip install openai langchain-openai")

        self.client = OpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1",
        )
        self.emb_model = emb_model or "deepseek-embedding"
        self.gen_model = gen_model or "deepseek-chat"

        self.embeddings = OpenAIEmbeddings(
            model=self.emb_model,
            openai_api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            openai_api_base="https://api.deepseek.com/v1",
        )

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Generating embeddings for {len(texts)} texts using DeepSeek")
        embs = self.embeddings.embed_documents(texts)
        logger.info(f"Generated {len(embs)} embeddings with {len(embs[0])} dimensions")
        return embs

    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.gen_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


def get_best_provider():
    """Get provider - prioritize local for testing"""
    # Force local provider for testing
    force_local = os.getenv("FORCE_LOCAL", "true").lower() in ("true", "1", "yes")
    if force_local:
        provider = LocalProvider()
        print("Using LocalProvider (forced for testing)")
        return provider

    # If user specifically wants a cloud provider, check those
    preferred_provider = os.getenv("PREFERRED_PROVIDER", "").lower()

    if preferred_provider == "openai" and os.getenv("OPENAI_API_KEY"):
        return OpenAIClient()
    elif preferred_provider == "deepseek" and os.getenv("DEEPSEEK_API_KEY"):
        return DeepSeekClient()
    elif preferred_provider == "gemini" and os.getenv("GEMINI_API_KEY"):
        return GeminiClient()
    elif preferred_provider == "huggingface" and os.getenv("HUGGINGFACE_TOKEN"):
        return HuggingFaceClient()

    # Default to local provider
    provider = LocalProvider()
    print("Using LocalProvider (default)")
    return provider


def get_local_provider():
    """Directly get local provider without any checks"""
    return LocalProvider()


if __name__ == "__main__":
    print("Testing LOCAL provider with LangChain...")

    # Test local provider directly
    provider = LocalProvider()

    # Test with sample legal texts
    samples = [
        "IPC Section 420: Cheating and dishonestly inducing delivery of property",
        "The Indian Penal Code defines cheating in section 415",
        "Whoever commits cheating shall be punished with imprisonment of either description for a term which may extend to seven years",
    ]

    print(f"Testing with {len(samples)} sample texts...")

    try:
        embeddings = provider.get_embeddings(samples)
        print(f" SUCCESS: Generated {len(embeddings)} embeddings using LangChain")
        print(f" Each embedding has {len(embeddings[0])} dimensions")
        print(f" Sample embedding (first 5 values): {embeddings[0][:5]}")

        # Test generation
        answer = provider.generate("Explain IPC Section 420 briefly")
        print(f" Generation test: {answer}")

        print("\n Local model with LangChain is working perfectly!")

    except Exception as e:
        print(f"ERROR: {e}")
        print(
            "Make sure you have installed: pip install langchain sentence-transformers"
        )
