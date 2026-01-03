# backend/app/provider_client.py
from typing import List, Optional
import logging
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ProviderClient:
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        raise NotImplementedError


class NVIDIAProvider(ProviderClient):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not found in environment variables")

        self.base_url = "https://integrate.api.nvidia.com/v1"  # NVIDIA NIM endpoint

        # Use Llama 3.1 8B Instruct as default (good balance of speed/quality)
        self.model = "meta/llama-3.1-8b-instruct"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Initialize embeddings locally
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info(" Local embeddings loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to load HuggingFace embeddings: {e}")
            raise

        # Test connection to NVIDIA
        self._test_connection()
        logger.info(f" NVIDIA NIM Provider initialized with model: {self.model}")

    def _test_connection(self):
        """Test NVIDIA NIM API connection"""
        try:
            # Simple test to check API key validity
            test_response = requests.get(
                f"{self.base_url}/models", headers=self.headers, timeout=10
            )

            if test_response.status_code == 200:
                logger.info(" NVIDIA NIM API connection successful")
            else:
                logger.warning(
                    f" NVIDIA API test returned: {test_response.status_code}"
                )
                # Don't raise error - API key might still work for chat completions

        except Exception as e:
            logger.error(f" NVIDIA NIM connection test failed: {e}")
            # Don't raise error - let it fail gracefully during actual generation

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using local HuggingFace model"""
        logger.info(f"Generating embeddings for {len(texts)} texts")
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Return zero vectors as fallback (prevents complete crash)
            embedding_size = 384  # all-MiniLM-L6-v2 dimension
            return [[0.0] * embedding_size for _ in range(len(texts))]

    def generate(self, prompt: str, max_tokens: int = 1000, **kwargs) -> str:
        """Generate legal analysis using NVIDIA NIM"""
        try:
            logger.info(f"Generating legal analysis via NVIDIA NIM ({self.model})...")

            # Use the same prompt format that's working in your current system
            messages = [
                {
                    "role": "system",
                    "content": """You are "Nyaya Mitra" - India's legal expert. 
                    Provide accurate legal information based strictly on the provided context. 
                    Be practical and helpful. 
                    Use relevant laws from context. 
                    When explaining procedures, give step-by-step instructions. 
                    Mention BNS/IPC when applicable.""",
                },
                {"role": "user", "content": prompt},
            ]

            start_time = time.time()

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "stream": False,
                },
                timeout=60,
            )

            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                generated_text = result["choices"][0]["message"]["content"].strip()

                # Clean up response (remove extra whitespace)
                import re

                generated_text = re.sub(r"\n\s*\n", "\n\n", generated_text)

                logger.info(
                    f"NVIDIA NIM success: {len(generated_text)} chars in {elapsed_time:.2f}s"
                )
                return generated_text

            elif response.status_code == 401:
                logger.error(" NVIDIA API key invalid or expired")
                return self._get_auth_error_response()

            elif response.status_code == 429:
                logger.warning("⚠️ NVIDIA NIM rate limit reached")
                return self._get_rate_limit_response()

            else:
                error_text = (
                    response.text[:200] if response.text else "No error details"
                )
                logger.error(
                    f" NVIDIA NIM error {response.status_code}: {error_text}"
                )
                return self._get_fallback_response()

        except requests.exceptions.Timeout:
            logger.error("NVIDIA NIM request timeout")
            return self._get_timeout_response()

        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to NVIDIA NIM")
            return self._get_connection_error_response()

        except Exception as e:
            logger.error(f"NVIDIA NIM generation error: {e}")
            return self._get_fallback_response()

    def set_model(self, model_name: str):
        """Change the NVIDIA model being used"""
        # You can hardcode supported models or fetch from API
        supported_models = [
            "meta/llama-3.1-8b-instruct",
            "meta/llama-3.1-70b-instruct",
            "mistralai/mistral-7b-instruct-v0.3",
            "google/gemma-2-9b-it",
        ]

        if model_name in supported_models:
            self.model = model_name
            logger.info(f"Changed NVIDIA model to: {model_name}")
        else:
            logger.warning(
                f"Model {model_name} not in supported list. Keeping: {self.model}"
            )

    def _get_fallback_response(self) -> str:
        """Fallback when NVIDIA fails"""
        return """I couldn't access the legal analysis service at the moment. 

However, based on the legal documents in our database, I can tell you:

• Relevant legal sections have been retrieved for your query
• The information is based on Indian legal documents (IPC, BNS, etc.)
• Please try again in a moment for detailed analysis

For immediate legal assistance, consult a qualified legal professional."""

    def _get_auth_error_response(self) -> str:
        return """System Notice: NVIDIA API Authentication Error

The AI service authentication failed. 

The system will continue to retrieve relevant legal documents, but detailed analysis is temporarily unavailable.

Please check with system administrator about NVIDIA API key."""

    def _get_rate_limit_response(self) -> str:
        return """Legal Analysis - Service Limit

The AI service is currently experiencing high demand. 

Providing key information from available legal documents:

• Relevant legal sections identified
• Document references available below
• Please try again later for detailed analysis"""

    def _get_timeout_response(self) -> str:
        return """Legal Analysis - Processing

The AI service is taking longer than expected. 

Key legal references have been retrieved. Please check the source documents below for immediate information.

Detailed analysis will be available once the service responds."""

    def _get_connection_error_response(self) -> str:
        return """System Notice: AI Service Unavailable

Cannot connect to the AI analysis service. 

However, relevant legal documents have been retrieved for your query. Please check the source references below.

Try again shortly for AI-powered analysis."""


def get_best_provider():
    """Get the NVIDIA provider (only provider now)"""
    logger.info("Initializing NVIDIA Provider...")

    nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    use_nvidia = os.getenv("USE_NVIDIA", "true").lower() in ("true", "1", "yes")

    if not use_nvidia:
        logger.warning("USE_NVIDIA is false. System requires NVIDIA provider.")
        # You might want to make this mandatory now
        use_nvidia = True

    if not nvidia_key:
        logger.error("NVIDIA_API_KEY not found in environment variables!")
        raise ValueError(
            "NVIDIA_API_KEY is required. Please set it in your .env file:\n"
            "NVIDIA_API_KEY=your_key_here\n"
            "USE_NVIDIA=true"
        )

    try:
        provider = NVIDIAProvider()
        logger.info(f"✅ NVIDIA NIM provider initialized with model: {provider.model}")
        return provider
    except Exception as e:
        logger.error(f"NVIDIA provider initialization failed: {e}")
        # If NVIDIA fails, the system won't work - raise error
        raise Exception(
            "NVIDIA provider initialization failed. "
            "Please check: 1) NVIDIA_API_KEY is valid, 2) Internet connection, "
            "3) NVIDIA API service status"
        )
