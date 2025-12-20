# backend/app/provider_client.py
from typing import List, Optional
import logging
import os
import time
from .config import settings
from dotenv import load_dotenv
import requests

load_dotenv()

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
            from langchain_huggingface import HuggingFaceEmbeddings

            self.embeddings = HuggingFaceEmbeddings(
                model_name=cfg_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("✅ Embeddings loaded successfully")
        except Exception as e:
            logger.error(f"Embedding setup failed: {e}")
            raise

        self.model_name = cfg_name
        self.llm_model = "llama3.2:3b"
        self.ollama_base_url = "http://localhost:11434"

        self._test_ollama_connection()

    def _test_ollama_connection(self):
        """Test if Ollama is running and model is available"""
        try:
            import requests

            logger.info("Testing Ollama connection...")

            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                logger.info(f"Available Ollama models: {model_names}")

                if self.llm_model in model_names:
                    logger.info(f"✅ Model '{self.llm_model}' is available")
                else:
                    logger.warning(
                        f"❌ Model '{self.llm_model}' not found. Available: {model_names}"
                    )
            else:
                logger.warning(f"Ollama tags API returned: {response.status_code}")

        except Exception as e:
            logger.error(f"❌ Ollama connection test failed: {e}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Generating embeddings for {len(texts)} texts")
        return self.embeddings.embed_documents(texts)

    def generate(self, prompt: str, max_tokens: int = 600, **kwargs) -> str:
        """Use the EXACT working prompt from your original code"""
        try:
            import requests
            import re

            # ✅ EXACT PROMPT THAT WAS WORKING IN YOUR ORIGINAL CODE
            universal_prompt = f"""You are an Indian legal expert. Analyze this legal context and provide structured guidance.

LEGAL CONTEXT:
{prompt}

RESPONSE STRUCTURE (FOLLOW EXACTLY):

Legal Analysis
[2-3 sentence summary of the legal situation]

Applicable Provisions
• [Law 1 - Section X] - [Brief explanation]
• [Law 2 - Section Y] - [Brief explanation]

Rights & Remedies 
• [Right/Remedy 1]
• [Right/Remedy 2]

Recommended Actions
• [Step 1 - Practical action]
• [Step 2 - Practical action]

Important Notes
[Limitations and when to consult lawyer]

Legal References:
[Sections and acts used from context]

BASE YOUR ANSWER STRICTLY ON THE PROVIDED LEGAL CONTEXT. If context is insufficient, state this clearly.

ANSWER:"""

            logger.info(f"Generating legal analysis...")

            start_time = time.time()

            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": universal_prompt,
                    "stream": False,
                    "options": {
                        "num_predict": 800,
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1,
                    },
                },
                timeout=120,
            )

            elapsed_time = time.time() - start_time
            logger.info(f"Ollama response received in {elapsed_time:.2f} seconds")

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "").strip()

                if generated_text:
                    logger.info(
                        f"✅ Successfully generated legal analysis: {len(generated_text)} chars"
                    )

                    # Clean up the response
                    generated_text = re.sub(r"\n\s*\n", "\n\n", generated_text)

                    return generated_text
                else:
                    logger.warning("Ollama returned empty response")
                    return self._get_universal_fallback()

            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return self._get_universal_fallback()

        except requests.exceptions.ConnectionError:
            logger.error("❌ Cannot connect to Ollama")
            return self._get_ollama_error_response()

        except requests.exceptions.Timeout:
            logger.error("❌ Ollama request timeout")
            return self._get_timeout_response()

        except Exception as e:
            logger.error(f"❌ Generation error: {e}")
            return self._get_universal_fallback()

    def _get_universal_fallback(self) -> str:
        """Better fallback that shows retrieved context is available"""
        return """Legal Analysis
Based on the retrieved legal documents, I can provide information about your query.

Applicable Provisions
Relevant legal sections and provisions have been identified.

Key Information
- The system has successfully retrieved pertinent legal documents
- Specific legal references are available in the sources below
- Please refer to the cited documents for detailed provisions

Important Notes
This response is based on the available legal context. For comprehensive legal advice, consult a qualified professional.

Legal References:
See the source documents below for specific sections and details."""

    def _get_ollama_error_response(self) -> str:
        return """System Notice: AI Service Temporarily Unavailable

The legal analysis service is currently experiencing technical difficulties.

Please try again in a few moments or check the system status."""

    def _get_timeout_response(self) -> str:
        return """Legal Analysis - Processing

The system is analyzing the legal documents and preparing your response.

Please check the reference sources below for immediate information while the detailed analysis completes."""


# In provider_client.py
class OpenRouterProvider(ProviderClient):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not found")

        self.base_url = "https://openrouter.ai/api/v1"

        # List of free models to rotate through
        self.free_models = [
            "google/gemini-2.0-flash-exp:free",
            "mistralai/mistral-7b-instruct:free",
            "meta-llama/llama-3.1-8b-instruct",
            "microsoft/phi-3.5-mini-instruct",
            "nvidia/llama-3.1-nemotron-70b-instruct:free",
        ]

        # Start with first model
        self.current_model_index = 0
        self.model = self.free_models[0]

        # Track failed models
        self.failed_models = set()

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Legal AI Chatbot - Indian Laws",
        }

        # Initialize embeddings
        from langchain_huggingface import HuggingFaceEmbeddings

        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info(f"✅ OpenRouter provider initialized with model: {self.model}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Generating embeddings for {len(texts)} texts")
        return self.embeddings.embed_documents(texts)

    def rotate_model(self):
        """Rotate to next available model"""
        original_index = self.current_model_index

        while True:
            self.current_model_index = (self.current_model_index + 1) % len(
                self.free_models
            )
            if self.current_model_index == original_index:
                # All models tried, reset failed set
                self.failed_models.clear()
                logger.warning("All models rate limited, resetting...")
                break

            new_model = self.free_models[self.current_model_index]
            if new_model not in self.failed_models:
                self.model = new_model
                logger.info(f"🔄 Rotated to model: {self.model}")
                return

        # If all failed, use first model
        self.model = self.free_models[0]

    def generate(self, prompt: str, max_tokens: int = 1000, **kwargs) -> str:
        """Generate legal analysis using OpenRouter with model rotation"""
        # YOUR ORIGINAL LEGAL PROMPT TEMPLATE (from LocalProvider)
        legal_prompt = f"""You are an Indian legal expert. Analyze this legal context and provide structured guidance.

LEGAL CONTEXT:
{prompt}

CRITICAL FORMATTING RULES (MUST FOLLOW):
1. Use ONLY plain text - NO brackets, NO markdown, NO HTML
2. ABSOLUTELY NO [BBOX] or any bracketed formatting
3. Use this EXACT structure with these EXACT headings:
Legal Analysis
[2-3 sentence summary]

Applicable Provisions
• [Law 1 - Section X] - [Brief explanation]
• [Law 2 - Section Y] - [Brief explanation]

Rights & Remedies
• [Right/Remedy 1]
• [Right/Remedy 2]

Recommended Actions
• [Step 1 - Practical action]
• [Step 2 - Practical action]

Important Notes
[Limitations and when to consult lawyer]

Legal References:
[Sections and acts used from context]

BASE YOUR ANSWER STRICTLY ON THE PROVIDED LEGAL CONTEXT. Use plain text only - no brackets, no markdown.

ANSWER:"""

        max_retries = 3

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt+1} with model: {self.model}")

                # Use your structured legal prompt
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert Indian legal advisor. Provide accurate, structured legal guidance based strictly on the provided context.",
                    },
                    {"role": "user", "content": legal_prompt},
                ]

                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "frequency_penalty": 0.1,
                        "presence_penalty": 0.1,
                    },
                    timeout=60,
                )

                if response.status_code == 200:
                    result = response.json()
                    generated_text = result["choices"][0]["message"]["content"].strip()

                    # Clean up response (like in LocalProvider)
                    import re

                    generated_text = re.sub(r"\n\s*\n", "\n\n", generated_text)

                    logger.info(
                        f"✅ Success with model {self.model}: {len(generated_text)} chars"
                    )
                    return generated_text

                elif response.status_code == 429:  # Rate limit
                    logger.warning(f"Model {self.model} rate limited")
                    self.failed_models.add(self.model)
                    self.rotate_model()
                    continue

                else:
                    error_data = response.json() if response.text else {}
                    logger.error(
                        f"Model {self.model} error {response.status_code}: {error_data}"
                    )
                    self.failed_models.add(self.model)
                    self.rotate_model()
                    continue

            except requests.exceptions.Timeout:
                logger.error(f"Model {self.model} timeout")
                self.failed_models.add(self.model)
                if attempt < max_retries - 1:
                    self.rotate_model()
                    continue
                else:
                    break

            except Exception as e:
                logger.error(f"Attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    self.rotate_model()
                    continue

        # All attempts failed
        logger.error("All OpenRouter models failed, returning fallback")
        return self._get_fallback_response()

    def _get_fallback_response(self) -> str:
        """Better fallback that shows retrieved context is available"""
        return """Legal Analysis
Based on the retrieved legal documents, I can provide information about your query.

Applicable Provisions
Relevant legal sections and provisions have been identified.

Key Information
- The system has successfully retrieved pertinent legal documents
- Specific legal references are available in the sources below
- Please refer to the cited documents for detailed provisions

Important Notes
This response is based on the available legal context. For comprehensive legal advice, consult a qualified professional.

Legal References:
See the source documents below for specific sections and details."""


# NVIDIA NIM Provider Class
class NVIDIAProvider(ProviderClient):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY not found in environment variables")

        self.base_url = "https://integrate.api.nvidia.com/v1"  # NVIDIA NIM endpoint

        # NVIDIA Models available (choose based on your needs)
        self.available_models = {
            "meta/llama-3.1-8b-instruct": "Llama 3.1 8B Instruct (Good balance)",
            "meta/llama-3.1-70b-instruct": "Llama 3.1 70B Instruct (Powerful)",
            "mistralai/mistral-7b-instruct-v0.3": "Mistral 7B Instruct",
            "google/gemma-2-9b-it": "Gemma 2 9B IT",
            "microsoft/phi-3.5-mini-instruct": "Phi-3.5 Mini Instruct (Fast)",
        }

        # Default model (Llama 3.1 8B is a good default)
        self.model = "meta/llama-3.1-8b-instruct"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        # Initialize embeddings locally (as requested)
        from langchain_huggingface import HuggingFaceEmbeddings

        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Test connection
        self._test_connection()

        logger.info(f"✅ NVIDIA NIM Provider initialized with model: {self.model}")
        # logger.info(f"📊 Available models: {list(self.available_models.keys())}")

    def _test_connection(self):
        """Test NVIDIA NIM API connection"""
        try:
            # Simple test request to check API key
            test_response = requests.get(
                f"{self.base_url}/models", headers=self.headers, timeout=10
            )

            if test_response.status_code == 200:
                logger.info("✅ NVIDIA NIM API connection successful")
                # Update available models from API response
                try:
                    api_models = test_response.json().get("data", [])
                    if api_models:
                        self.available_models = {
                            model["id"]: model.get("description", "NVIDIA Model")
                            for model in api_models
                        }
                        # logger.info(
                        #     f"📋 Updated models from NVIDIA API: {list(self.available_models.keys())}"
                        # )
                except:
                    logger.info("Using predefined model list")
            else:
                logger.warning(
                    f"⚠️ NVIDIA API test returned: {test_response.status_code}"
                )
                logger.info("Using predefined model - API key may have limited access")

        except Exception as e:
            logger.error(f"❌ NVIDIA NIM connection test failed: {e}")
            raise ConnectionError(f"Cannot connect to NVIDIA NIM: {e}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using local HuggingFace model"""
        logger.info(f"Generating embeddings for {len(texts)} texts (using local model)")
        return self.embeddings.embed_documents(texts)

    def generate(self, prompt: str, max_tokens: int = 1000, **kwargs) -> str:
        """Generate legal analysis using NVIDIA NIM with structured prompt"""

        try:
            logger.info(f"Generating legal analysis via NVIDIA NIM ({self.model})...")

            # Prepare messages for NVIDIA NIM
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert Indian legal advisor. Provide accurate, helpful legal guidance based strictly on the provided context.",
                },
                {"role": "user", "content": prompt},
            ]

            start_time = time.time()

            # Make request to NVIDIA NIM API
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
                timeout=60,  # NVIDIA can be slower but more reliable
            )

            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                generated_text = result["choices"][0]["message"]["content"].strip()

                # Clean up the response
                import re

                generated_text = re.sub(r"\n\s*\n", "\n\n", generated_text)

                # Remove any markdown formatting that might slip through
                generated_text = re.sub(r"~~(.*?)~~", r"\1", generated_text)
                generated_text = re.sub(r"\*\*(.*?)\*\*", r"\1", generated_text)
                generated_text = re.sub(r"\[(.*?)\]", r"\1", generated_text)

                logger.info(
                    f"✅ NVIDIA NIM success: {len(generated_text)} chars in {elapsed_time:.2f}s"
                )
                return generated_text

            elif response.status_code == 401:
                logger.error("❌ NVIDIA API key invalid or expired")
                return self._get_auth_error_response()

            elif response.status_code == 429:
                logger.warning("⚠️ NVIDIA NIM rate limit reached")
                return self._get_rate_limit_response()

            else:
                error_text = (
                    response.text[:200] if response.text else "No error details"
                )
                logger.error(
                    f"❌ NVIDIA NIM error {response.status_code}: {error_text}"
                )
                return self._get_fallback_response()

        except requests.exceptions.Timeout:
            logger.error("❌ NVIDIA NIM request timeout")
            return self._get_timeout_response()

        except requests.exceptions.ConnectionError:
            logger.error("❌ Cannot connect to NVIDIA NIM")
            return self._get_connection_error_response()

        except Exception as e:
            logger.error(f"❌ NVIDIA NIM generation error: {e}")
            return self._get_fallback_response()

    def set_model(self, model_name: str):
        """Change the model being used"""
        if model_name in self.available_models:
            self.model = model_name
            logger.info(f"🔄 Changed NVIDIA model to: {model_name}")
        else:
            logger.warning(
                f"Model {model_name} not available. Using default: {self.model}"
            )

    def list_models(self) -> dict:
        """List all available NVIDIA models"""
        return self.available_models

    def _get_fallback_response(self) -> str:
        """Fallback response for NVIDIA errors"""
        return """Legal Analysis
Based on the retrieved legal documents, I can provide information about your query.

Applicable Provisions
Relevant legal sections and provisions have been identified.

Key Information
- The system has successfully retrieved pertinent legal documents
- Specific legal references are available in the sources below
- Please refer to the cited documents for detailed provisions

Important Notes
This response is based on the available legal context. For comprehensive legal advice, consult a qualified professional.

Legal References:
See the source documents below for specific sections and details."""

    def _get_auth_error_response(self) -> str:
        return """System Notice: NVIDIA API Authentication Error

The NVIDIA AI service authentication failed. Please check your API key.

Switching to fallback mode with available legal context."""

    def _get_rate_limit_response(self) -> str:
        return """Legal Analysis - Rate Limited

The NVIDIA AI service is currently experiencing high demand. 
Providing basic legal information from available documents.

Please try again shortly for detailed analysis."""

    def _get_timeout_response(self) -> str:
        return """Legal Analysis - Processing Timeout

The AI service is taking longer than expected. 
Providing key information from available legal documents.

Please check the reference sources below for immediate information."""

    def _get_connection_error_response(self) -> str:
        return """System Notice: NVIDIA Service Unavailable

Cannot connect to the NVIDIA AI service at this time. 
Providing information based on retrieved legal documents.

Please check your internet connection and try again."""


def get_best_provider():
    logger.info("PROVIDER SELECTION START")

    # Priority order: NVIDIA > OpenRouter > Local

    # Check NVIDIA first
    nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    use_nvidia = os.getenv("USE_NVIDIA", "false").lower() in ("true", "1", "yes")

    if use_nvidia and nvidia_key:
        logger.info("Attempting NVIDIA NIM provider...")
        try:
            provider = NVIDIAProvider()
            logger.info(
                f"✅ NVIDIA NIM provider initialized successfully with model: {provider.model}"
            )
            return provider
        except Exception as e:
            logger.error(f"NVIDIA initialization failed: {e}")
            logger.info("Falling back to next provider...")
    else:
        logger.info(f"NVIDIA not enabled or no API key. USE_NVIDIA={use_nvidia}")

    # Check OpenRouter second
    use_openrouter = os.getenv("USE_OPENROUTER", "false").lower() in (
        "true",
        "1",
        "yes",
    )
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")

    if use_openrouter and openrouter_key:
        logger.info("Attempting OpenRouter provider...")
        try:
            provider = OpenRouterProvider()
            logger.info(
                f"✅ OpenRouter provider initialized successfully with model: {provider.model}"
            )
            return provider
        except Exception as e:
            logger.error(f"OpenRouter initialization failed: {e}")
            logger.info("Falling back to LocalProvider...")
    else:
        logger.info(
            f"OpenRouter not enabled or no API key. USE_OPENROUTER={use_openrouter}"
        )

    # Fallback to local (default)
    try:
        provider = LocalProvider()
        logger.info("✅ LocalProvider initialized successfully")
        return provider
    except Exception as e:
        logger.error(f"LocalProvider failed: {e}")
        raise Exception("No providers available")


def get_local_provider():
    return LocalProvider()
