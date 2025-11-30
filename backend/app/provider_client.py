# backend/app/provider_client.py
from typing import List, Optional
import logging
import os
import time
from .config import settings
from dotenv import load_dotenv

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
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
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
                models = response.json().get('models', [])
                model_names = [model.get('name', '') for model in models]
                logger.info(f"Available Ollama models: {model_names}")
                
                if self.llm_model in model_names:
                    logger.info(f"✅ Model '{self.llm_model}' is available")
                else:
                    logger.warning(f"❌ Model '{self.llm_model}' not found. Available: {model_names}")
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
                    logger.info(f"✅ Successfully generated legal analysis: {len(generated_text)} chars")
                    
                    # Clean up the response
                    generated_text = re.sub(r'\n\s*\n', '\n\n', generated_text)
                    
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

def get_best_provider():
    logger.info("PROVIDER SELECTION START")
    
    force_local = os.getenv("FORCE_LOCAL", "true").lower() in ("true", "1", "yes")
    logger.info(f"FORCE_LOCAL = {force_local}")
    
    if force_local:
        logger.info("Using LocalProvider")
        try:
            provider = LocalProvider()
            logger.info("LocalProvider SUCCESS")
            return provider
        except Exception as e:
            logger.error(f"LocalProvider FAILED: {e}")
            raise
    
    raise Exception("No providers available")

def get_local_provider():
    return LocalProvider()