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
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(model_name=cfg_name)
        except ImportError:
            try:
                from langchain.embeddings import HuggingFaceEmbeddings
                self.embeddings = HuggingFaceEmbeddings(model_name=cfg_name)
            except ImportError:
                raise ImportError("Install: pip install langchain sentence-transformers")

        self.model_name = cfg_name
        self.llm_model = "llama3.2:3b"
        self.ollama_base_url = "http://localhost:11434"

        logger.info(f"Loading local model with LangChain: {self.model_name}")
        logger.info(f"Ollama LLM model: {self.llm_model}")
        logger.info("Local model loaded successfully with LangChain!")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Generating embeddings for {len(texts)} texts using LangChain")
        embs = self.embeddings.embed_documents(texts)
        logger.info(f"Generated {len(embs)} embeddings with {len(embs[0])} dimensions")
        return embs

    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        try:
            import requests
            import re

            enhanced_prompt = f"""You are a legal expert specializing in Indian law. Provide a CLEAN, STRUCTURED answer based ONLY on the context.

CONTEXT: {prompt}

RESPONSE FORMATTING RULES - FOLLOW EXACTLY:
1. STRUCTURE:
- Start with 1-2 sentence overview
- Use **bold headings** for main sections
- Use • bullet points for lists (not * or -)
- Keep paragraphs short (2-3 lines maximum)
- End with "**Legal References:**" section

2. CONTENT:
- Use ONLY information from provided context
- Cite specific sections (Section 302, Section 304, etc.)
- Be precise and factual
- If context is insufficient, state this clearly

3. FORMAT EXAMPLE:
**Overview**
Brief introduction here.

**Key Definitions**
• Definition 1 with section reference
• Definition 2 with section reference

**Main Differences**
• Difference 1
• Difference 2

**Legal Provisions**
• Provision 1
• Provision 2

**Legal References:**
Sections XXX, YYY of Relevant Act

STRICTLY FOLLOW THIS FORMAT. DO NOT USE MARKDOWN TABLES OR COMPLEX FORMATTING.

ANSWER:"""

            response = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": enhanced_prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.2,
                        "top_p": 0.8,
                        "repeat_penalty": 1.2,
                    },
                },
                timeout=120,
            )

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "").strip()

                if generated_text:
                    generated_text = re.sub(r'\n\s*\n', '\n\n', generated_text)
                    if "**Legal References:**" not in generated_text and "**Reference Sources:**" not in generated_text:
                        generated_text += "\n\n**Legal References:** Relevant legal provisions cited above"
                    
                    logger.info("✅ Successfully generated structured legal response")
                    return generated_text.strip()
                else:
                    return "No response generated from the local LLM."

            else:
                error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return f"Local LLM unavailable. Error: {response.status_code}"

        except requests.exceptions.ConnectionError:
            error_msg = "Cannot connect to Ollama. Make sure Ollama is running: 'ollama serve'"
            logger.error(error_msg)
            return f"[LOCAL LLM OFFLINE] {error_msg}"

        except requests.exceptions.Timeout:
            error_msg = "Ollama request timed out. The model might be processing."
            logger.error(error_msg)
            return f"[LOCAL LLM TIMEOUT] {error_msg}"

        except Exception as e:
            error_msg = f"Unexpected error with local LLM: {str(e)}"
            logger.error(error_msg)
            return f"[LOCAL LLM ERROR] {error_msg}"


class DeepSeekClient(ProviderClient):
    def __init__(
        self,
        api_key: Optional[str] = None,
        emb_model: Optional[str] = None,
        gen_model: Optional[str] = None,
    ):
        print("🚀 DEBUG: DeepSeekClient initialization starting...")
        print(f"🔑 DEBUG: API Key provided: {'***' + api_key[-4:] if api_key else 'NONE'}")
        
        try:
            from openai import OpenAI
            print("✅ DEBUG: OpenAI import successful")
        except ImportError as e:
            print(f"❌ DEBUG: OpenAI import failed: {e}")
            raise

        try:
            self.client = OpenAI(
                api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com/v1",
            )
            print("✅ DEBUG: DeepSeek client created successfully")
        except Exception as e:
            print(f"❌ DEBUG: DeepSeek client creation failed: {e}")
            raise
        
        self.gen_model = gen_model or "deepseek-chat"
        print(f"🎯 DEBUG: Using model: {self.gen_model}")
        
        # Use local embeddings since DeepSeek doesn't provide embedding API
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            print("✅ DEBUG: Local embeddings loaded successfully")
        except ImportError as e:
            print(f"❌ DEBUG: Embeddings import failed: {e}")
            raise

        print("🎉 DEBUG: DeepSeekClient initialized successfully!")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Generating embeddings for {len(texts)} texts using local model")
        embs = self.embeddings.embed_documents(texts)
        logger.info(f"Generated {len(embs)} embeddings with {len(embs[0])} dimensions")
        return embs

    def generate(self, prompt: str, max_tokens: int = 512, **kwargs) -> str:
        try:
            legal_prompt = f"""You are a legal expert specializing in Indian law. Provide accurate, structured information based on the context.

Context: {prompt}

Response Requirements:
1. Structure your answer with clear sections
2. Use bullet points (•) for lists
3. Cite specific legal sections when possible
4. Be precise and factual
5. If context is insufficient, state this clearly

Format:
- Start with a brief overview
- Key Definitions (bullet points)
- Legal Provisions (bullet points)
- Important Points (bullet points)
- Legal References (sections/acts used)

Answer:"""

            print("🔄 DEBUG: Sending request to DeepSeek API...")
            response = self.client.chat.completions.create(
                model=self.gen_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a legal expert specializing in Indian law. Provide accurate, structured legal information."
                    },
                    {
                        "role": "user",
                        "content": legal_prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.9,
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            if "**Legal References:**" not in generated_text:
                generated_text += "\n\n**Legal References:** Relevant legal provisions cited above"
            
            print("✅ DEBUG: DeepSeek generation successful!")
            logger.info("✅ Successfully generated response with DeepSeek")
            return generated_text
            
        except Exception as e:
            logger.error(f"DeepSeek generation failed: {e}")
            print(f"❌ DEBUG: DeepSeek generation error: {e}")
            return f"[DEEPSEEK ERROR] Failed to generate response: {str(e)}"


# Other provider classes (Gemini, OpenAI, HuggingFace) remain the same...
class GeminiClient(ProviderClient):
    def __init__(self, api_key: Optional[str] = None, emb_model: Optional[str] = None, gen_model: Optional[str] = None):
        try:
            import google.generativeai as genai
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
            except ImportError:
                from langchain.embeddings import GoogleGenerativeAIEmbeddings
        except ImportError:
            raise ImportError("Install: pip install google-generativeai langchain-google-genai")

        self.genai = genai
        self.api_key = api_key or getattr(settings, "GEMINI_API_KEY", None) or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key required")

        self.genai.configure(api_key=self.api_key)
        self.emb_model = emb_model or getattr(settings, "EMBED_MODEL", "models/embedding-001")
        self.gen_model = gen_model or getattr(settings, "GEN_MODEL", "gemini-pro")
        self.embeddings = GoogleGenerativeAIEmbeddings(model=self.emb_model, google_api_key=self.api_key)

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
    def __init__(self, api_key: Optional[str] = None, emb_model: Optional[str] = None, gen_model: Optional[str] = None):
        try:
            from openai import OpenAI
            try:
                from langchain_openai import OpenAIEmbeddings
            except ImportError:
                from langchain.embeddings import OpenAIEmbeddings
        except ImportError:
            raise ImportError("Install: pip install openai langchain-openai")

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.emb_model = emb_model or "text-embedding-3-small"
        self.gen_model = gen_model or "gpt-3.5-turbo"
        self.embeddings = OpenAIEmbeddings(model=self.emb_model, openai_api_key=api_key or os.getenv("OPENAI_API_KEY"))

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
    def __init__(self, api_key: Optional[str] = None, emb_model: Optional[str] = None, gen_model: Optional[str] = None):
        try:
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


def get_best_provider():
    """Get provider - prioritize DeepSeek if available"""
    print("🚀 ======= PROVIDER SELECTION START =======")
    
    # ✅ FIXED: Check for DeepSeek FIRST
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    print(f"🔑 DEBUG: DeepSeek API Key from env: {'***' + deepseek_key[-4:] if deepseek_key else 'NOT FOUND'}")
    
    if deepseek_key:
        print("🎯 DEBUG: Attempting DeepSeek...")
        try:
            provider = DeepSeekClient(api_key=deepseek_key)
            print("✅ DEBUG: DeepSeek SUCCESS - Using DeepSeek Provider")
            print("🚀 ======= PROVIDER SELECTION END =======")
            return provider
        except Exception as e:
            print(f"❌ DEBUG: DeepSeek FAILED: {e}")
            print("🔄 DEBUG: Falling back to other providers...")
    else:
        print("❌ DEBUG: No DeepSeek API key found")

    # ✅ FIXED: Check force_local AFTER DeepSeek
    force_local = os.getenv("FORCE_LOCAL", "false").lower() in ("true", "1", "yes")
    print(f"🔧 DEBUG: FORCE_LOCAL = {force_local}")
    
    if force_local:
        provider = LocalProvider()
        print("🤖 DEBUG: Using LocalProvider (forced for testing)")
        print("🚀 ======= PROVIDER SELECTION END =======")
        return provider

    # If user specifically wants other providers
    preferred_provider = os.getenv("PREFERRED_PROVIDER", "").lower()
    print(f"🎯 DEBUG: PREFERRED_PROVIDER = {preferred_provider}")

    if preferred_provider == "openai" and os.getenv("OPENAI_API_KEY"):
        provider = OpenAIClient()
        print("🤖 DEBUG: Using OpenAI Provider")
        print("🚀 ======= PROVIDER SELECTION END =======")
        return provider
    elif preferred_provider == "gemini" and os.getenv("GEMINI_API_KEY"):
        provider = GeminiClient()
        print("🤖 DEBUG: Using Gemini Provider")
        print("🚀 ======= PROVIDER SELECTION END =======")
        return provider
    elif preferred_provider == "huggingface" and os.getenv("HUGGINGFACE_TOKEN"):
        provider = HuggingFaceClient()
        print("🤖 DEBUG: Using HuggingFace Provider")
        print("🚀 ======= PROVIDER SELECTION END =======")
        return provider

    # Default to local provider
    provider = LocalProvider()
    print("🤖 DEBUG: Using LocalProvider (default fallback)")
    print("🚀 ======= PROVIDER SELECTION END =======")
    return provider


def get_local_provider():
    return LocalProvider()