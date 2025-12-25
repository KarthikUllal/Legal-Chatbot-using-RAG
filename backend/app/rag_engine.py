# backend/app/rag_engine.py
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import logging
from .config import settings
from .provider_client import get_best_provider, ProviderClient
from .vector_store import VectorStore
from .ingestion import split_into_chunks, load_pdf_text
from .schemas import ChatResponse, SourceItem
from .translation import translator
import re
from datetime import datetime



try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.retrievers import BaseRetriever
except ImportError:
    from langchain.prompts import PromptTemplate
    from langchain.schema.runnable import RunnablePassthrough

logger = logging.getLogger(__name__)


# Main RAG engine class - handles document ingestion, retrieval, and generation
class RAGEngine:
    def __init__(self, provider: ProviderClient = None):
        self.provider = provider or get_best_provider()
        self.vstore = VectorStore()
        self.embed_batch = settings.BATCH_SIZE
        self.conversation_memory = {}  # session_id -> list of exchanges
        self.max_memory_per_session = 8  # Store last 8 exchanges

        self.conversation_transcripts = {}  # session_id -> full lawyer-style transcript



        self._setup_prompts()
        self._setup_chain()

        logger.info(
            f"RAG Engine initialized with provider: {type(self.provider).__name__}"
        )
    

    def _setup_prompts(self):
        self.context_prompt = PromptTemplate(
            input_variables=["documents"],
            template="""Legal Documents Context:
{documents}

Use the above legal documents to answer the question accurately.""",
        )

        self.qa_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""You are "Nyaya Mitra" - India's legal expert.

User Query: "{question}"
Available Legal Context: {context}

**Response Decision:**
1. If query is purely casual (greetings, thanks, farewells): Respond with simple, consistent friendly response.
2. If query contains legal content: Use the context to provide helpful legal information.
3. Avoid giving "Response :" label while answering  

**Casual Response Examples (be consistent):**
- "hello" → "Hello! How can I help?"
- "thanks" → "You're welcome!"
- "thank you" → "You're welcome!"

**Legal Response Approach:**
- Use relevant laws from context
-And when explaining about procedure to file complaint , please give it in step by step manner.
- Be practical and helpful
- Mention BNS/IPC when applicable
- BNS vs IPC: When discussing laws, mention if information comes from BNS (new law) or IPC (old law).

**Now respond to this query appropriately and consistently:**""",
)

    def _setup_chain(self):
        pass

    def _clean_response(self, response: str) -> str:
        response = re.sub(r"\n\s*\n", "\n\n", response)

        sections = [
            "Overview",
            "Key Definitions",
            "Legal Provisions",
            "Punishments & Penalties",
            "Important Points",
            "Legal References:",
        ]

        for section in sections:
            if section in response and not response.startswith(section):
                response = response.replace(section, f"\n\n{section}")

        return response.strip()

    def _format_context(
        self, docs: List[str], metas: List[dict], ids: List[str], dists: List[float]
    ) -> str:
        try:
            document_parts = []
            for i, (doc, meta, doc_id, distance) in enumerate(
                zip(docs, metas, ids, dists)
            ):
                act_name = meta.get("act", "Unknown Act")
                relevance_score = 1.0 - (distance / 2.0)
                document_parts.append(
                    f"[Document {i+1} | Source: {act_name} | Relevance: {relevance_score:.2f}]\n"
                    f"{doc[:800]}..."
                )

            documents_text = "\n\n".join(document_parts)

            formatted_context = self.context_prompt.format(documents=documents_text)
            return formatted_context

        except Exception as e:
            logger.error(f"Error formatting context: {e}")
            return "\n\n".join(
                [f"[Source {i+1}]: {doc[:600]}..." for i, doc in enumerate(docs)]
            )

    def ingest_text(self, doc_id: str, text: str, metadata: dict = None) -> bool:
        try:
            logger.info(f"Ingesting document: {doc_id}")

            chunks = split_into_chunks(text)
            logger.info(f"Split into {len(chunks)} chunks")

            ids = [f"{doc_id}__{i}" for i in range(len(chunks))]
            metadatas = [
                {**(metadata or {}), "chunk_index": i, "doc_id": doc_id}
                for i in range(len(chunks))
            ]

            embeddings = []
            for i in range(0, len(chunks), self.embed_batch):
                batch = chunks[i : i + self.embed_batch]
                logger.info(
                    f"Generating embeddings for batch {i//self.embed_batch + 1}"
                )
                embs = self.provider.get_embeddings(batch)
                embeddings.extend(embs)

            self.vstore.collection.add(
                ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas
            )

            logger.info(f"Successfully ingested {len(chunks)} chunks from {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to ingest {doc_id}: {e}")
            return False

    def ingest_file(
        self, file_path: str, doc_id: str = None, act_name: str = None
    ) -> bool:
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            doc_id = doc_id or path.stem
            logger.info(f"Processing file: {path.name} as {doc_id}")

            text = load_pdf_text(path)
            if not text:
                raise ValueError(
                    "No text extracted from PDF. File might be scanned or corrupted."
                )

            metadata = {
                "act": act_name or doc_id,
                "source_file": path.name,
                "source_type": "pdf",
            }

            return self.ingest_text(doc_id, text, metadata)

        except Exception as e:
            logger.error(f"Failed to ingest file {file_path}: {e}")
            return False

    def retrieve(self, query: str, k: int = 6) -> Dict:
        try:
            logger.info(f"Retrieving for: '{query}'")

            # ✅ LAW CODE DETECTION
            requested_law = None
            query_lower = query.lower()

            if "bharatiya nyaya" in query_lower or "bns" in query_lower:
                requested_law = "BNS"
            elif "bharatiya nagarik" in query_lower or "bnss" in query_lower:
                requested_law = "BNSS"
            elif "indian penal code" in query_lower or "ipc" in query_lower:
                requested_law = "IPC"
            elif "bharatiya sakshya" in query_lower or "bsa" in query_lower:
                requested_law = "BSA"

            legal_keywords = [
                "section",
                "act",
                "law",
                "legal",
                "right",
                "remedy",
                "punishment",
                "penalty",
                "offense",
                "crime",
                "consumer",
                "protection",
                "domestic",
                "violence",
                "contract",
                "property",
            ]

            enhanced_query = query
            if not any(keyword in query.lower() for keyword in legal_keywords):
                enhanced_query += " legal law section act rights remedies"

            query_embedding = self.provider.get_embeddings([enhanced_query])[0]
            results = self.vstore.query(query_embedding, n_results=k * 2)

            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            ids = results.get("ids", [[]])[0]

            print(
                f"Initial retrieval: {len(docs)} documents, Requested law: {requested_law}"
            )

            if not docs:
                return {
                    "documents": [[]],
                    "metadatas": [[]],
                    "distances": [[]],
                    "ids": [[]],
                }

            filtered_docs = []
            filtered_metas = []
            filtered_dists = []
            filtered_ids = []

            for i, (doc, meta, distance, chunk_id) in enumerate(
                zip(docs, metas, dists, ids)
            ):
                if doc and meta:
                    # ✅ PRIORITIZE REQUESTED LAW
                    if requested_law:
                        doc_act = meta.get("act", "").upper()
                        doc_text_lower = doc.lower()

                        law_matches = (
                            requested_law in doc_act
                            or requested_law.lower() in doc_text_lower
                        )

                        if law_matches:
                            filtered_docs.append(doc)
                            filtered_metas.append(meta)
                            filtered_dists.append(distance)
                            filtered_ids.append(chunk_id)  # ✅ FIX 1: Use chunk_id
                            continue

                    # Original filtering
                    doc_lower = doc.lower()
                    query_lower = query.lower()

                    has_query_terms = any(
                        term in doc_lower for term in query_lower.split()
                    )
                    has_legal_content = any(
                        keyword in doc_lower for keyword in legal_keywords
                    )
                    is_relevant_distance = distance < 1.0

                    if has_query_terms or (has_legal_content and is_relevant_distance):
                        filtered_docs.append(doc)
                        filtered_metas.append(meta)
                        filtered_dists.append(distance)
                        filtered_ids.append(chunk_id)  # ✅ FIX 1: Use chunk_id

            if not filtered_docs and docs:
                filtered_docs = docs[:k]
                filtered_metas = metas[:k]
                filtered_dists = dists[:k]
                filtered_ids = ids[:k]  # ✅ FIX 2: Use ids, not metadata

            print(f"Final filtered: {len(filtered_docs)} documents")

            return {
                "documents": [filtered_docs[:k]],
                "metadatas": [filtered_metas[:k]],
                "distances": [filtered_dists[:k]],
                "ids": [filtered_ids[:k]],
            }

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "ids": [[]],
            }

    def generate_answer(
        self, question: str, retrieved: Dict
    ) -> Tuple[str, List[SourceItem]]:
        try:
            docs = retrieved.get("documents", [[]])[0]
            metas = retrieved.get("metadatas", [[]])[0]
            ids = retrieved.get("ids", [[]])[0]
            dists = retrieved.get("distances", [[]])[0]

            if not docs:
                return (
                    "I couldn't find any relevant legal documents...",
                    [],
                )

            context = self._format_context(docs, metas, ids, dists)

            formatted_prompt = self.qa_prompt.format(
                question=question, context=context  # Just use original context
            )

            answer = self.provider.generate(
                formatted_prompt,
                max_tokens=1000,
            )
            answer = self._clean_response(answer)
            sources = []

            logger.info(f"Generated answer with NO sources (by design)")
            return answer, sources  # Always empty list

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {str(e)}", []  # Empty sources

    def query(
        self, question: str, top_k: int = 4, session_id: str = "default"
    ) -> ChatResponse:
        try:
            # Initialize session memory
            if session_id not in self.conversation_memory:
                self.conversation_memory[session_id] = []

            # Get session memory
            session_memory = self.conversation_memory[session_id]

            # Check if question references previous answer
            enhanced_question = self._enhance_question_with_context(
                question, session_memory
            )

            retrieved = self.retrieve(enhanced_question, k=top_k)
            answer, sources = self.generate_answer(enhanced_question, retrieved)

            # Store in memory
            session_memory.append({"question": question, "answer": answer[:400]})



            # Keep only last 8 exchanges
            if len(session_memory) > self.max_memory_per_session:
                self.conversation_memory[session_id] = session_memory[
                    -self.max_memory_per_session :
                ]

            # FULL transcript (for download)
            self.conversation_transcripts.setdefault(session_id, []).append({
                "timestamp": datetime.now().isoformat(),
                "user_query": question,
                "legal_response": answer
            })

            return ChatResponse(answer=answer, sources=sources)

        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return ChatResponse(
                answer=f"Sorry, I encountered an error: {str(e)}",
                sources=[],
            )
        
    def get_full_transcript(self, session_id: str):
        return self.conversation_transcripts.get(session_id, [])
    
    

    def _get_session_memory(self, session_id: str) -> list:
        """Get or create session memory"""
        if session_id not in self.conversation_memory:
            self.conversation_memory[session_id] = []
        return self.conversation_memory[session_id]

    def _enhance_question_with_context(
        self, question: str, session_memory: list
    ) -> str:
        """Enhance question with conversation context if it's a follow-up"""
        if not session_memory:
            return question

        question_lower = question.lower()

        # Check if this is a follow-up about previously mentioned sections
        for exchange in reversed(session_memory):  # Check most recent first
            prev_answer_lower = exchange["answer"].lower()

            # Look for section references in previous answer
            import re

            section_matches = re.findall(r"section\s+(\d+[a-z]*)", prev_answer_lower)

            # Check if current question asks about those sections
            for section in section_matches:
                if f"section {section}" in question_lower or section in question_lower:
                    return f"""Previous context mentioned Section {section.upper()}. 
                    Current question: {question}
                    
                    Note: If Section {section.upper()} was mentioned earlier but isn't in the legal documents, 
                    I might not have detailed information about it."""

        # Check for pronoun references (this, that, it)
        if any(
            pronoun in question_lower for pronoun in ["this", "that", "it", "he", "she"]
        ):
            last_exchange = session_memory[-1]
            return f"""Following up on previous question about: {last_exchange['question'][:100]}...
            Current question: {question}"""

        return question

    # function to save Conversation
    def save_conversation(self, session_id: str, user_id: str = "anonymous"):
        """Save completed conversation"""
        if session_id in self.conversation_memory:
            conversation = self.conversation_memory[session_id]

            if user_id not in self.conversation_history:
                self.conversation_history[user_id] = []

            self.conversation_history[user_id].append(
                {
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "exchanges": conversation,
                    "preview": (
                        conversation[0]["question"][:100] if conversation else ""
                    ),
                }
            )

    # Provides stats about Working
    def get_stats(self) -> Dict:
        try:
            total_chunks = self.vstore.collection.count()

            results = self.vstore.collection.get(include=["metadatas"])
            unique_docs = set()

            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    if metadata and "doc_id" in metadata:
                        unique_docs.add(metadata["doc_id"])

            total_documents = len(unique_docs)

            return {
                "total_documents": total_documents,
                "total_chunks": total_chunks,
                "provider": type(self.provider).__name__,
                "collection": self.vstore.col_name,
                "embedding_model": getattr(self.provider, "model_name", "Unknown"),
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}

    def clear_collection(self) -> bool:
        try:
            self.vstore.client.delete_collection(self.vstore.col_name)
            self.vstore.collection = self.vstore.client.create_collection(
                name=self.vstore.col_name, metadata={"source": "legal_docs"}
            )
            logger.info("Collection cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    def get_prompt_templates(self) -> Dict:
        return {
            "qa_prompt": self.qa_prompt.template[:100] + "...",
            "context_prompt": self.context_prompt.template[:100] + "...",
            "input_variables": {
                "qa_prompt": self.qa_prompt.input_variables,
                "context_prompt": self.context_prompt.input_variables,
            },
        }

    def health_check(self) -> Dict:
        try:
            doc_count = self.vstore.collection.count()

            provider_status = "healthy"
            try:
                test_embeddings = self.provider.get_embeddings(["test"])
                if not test_embeddings or len(test_embeddings) == 0:
                    provider_status = "unhealthy"
            except Exception as e:
                provider_status = f"unhealthy: {str(e)}"

            return {
                "status": "healthy",
                "vector_store_documents": doc_count,
                "provider_status": provider_status,
                "provider_name": type(self.provider).__name__,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def query_with_language(
        self,
        question: str,
        language: str = "en",
        top_k: int = 4,
        session_id: str = "default",
    ):
        try:
            logger.info(f"Processing question in {language}: {question}")

            processed_question = question
            if language != "en":
                try:
                    processed_question = translator.translate_legal_response(
                        question, "en"
                    )
                    logger.info(f"Translated question to English: {processed_question}")
                except Exception as trans_error:
                    logger.warning(
                        f"Question translation failed, using original: {trans_error}"
                    )

            response = self.query(
                question=processed_question, top_k=top_k, session_id=session_id
            )

            if language != "en":
                try:
                    translated_answer = translator.translate_legal_response(
                        response.answer, language
                    )

                    return ChatResponse(
                        answer=translated_answer, sources=response.sources
                    )

                except Exception as translation_error:
                    logger.error(f"Answer translation failed: {translation_error}")
                    return response
            else:
                return response

        except Exception as e:
            logger.error(f"Language query failed: {e}")
            error_msg = "Sorry, I encountered an error processing your question."
            if language != "en":
                try:
                    error_msg = translator.translate_legal_response(error_msg, language)
                except:
                    pass
            return ChatResponse(answer=error_msg, sources=[])

# Factory function to get configured RAG engine instance
def get_rag_engine(provider: ProviderClient = None) -> RAGEngine:
    return RAGEngine(provider)