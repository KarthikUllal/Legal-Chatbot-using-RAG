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

        self._setup_prompts()
        self._setup_chain()

        logger.info(
            f"RAG Engine initialized with provider: {type(self.provider).__name__}"
        )

        # older prompt template ... if changed one fails then this will be replaced

    #     def _setup_prompts(self):
    #         self.context_prompt = PromptTemplate(
    #             input_variables=["documents"],
    #             template="""Legal Documents Context:
    # {documents}

    # Use the above legal documents to answer the question accurately.""",
    #         )

    #         self.qa_prompt = PromptTemplate(
    #             input_variables=["question", "context"],
    #             template="""You are an expert Indian legal advisor. Analyze the user's situation and provide practical legal guidance based on the context.

    # USER'S SITUATION: {question}

    # RELEVANT LEGAL CONTEXT:
    # {context}

    # IMPORTANT - FOLLOW THIS STRUCTURE EXACTLY:

    # Situation Analysis
    # Briefly summarize the legal situation
    # Identify key legal issues involved

    # Applicable Laws & Sections
    # List relevant legal provisions with section numbers
    # Explain how each law applies to this situation

    # Legal Rights & Remedies
    # What legal rights does the person have?
    # Available legal remedies and procedures

    # Recommended Actions
    # Step-by-step practical advice
    # Timeline for actions
    # Required documents/evidence

    # Potential Outcomes
    # Best-case and worst-case scenarios
    # Typical resolution timeframes

    # Important Caveats
    # Limitations of this advice
    # When to consult a practicing lawyer

    # Legal References:
    # Sections XXX, YYY of Relevant Act

    # BASE YOUR ANSWER STRICTLY ON THE PROVIDED LEGAL CONTEXT. If the context doesn't cover specific situational aspects, acknowledge this limitation.

    # ANSWER:""",
    #         )
    def _setup_prompts(self):
        self.qa_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""You are "Nyaya Mitra" - India's legal expert.

    User Query: {question}

    Legal Information: {context}

    RESPONSE RULES:

    A. FOR CASUAL MESSAGES (hi, hello, thanks, bye, how are you, etc.):
    - Respond DIRECTLY without any prefix or analysis
    - Simple, friendly response only
    - NO "User Query:" or "Response:" labels
    - Examples:
        • "hi" → "Hello! 😊"
        • "thank you" → "You're welcome! 😊"
        • "how are you" → "I'm good, thanks! How can I help?"

    B. FOR LEGAL QUERIES (everything else):
    Provide information in a CONVERSATIONAL, helpful manner:

    **First, let me explain the relevant legal sections:**
    - [Based on the context, mention the specific laws and sections that apply to this query]

    **Now, about your specific situation:**
    - [Using the context, explain what this means for the user's case in simple terms]

    **Here are the steps you should take:**
    1. [First practical step based on the legal context]
    2. [Second practical step based on the legal context]
    3. [Third practical step based on the legal context]

    **Important details to remember:**
    - Time limit: [If applicable, based on context]
    - Where to file: [Specific authority/office based on context]

    **Additional advice:**
    - [Any extra tips or warnings based on the legal context]

    EXAMPLES:

    === CASUAL (Direct Only) ===
    User: "hi"
    You: "Namaste! 🙏 How can I assist you with legal information today?"

    User: "thanks"
    You: "Happy to help! Let me know if you need anything else. 😊"

    === LEGAL - EXAMPLE 1 ===
    User: "what is section 420?"
    You: "**First, let me explain the relevant legal sections:**

    This refers to Section 420 of the Indian Penal Code (IPC).

    **Now, about your specific situation:**
    Section 420 deals with cheating and dishonestly inducing delivery of property. It's a serious criminal offense.

    **Here are the steps you should take if you're a victim:**
    1. Gather all evidence - documents, communications, transaction proofs
    2. File an FIR at the nearest police station with full details
    3. Cooperate with the police investigation
    4. The case will be tried in court

    **Important details to remember:**
    - Time limit: No specific limit, but file as soon as possible
    - Where to file: Local police station

    **Additional advice:**
    Keep copies of all documents and maintain a record of all communications related to the case."

    === LEGAL - EXAMPLE 2 ===
    User: "how to file a complaint?"
    You: "**First, let me explain the relevant legal sections:**

    The complaint procedure depends on the type of case, but generally falls under the relevant specific act and its procedural rules.

    **Now, about your specific situation:**
    Filing a complaint requires following proper legal procedure to ensure it's accepted and processed.

    **Here are the steps you should take:**
    1. Identify the correct authority/jurisdiction for your complaint
    2. Draft a clear complaint with all facts, dates, and evidence
    3. Submit it in the prescribed format with required fees
    4. Follow up regularly on the status

    **Important details to remember:**
    - Time limit: Varies by case type (check specific law)
    - Where to file: Depends on the nature of the complaint

    **Additional advice:**
    Consult a lawyer if unsure about the correct procedure, as improperly filed complaints may get rejected."

    === LEGAL - EXAMPLE 3 ===
    User: "what are my rights?"
    You: "**First, let me explain the relevant legal sections:**

    Your rights depend on the specific situation, but are generally protected under various Indian laws.

    **Now, about your specific situation:**
    Indian law provides protection for citizens in various situations including consumer rights, civil rights, and criminal justice.

    **Here are the steps you should take:**
    1. Identify which law applies to your situation
    2. Document the violation of your rights
    3. Approach the appropriate legal authority
    4. Seek legal aid if needed

    **Important details to remember:**
    - Time limit: Varies depending on the right being violated
    - Where to seek help: Depends on the specific rights violation

    **Additional advice:**
    You can contact legal aid clinics or NGOs that specialize in the relevant area for free guidance."

    CRITICAL INSTRUCTION:
    - For casual messages: Respond directly and friendly
    - For legal queries: Use the conversational flow above
    - Base ALL information on the provided {context} - never make up laws or sections
    - If {context} doesn't contain information, say "I couldn't find specific information about this in my legal database"
    - Make it sound like you're explaining to a friend, not reciting a legal document
    - Use bullet points only for listing sections or steps
    - Keep language simple and practical
    - NEVER show "User Query:" or "Response:" in your answer
    - Always connect back to the user's specific situation mentioned in {question}

    Now respond appropriately:""",
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

    def generate_answer(self, question: str, retrieved: Dict) -> Tuple[str, List[SourceItem]]:
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
                question=question, 
                context=context  # Just use original context
            )

            answer = self.provider.generate(
                formatted_prompt,
                max_tokens=1000,
            )
            answer = self._clean_response(answer)

            # if "Reference Sources:" not in answer and "Sources:" not in answer:
            #     answer += "\n\nReference Sources: See cited legal documents below"

            sources = []
            # for i, (doc_id, meta, doc_text) in enumerate(zip(ids, metas, docs)):
            #     sources.append(
            #         SourceItem(
            #             id=doc_id,
            #             source=meta,
            #             snippet=(
            #                 doc_text[:300] + "..." if len(doc_text) > 300 else doc_text
            #             ),
            #         )
            #     )

            logger.info(f"Generated answer with NO sources (by design)")
            return answer, sources  # Always empty list


        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {str(e)}", []  # Empty sources
        


    # def query(self, question: str, top_k: int = 4) -> ChatResponse:
    #     try:
    #         retrieved = self.retrieve(question, k=top_k)

    #         answer, sources = self.generate_answer(question, retrieved)

    #         return ChatResponse(answer=answer, sources=sources)

    #     except Exception as e:
    #         logger.error(f"RAG query failed: {e}")
    #         return ChatResponse(
    #             answer=f"Sorry, I encountered an error while processing your question: {str(e)}",
    #             sources=[],
    #         )


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

            return ChatResponse(answer=answer, sources=sources)

            return ChatResponse(answer=answer, sources=sources)

        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return ChatResponse(
                answer=f"Sorry, I encountered an error: {str(e)}",
                sources=[],
            )

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
