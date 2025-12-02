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

try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.retrievers import BaseRetriever
except ImportError:
    from langchain.prompts import PromptTemplate
    from langchain.schema.runnable import RunnablePassthrough

logger = logging.getLogger(__name__)


class RAGEngine:
    def __init__(self, provider: ProviderClient = None):
        self.provider = provider or get_best_provider()
        self.vstore = VectorStore()
        self.embed_batch = settings.BATCH_SIZE

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
            template="""You are an expert Indian legal advisor. Analyze the user's situation and provide practical legal guidance based on the context.

USER'S SITUATION: {question}

RELEVANT LEGAL CONTEXT:
{context}

IMPORTANT - FOLLOW THIS STRUCTURE EXACTLY:

Situation Analysis
Briefly summarize the legal situation
Identify key legal issues involved

Applicable Laws & Sections
List relevant legal provisions with section numbers
Explain how each law applies to this situation

Legal Rights & Remedies
What legal rights does the person have?
Available legal remedies and procedures

Recommended Actions
Step-by-step practical advice
Timeline for actions
Required documents/evidence

Potential Outcomes
Best-case and worst-case scenarios
Typical resolution timeframes

Important Caveats
Limitations of this advice
When to consult a practicing lawyer

Legal References:
Sections XXX, YYY of Relevant Act

BASE YOUR ANSWER STRICTLY ON THE PROVIDED LEGAL CONTEXT. If the context doesn't cover specific situational aspects, acknowledge this limitation.

ANSWER:""",
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

            print(f"Initial retrieval: {len(docs)} documents")

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

            for i, (doc, meta, distance) in enumerate(zip(docs, metas, dists)):
                if doc and meta:
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
                        filtered_ids.append(meta.get("doc_id", f"doc_{i}"))

            if not filtered_docs and docs:
                filtered_docs = docs[:k]
                filtered_metas = metas[:k]
                filtered_dists = dists[:k]
                filtered_ids = [
                    meta.get("doc_id", f"doc_{i}") for i, meta in enumerate(metas[:k])
                ]

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
                    "I couldn't find any relevant legal documents to answer your question. Please try rephrasing or check if relevant documents have been ingested.",
                    [],
                )

            context = self._format_context(docs, metas, ids, dists)

            formatted_prompt = self.qa_prompt.format(question=question, context=context)

            answer = self.provider.generate(
                formatted_prompt,
                max_tokens=1000,
            )
            answer = self._clean_response(answer)

            if "Reference Sources:" not in answer and "Sources:" not in answer:
                answer += "\n\nReference Sources: See cited legal documents below"

            sources = []
            for i, (doc_id, meta, doc_text) in enumerate(zip(ids, metas, docs)):
                sources.append(
                    SourceItem(
                        id=doc_id,
                        source=meta,
                        snippet=(
                            doc_text[:300] + "..." if len(doc_text) > 300 else doc_text
                        ),
                    )
                )

            logger.info(f"Generated structured answer with {len(sources)} sources")
            return answer, sources

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error generating answer: {str(e)}", []

    def query(self, question: str, top_k: int = 4) -> ChatResponse:
        try:
            retrieved = self.retrieve(question, k=top_k)

            answer, sources = self.generate_answer(question, retrieved)

            return ChatResponse(answer=answer, sources=sources)

        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return ChatResponse(
                answer=f"Sorry, I encountered an error while processing your question: {str(e)}",
                sources=[],
            )

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

    def query_with_language(self, question: str, language: str = "en", top_k: int = 4):
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

            response = self.query(question=processed_question, top_k=top_k)

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

def get_rag_engine(provider: ProviderClient = None) -> RAGEngine:
    return RAGEngine(provider)
