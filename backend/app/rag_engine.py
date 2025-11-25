# backend/app/rag_engine.py
from typing import Tuple, List, Dict, Optional
from pathlib import Path
import logging
from .config import settings
from .provider_client import get_best_provider, ProviderClient
from .vector_store import VectorStore
from .ingestion import split_into_chunks, load_pdf_text
from .schemas import ChatResponse, SourceItem

# For LangChain >= 0.1.0
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

# For compatibility with both versions
try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
except ImportError:
    from langchain.prompts import PromptTemplate
    from langchain.schema.runnable import RunnablePassthrough

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self, provider: ProviderClient = None):
        # Use provided provider or get the best available
        self.provider = provider or get_best_provider()
        self.vstore = VectorStore()
        self.embed_batch = settings.BATCH_SIZE
        
        # Initialize LangChain components
        self._setup_prompts()
        self._setup_chain()
        
        logger.info(f"RAG Engine initialized with provider: {type(self.provider).__name__}")

    def _setup_prompts(self):
        """Setup LangChain prompt templates with precise structured formatting"""
        
        self.qa_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""You are a legal expert on Indian law. Provide a CLEAN, WELL-STRUCTURED answer using ONLY the provided legal context.

    CONTEXT:
    {context}

    QUESTION: {question}

    RESPONSE REQUIREMENTS:
    1. STRUCTURE:
    - Start with 1-2 sentence overview
    - Use **bold headings** for main sections
    - Use • bullet points for lists
    - Keep paragraphs to 2-3 lines maximum
    - End with "**Legal References:**" section

    2. CONTENT:
    - Use ONLY information from provided context
    - Cite specific sections (Section 302, Section 304, etc.)
    - Be precise and factual
    - If context is insufficient, state this clearly

    3. FORMATTING:
    **Overview**
    [Brief introduction]

    **Key Definitions**
    • [Point 1]
    • [Point 2]

    **Main Differences**
    • [Difference 1]
    • [Difference 2]

    **Legal Provisions**
    • [Provision 1]
    • [Provision 2]

    **Legal References:**
    [Mention specific acts/sections used]

    ANSWER:"""
        )
        # Context formatting prompt
        self.context_prompt = PromptTemplate(
            input_variables=["documents"],
            template="""Format the following legal documents into a coherent context:

{documents}

Please format them in a way that maintains legal accuracy and readability."""
        )

    def _setup_chain(self):
        """Setup LangChain runnable chain (for future enhancement)"""
        # This sets up the structure for when we fully integrate LangChain
        # For now, we'll use our custom retrieval but with LangChain prompts
        pass

    def _format_context(self, docs: List[str], metas: List[dict], ids: List[str], dists: List[float]) -> str:
        """Format retrieved documents into a coherent context using LangChain prompt"""
        try:
            # Build document strings with metadata
            document_parts = []
            for i, (doc, meta, doc_id, distance) in enumerate(zip(docs, metas, ids, dists)):
                act_name = meta.get('act', 'Unknown Act')
                relevance_score = 1.0 - (distance / 2.0)  # Convert distance to relevance score
                document_parts.append(
                    f"[Document {i+1} | Source: {act_name} | Relevance: {relevance_score:.2f}]\n"
                    f"{doc[:800]}..."
                )
            
            documents_text = "\n\n".join(document_parts)
            
            # Use LangChain prompt to format context
            formatted_context = self.context_prompt.format(documents=documents_text)
            return formatted_context
            
        except Exception as e:
            logger.error(f"Error formatting context: {e}")
            # Fallback to simple concatenation
            return "\n\n".join([f"[Source {i+1}]: {doc[:600]}..." 
                              for i, doc in enumerate(docs)])

    def ingest_text(self, doc_id: str, text: str, metadata: dict = None) -> bool:
        """Ingest text content into the vector store"""
        try:
            logger.info(f"Ingesting document: {doc_id}")
            
            # Split text into chunks using LangChain text splitter
            chunks = split_into_chunks(text)
            logger.info(f"Split into {len(chunks)} chunks")
            
            # Generate chunk IDs and metadata
            ids = [f"{doc_id}__{i}" for i in range(len(chunks))]
            metadatas = [
                {**(metadata or {}), "chunk_index": i, "doc_id": doc_id} 
                for i in range(len(chunks))
            ]

            # Compute embeddings in batches
            embeddings = []
            for i in range(0, len(chunks), self.embed_batch):
                batch = chunks[i:i + self.embed_batch]
                logger.info(f"Generating embeddings for batch {i//self.embed_batch + 1}")
                embs = self.provider.get_embeddings(batch)
                embeddings.extend(embs)

            # Store in vector store
            self.vstore.collection.add(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            logger.info(f"✅ Successfully ingested {len(chunks)} chunks from {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to ingest {doc_id}: {e}")
            return False

    def ingest_file(self, file_path: str, doc_id: str = None, act_name: str = None) -> bool:
        """Ingest a PDF file into the vector store"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            doc_id = doc_id or path.stem
            logger.info(f"Processing file: {path.name} as {doc_id}")
            
            # Extract text from PDF
            text = load_pdf_text(path)
            if not text:
                raise ValueError("No text extracted from PDF. File might be scanned or corrupted.")
            
            # Prepare metadata
            metadata = {
                "act": act_name or doc_id, 
                "source_file": path.name,
                "source_type": "pdf"
            }
            
            # Ingest the text
            return self.ingest_text(doc_id, text, metadata)
            
        except Exception as e:
            logger.error(f"❌ Failed to ingest file {file_path}: {e}")
            return False

    def retrieve(self, query: str, k: int = 4) -> Dict:
        """Retrieve relevant documents for a query.
        
        Args:
            query: The search query string
            k: Number of results to return (default: 4)
            
        Returns:
            Dict: Contains documents, metadatas, distances, and ids
        """
        try:
            logger.info(f"Retrieving documents for query: '{query[:50]}...'")
            
            # Generate query embedding
            query_embedding = self.provider.get_embeddings([query])[0]
            
            # Query vector store
            results = self.vstore.query(query_embedding, n_results=k)
            
            # Extract IDs from results if needed
            if results.get('metadatas') and results['metadatas'][0]:
                results['ids'] = [[meta.get('doc_id', f"chunk_{i}") 
                                for i, meta in enumerate(results['metadatas'][0])]]
            else:
                results['ids'] = [[]]
                
            logger.info(f"Retrieved {len(results.get('documents', [[]])[0])} documents")
            return results
            
        except Exception as e:
            logger.error(f"❌ Retrieval failed: {e}")
            return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}

    def generate_answer(self, question: str, retrieved: Dict) -> Tuple[str, List[SourceItem]]:
            """Generate structured answer based on retrieved documents"""
            try:
                # Extract retrieved information
                docs = retrieved.get("documents", [[]])[0]
                metas = retrieved.get("metadatas", [[]])[0]
                ids = retrieved.get("ids", [[]])[0]
                dists = retrieved.get("distances", [[]])[0]

                if not docs:
                    return "I couldn't find any relevant legal documents to answer your question. Please try rephrasing or check if relevant documents have been ingested.", []

                # Format context
                context = self._format_context(docs, metas, ids, dists)

                # Generate structured answer
                formatted_prompt = self.qa_prompt.format(
                    question=question,
                    context=context
                )
                
                # Generate answer with specific instructions for structure
                answer = self.provider.generate(
                    formatted_prompt, 
                    max_tokens=800,  # Increased for structured content
                )
                
                # Ensure the answer ends with sources reference
                if "**Reference Sources:**" not in answer and "Sources:" not in answer:
                    answer += "\n\n**Reference Sources:** See cited legal documents below"
                
                # Prepare sources for response
                sources = []
                for i, (doc_id, meta, doc_text) in enumerate(zip(ids, metas, docs)):
                    sources.append(SourceItem(
                        id=doc_id,
                        source=meta,
                        snippet=doc_text[:300] + "..." if len(doc_text) > 300 else doc_text
                    ))
                
                logger.info(f"✅ Generated structured answer with {len(sources)} sources")
                return answer, sources
                
            except Exception as e:
                logger.error(f"❌ Answer generation failed: {e}")
                return f"Error generating answer: {str(e)}", []

    def query(self, question: str, top_k: int = 4) -> ChatResponse:
        """Complete RAG pipeline: retrieve + generate using LangChain concepts"""
        try:
            # Step 1: Retrieve relevant documents
            retrieved = self.retrieve(question, k=top_k)
            
            # Step 2: Generate answer using LangChain prompts
            answer, sources = self.generate_answer(question, retrieved)
            
            return ChatResponse(
                answer=answer,
                sources=sources
            )
            
        except Exception as e:
            logger.error(f"❌ RAG query failed: {e}")
            return ChatResponse(
                answer=f"Sorry, I encountered an error while processing your question: {str(e)}",
                sources=[]
            )

    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        try:
            count = self.vstore.collection.count()
            return {
                "total_documents": count,
                "provider": type(self.provider).__name__,
                "collection": self.vstore.col_name,
                "embedding_model": getattr(self.provider, 'model_name', 'Unknown')
            }
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {"error": str(e)}

    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            self.vstore.client.delete_collection(self.vstore.col_name)
            # Recreate collection
            self.vstore.collection = self.vstore.client.create_collection(
                name=self.vstore.col_name,
                metadata={"source": "legal_docs"}
            )
            logger.info("✅ Collection cleared successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to clear collection: {e}")
            return False

    def get_prompt_templates(self) -> Dict:
        """Get information about the prompt templates being used"""
        return {
            "qa_prompt": self.qa_prompt.template[:100] + "...",
            "context_prompt": self.context_prompt.template[:100] + "...",
            "input_variables": {
                "qa_prompt": self.qa_prompt.input_variables,
                "context_prompt": self.context_prompt.input_variables
            }
        }


# Utility function for easy usage
def get_rag_engine(provider: ProviderClient = None) -> RAGEngine:
    """Get a RAG engine instance with optional custom provider"""
    return RAGEngine(provider)


# Test the RAG engine
if __name__ == "__main__":
    print("🧪 Testing RAG Engine with LangChain...")
    
    try:
        # Initialize RAG engine
        rag = RAGEngine()
        
        # Test stats
        stats = rag.get_stats()
        print(f"📊 Vector Store Stats: {stats}")
        
        # Test prompt templates
        prompts = rag.get_prompt_templates()
        print(f"📝 Prompt Templates: {prompts}")
        
        # Test retrieval with a sample legal question
        test_question = "What is cheating under Indian law?"
        print(f"🔍 Testing query: {test_question}")
        
        response = rag.query(test_question, top_k=2)
        print(f"🤖 Answer: {response.answer}")
        print(f"📚 Sources: {len(response.sources)}")
        
        for i, source in enumerate(response.sources):
            print(f"   {i+1}. {source.snippet[:100]}...")
            
        print("✅ RAG Engine with LangChain test completed successfully!")
        
    except Exception as e:
        print(f"❌ RAG Engine test failed: {e}")
        import traceback
        traceback.print_exc()