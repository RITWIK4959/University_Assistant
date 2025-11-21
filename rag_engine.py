# rag_engine.py
import chromadb
import logging
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    Settings,
    Document,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.groq import Groq

# Load environment variables
load_dotenv(".env.local")

# ---------------- CONFIGURATION ---------------- #
DATA_DIR = Path("./Nexi Data")  # Folder where your PDFs are stored
PERSIST_DIR = Path("./chromadb")              # Folder where vector DB will be saved
PERSIST_DIR.mkdir(parents=True, exist_ok=True)

# ✅ FIXED: Set up Groq LLM for proper chunk processing
groq_api_key = os.getenv("GROQ_API_KEY")
if groq_api_key:
    Settings.llm = Groq(
        model="llama-3.1-8b-instant",
        api_key=groq_api_key,
        temperature=0.0  # ✅ FIXED: Zero temperature for maximum factual accuracy
    )
    print("✅ RAG Engine configured with Groq LLM for proper chunk processing")
else:
    print("⚠️ GROQ_API_KEY not found - RAG will return raw chunks without LLM processing")
    Settings.llm = None 
# Enhanced prompt template for perfect document accuracy
UNIVERSITY_PROMPT = PromptTemplate(
    """
    Extract the exact answer from the context below. Follow these rules strictly:
    
    1. ONLY use information directly stated in the context
    2. Quote exact text, dates, numbers, and procedures from the context
    3. If the answer is not in the context, respond: "This information is not available in the provided documents"
    4. Keep answers brief and factual
    5. Do not interpret, assume, or add any information
    
    Context:
    {context_str}
    
    Question: {query_str}
    
    Answer (extract exact information only):
    """
)

# ---------------- LOGGING ---------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RAG")

# ---------------- RAG ENGINE ---------------- #
class UniversityRAGEngine:
    def __init__(self):
        """Initialize embedding model, Chroma, and query engine."""
        logger.info("Initializing University RAG Engine...")
        self.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        Settings.embed_model = self.embed_model

        self.client = chromadb.PersistentClient(path=str(PERSIST_DIR))
        self.collection = self.client.get_or_create_collection("university_data")
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

        if self.collection.count() == 0:
            logger.info("No existing data found — ingesting documents.")
            self._ingest_documents()
        else:
            logger.info(f"Loaded existing collection with {self.collection.count()} items.")
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                storage_context=self.storage_context,
            )

        # Build query engine with maximum accuracy
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,  # Increased to get more relevant context
            response_mode="compact",  # Simple and direct responses
            use_async=False,
            streaming=False,
            verbose=False  # Disable verbose to reduce noise
        )
        logger.info("RAG Engine ready.")

    def _ingest_documents(self):
        """Load PDFs from DATA_DIR, embed and store them in Chroma with improved chunking."""
        if not DATA_DIR.exists():
            logger.warning(f"Data folder '{DATA_DIR}' not found. Creating empty index.")
            self.index = VectorStoreIndex([], storage_context=self.storage_context)
            return

        # Load documents with better text extraction
        documents = SimpleDirectoryReader(
            input_dir=str(DATA_DIR),
            required_exts=[".pdf"],
            recursive=True
        ).load_data()
        
        logger.info(f"Loaded {len(documents)} documents from {DATA_DIR}")
        
        # Configure text splitter for better chunking
        text_splitter = SentenceSplitter(
            chunk_size=512,  # Smaller chunks for better accuracy
            chunk_overlap=50,  # Some overlap to maintain context
            separator=" "
        )
        
        # Process documents with improved chunking
        logger.info("Processing documents with improved chunking...")
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            embed_model=self.embed_model,
            transformations=[text_splitter]  # Apply custom chunking
        )
        
        self.index.storage_context.persist()
        logger.info(f"Documents successfully ingested with {len(documents)} files and improved chunking.")

    def get_rag_answer(self, query: str) -> str:
        """Query the vector database and return an answer."""
        if not query.strip():
            return "Please ask a valid question."
        try:
            logger.info(f"Querying RAG engine: {query}")
            response = self.query_engine.query(query)
            return str(response)
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return "Sorry, I could not retrieve the answer right now."

    async def get_rag_answer_async(self, query: str) -> str:
        """Async wrapper for LiveKit integration."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_rag_answer, query)

# ---------------- GLOBAL INSTANCE ---------------- #
_rag_engine = None

def initialize_rag_engine():
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = UniversityRAGEngine()
    return _rag_engine

def get_rag_answer(query: str):
    global _rag_engine
    if _rag_engine is None:
        initialize_rag_engine()
    return _rag_engine.get_rag_answer(query)

async def get_rag_answer_async(query: str):
    global _rag_engine
    if _rag_engine is None:
        initialize_rag_engine()
    return await _rag_engine.get_rag_answer_async(query)

# ---------------- TEST ---------------- #
if __name__ == "__main__":
    print("=== University RAG Engine Test ===")
    initialize_rag_engine()
    
    test_questions = [
        "What is the minimum attendance percentage required?",
        "What are the consequences of poor attendance?",
        "What is considered academic misconduct according to the code of conduct?",
        "What happens if a student is caught cheating?",
        "Tell me about attendance policy"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Q: {question}")
        try:
            answer = get_rag_answer(question)
            print(f"   A: {answer}")
        except Exception as e:
            print(f"   Error: {e}")
        
    print("\n=== RAG Engine Test Complete ===")
