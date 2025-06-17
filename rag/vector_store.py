"""
Main VectorStore class for the SEC Vector Store.

Data Sources:
=============
This vector store is designed to work with SEC 10-K filings data. The data pipeline involves:

1. **Raw Data**: JanosAudran/financial-reports-sec dataset from Hugging Face
   - Contains sentence-level SEC filing data for multiple companies
   - Each row represents a sentence with metadata (ticker, fiscal year, section, etc.)

2. **Processing Options**:
   - **Chunked Data**: Sentences grouped into semantic chunks (data/chunks.pkl)
   - **Full Documents**: All sentences concatenated per filing (via DownloadCorpus)
   - **Embeddings**: Pre-computed OpenAI embeddings (embeddings/ directory)

3. **Loading**: Use load_data.py to populate the vector store from processed data

Example Usage:
    # Load pre-processed chunks
    from rag.load_data import load_chunks_to_vectorstore
    vs = load_chunks_to_vectorstore()
    
    # Or create empty store and load via document store
    from rag.document_store import DocumentStore
    doc_store = DocumentStore()
    filings = doc_store.get_filings(['AAPL', 'META'], [2022, 2023])
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from .config import VectorStoreConfig
from .embedding import EmbeddingManager
from .parser import QueryParser
from .collection import CollectionManager
from .search import SearchManager
from .generation import AnswerGenerator

if TYPE_CHECKING:
    from .pipeline import RAGPipeline


class VectorStore:
    """A wrapper for the Qdrant client that simplifies interactions."""

    def __init__(
        self,
        *,
        use_docker: bool = False,
        collection_name: str = "sec_filings",
        **kwargs,
    ) -> None:
        """
        Initializes the VectorStore.
        
        Args:
            use_docker: Whether to connect to a Docker-based Qdrant instance.
            collection_name: The name of the collection to use.
            **kwargs: Additional arguments for the QdrantClient.
        """
        self.client = self._setup_client(use_docker=use_docker, **kwargs)
        self.config = VectorStoreConfig(collection_name=collection_name)
        
        self.query_parser = QueryParser()
        self.embedding_manager = EmbeddingManager()
        self.answer_generator = AnswerGenerator()

        self.collection_manager = CollectionManager(
            client=self.client,
            config=self.config,
            embedding_manager=self.embedding_manager
        )
        
        self.search_manager = SearchManager(
            client=self.client,
            config=self.config,
            embedding_manager=self.embedding_manager,
            query_parser=self.query_parser
        )

        # Ensure the collection exists
        self.collection_manager.init_collection()

    def _setup_client(self, use_docker: bool, **kwargs) -> QdrantClient:
        """Set up the Qdrant client based on configuration."""
        if use_docker:
            # Docker connection logic...
            try:
                client = QdrantClient(**kwargs)
                client.get_collections()
                print("✅ Docker Qdrant connection successful.")
                return client
            except Exception as e:
                print(f"❌ Docker connection failed: {e}")
                raise
        else:
            print("🧠 Using in-memory Qdrant")
            return QdrantClient(":memory:")

    def answer(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Answer a question by searching for context and generating a response.
        """
        # First, search for relevant chunks
        chunks = self.search(query=question, **kwargs)
        
        # Then, generate an answer using the retrieved chunks
        result = self.answer_generator.generate_answer(
            question=question,
            chunks=chunks,
        )
        
        result["search_results_count"] = len(chunks)
        result["top_score"] = chunks[0]["score"] if chunks else 0.0
        
        return result

    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Search for similar documents with optional filtering."""
        return self.search_manager.search(query=query, **kwargs)

    def upsert_chunks(self, chunks: List[Dict[str, Any]]):
        """Upsert a list of chunk dictionaries into the collection."""
        self.collection_manager.upsert_chunks(chunks)

    def retrieve_by_filter(self, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve documents based on metadata filters."""
        return self.search_manager.retrieve_by_filter(**kwargs)

    def upsert_chunks_with_embeddings(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Upsert chunks and their pre-computed embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("The number of chunks and embeddings must be the same.")

        # Ensure the collection is initialized before upserting
        self.collection_manager.init_collection()

        # The CollectionManager expects these specific arguments
        self.collection_manager.upsert(
            metas=[chunk['metadata'] for chunk in chunks],
            ids=[chunk['id'] for chunk in chunks],
            vectors=embeddings,
            texts=[chunk['text'] for chunk in chunks]
        )

    def generate_summary(self, topic: str, chunks: List[Dict[str, Any]]) -> str:
        """Helper method to generate a summary from chunks."""
        return self.answer_generator.generate_summary(chunks, topic)

    def get_status(self) -> Dict[str, Any]:
        """Get the status of the vector store and its collection."""
        return self.collection_manager.get_status()


def create_vector_store(
    use_docker: bool = False,
    collection_name: str = "sec_filings",
    auto_fallback_to_memory: bool = True,
    **kwargs
) -> VectorStore:
    """
    Convenience function to create a VectorStore.
    
    Args:
        use_docker: If True, use Docker Qdrant. If False, use in-memory.
        collection_name: Name of the collection
        auto_fallback_to_memory: If True, automatically fallback to memory mode on Docker errors
        **kwargs: Additional arguments passed to VectorStore
    
    Returns:
        Configured VectorStore instance
    """
    return VectorStore(
        collection_name=collection_name,
        use_docker=use_docker,
        auto_fallback_to_memory=auto_fallback_to_memory,
        **kwargs
    ) 