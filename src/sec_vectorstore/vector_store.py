"""
Main VectorStore class for the SEC Vector Store.
"""

from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient

from .config import VectorStoreConfig
from .embedding import EmbeddingManager
from .parser import QueryParser
from .collection import CollectionManager
from .search import SearchManager
from .generation import AnswerGenerator
from .docker_utils import get_docker_status


class VectorStore:
    """
    Main facade class for the SEC Vector Store.
    Orchestrates embedding, collection management, and search operations.
    """

    def __init__(
        self,
        *,
        collection_name: str = "sec_filings",
        dim: int = 1536,
        model: str = "text-embedding-3-small",
        client: Optional[QdrantClient] = None,
        openai_key: Optional[str] = None,
        use_docker: bool = False,
        docker_host: str = "localhost",
        docker_port: int = 6333,
        auto_fallback_to_memory: bool = True,
    ) -> None:
        """
        Initialize the VectorStore.
        
        Args:
            collection_name: Name of the Qdrant collection
            dim: Vector dimension (should match embedding model)
            model: OpenAI embedding model name
            client: Optional pre-configured Qdrant client
            openai_key: OpenAI API key (uses env var if not provided)
            use_docker: Whether to use Docker Qdrant or in-memory
            docker_host: Docker host address
            docker_port: Docker port number
            auto_fallback_to_memory: Auto-fallback to memory on Docker errors
        """
        # Configuration
        self.config = VectorStoreConfig(
            collection_name=collection_name,
            dim=dim,
            model=model,
            openai_key=openai_key,
            use_docker=use_docker,
            docker_host=docker_host,
            docker_port=docker_port,
            auto_fallback_to_memory=auto_fallback_to_memory,
        )

        # Set up Qdrant client
        self.client = self._setup_client(client)

        # Initialize managers
        self.embedding_manager = EmbeddingManager(
            model=self.config.model,
            openai_key=self.config.openai_key
        )
        
        self.query_parser = QueryParser(openai_key=self.config.openai_key)
        
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
        
        self.answer_generator = AnswerGenerator(
            openai_key=self.config.openai_key,
            model="gpt-4o-mini"
        )

    def _setup_client(self, client: Optional[QdrantClient]) -> QdrantClient:
        """Set up the Qdrant client based on configuration."""
        if client:
            return client
        elif self.config.use_docker:
            print(f"🐳 Using Docker Qdrant at {self.config.docker_host}:{self.config.docker_port}")
            
            # Create Docker client with timeout
            docker_client = QdrantClient(
                host=self.config.docker_host,
                port=self.config.docker_port,
                timeout=self.config.docker_timeout,
            )
            
            # Test the connection
            try:
                collections = docker_client.get_collections()
                print(f"✅ Docker connection successful, found {len(collections.collections)} collections")
                return docker_client
            except Exception as e:
                print(f"❌ Docker connection failed: {e}")
                print("💡 Make sure Docker Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
                raise
        else:
            print("🧠 Using in-memory Qdrant")
            return QdrantClient(":memory:")

    def init_collection(self) -> None:
        """Initialize the collection."""
        self.collection_manager.init_collection()
        
        # Update client reference in case of fallback to memory mode
        if self.collection_manager.client != self.client:
            self.client = self.collection_manager.client
            self.search_manager.client = self.client

    def upsert(
        self,
        *,
        metas: List[Dict[str, Any]],
        texts: Optional[List[str]] = None,
        vectors: Optional[List[List[float]]] = None,
        ids: Optional[List[Any]] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """Upsert vectors and metadata."""
        self.collection_manager.upsert(
            metas=metas,
            texts=texts,
            vectors=vectors,
            ids=ids,
            batch_size=batch_size
        )
        
        # Update client reference in case of fallback to memory mode
        if self.collection_manager.client != self.client:
            self.client = self.collection_manager.client
            self.search_manager.client = self.client

    def search(
        self,
        query: str,
        *,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        sections: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        return self.search_manager.search(
            query=query,
            ticker=ticker,
            fiscal_year=fiscal_year,
            sections=sections,
            top_k=top_k
        )

    def answer(
        self,
        question: str,
        *,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        sections: Optional[List[str]] = None,
        top_k: int = 10,
        max_chunks: int = 10,
        max_context_length: int = 8000,
    ) -> Dict[str, Any]:
        """
        Answer a question using retrieved documents and GPT.
        
        Args:
            question: The question to answer
            ticker: Optional ticker symbol filter
            fiscal_year: Optional fiscal year filter
            sections: Optional SEC sections filter
            top_k: Number of chunks to retrieve
            max_chunks: Maximum chunks to use for answer generation
            max_context_length: Maximum context length for GPT
            
        Returns:
            Dictionary with answer, sources, and confidence information
        """
        # First, search for relevant chunks
        chunks = self.search(
            query=question,
            ticker=ticker,
            fiscal_year=fiscal_year,
            sections=sections,
            top_k=top_k
        )
        
        # Generate answer using retrieved chunks
        result = self.answer_generator.generate_answer(
            question=question,
            chunks=chunks,
            max_chunks=max_chunks,
            max_context_length=max_context_length
        )
        
        # Add search metadata
        result["search_results"] = len(chunks)
        result["top_score"] = chunks[0]["score"] if chunks else 0.0
        
        return result

    def summarize(
        self,
        topic: str,
        *,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        sections: Optional[List[str]] = None,
        top_k: int = 15,
    ) -> Dict[str, Any]:
        """
        Generate a summary about a topic using retrieved documents.
        
        Args:
            topic: The topic to summarize
            ticker: Optional ticker symbol filter
            fiscal_year: Optional fiscal year filter
            sections: Optional SEC sections filter
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with summary and source information
        """
        # Search for relevant chunks
        chunks = self.search(
            query=topic,
            ticker=ticker,
            fiscal_year=fiscal_year,
            sections=sections,
            top_k=top_k
        )
        
        if not chunks:
            return {
                "summary": f"No relevant information found about {topic}.",
                "sources": [],
                "chunks_used": 0
            }
        
        # Generate summary
        summary = self.answer_generator.generate_summary(chunks, topic)
        
        # Extract source information
        sources = []
        for chunk in chunks:
            ticker_info = chunk.get("ticker", "UNKNOWN")
            year_info = chunk.get("fiscal_year", "UNKNOWN")
            section_info = chunk.get("item", "UNKNOWN")
            section_desc = chunk.get("item_desc", "")
            score = chunk.get("score", 0.0)
            
            source = f"{ticker_info} {year_info} Section {section_info}"
            if section_desc:
                source += f" ({section_desc})"
            source += f" [Score: {score:.3f}]"
            sources.append(source)
        
        return {
            "summary": summary,
            "sources": sources[:10],  # Limit to top 10 sources
            "chunks_used": len(chunks)
        }

    def get_status(self) -> Dict[str, Any]:
        """Get status information about the vector store."""
        status = {
            "mode": "docker" if self.config.use_docker else "memory",
            "collection_name": self.config.collection_name,
            "model": self.config.model,
            "dimensions": self.config.dim,
        }
        
        if self.config.use_docker:
            docker_status = get_docker_status(
                self.config.docker_host, 
                self.config.docker_port
            )
            status["docker"] = docker_status
        
        try:
            if self.client.collection_exists(self.config.collection_name):
                info = self.client.get_collection(self.config.collection_name)
                status["points_count"] = info.points_count
            else:
                status["points_count"] = 0
        except Exception as e:
            status["error"] = str(e)
        
        return status


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