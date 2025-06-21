"""
The RAGPipeline class, which orchestrates the entire RAG system,
from data loading to answer generation.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from .chunkers import SmartChunker

# from .config import RAW_DATA_PATH
from .document_store import DocumentStore
from .embedding import EmbeddingManager
from .generation import AnswerGenerator
from .parser import QueryParser
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrates the entire RAG pipeline from data loading to querying.

    This is the main entry point for interacting with the SEC insights library.
    It handles data loading, chunking, embedding, and querying.

    Attributes
    ----------
    document_store : DocumentStore
        The store for accessing raw document data.
    embedding_manager : EmbeddingManager
        The manager for creating text embeddings.
    query_parser : QueryParser
        The parser for interpreting natural language queries.
    answer_generator : AnswerGenerator
        The generator for creating answers from context.
    vector_store : VectorStore
        The vector database for storing and retrieving chunks.
    chunks : List[Chunk]
        A list of all processed chunk objects for the current configuration.
    adjacent_map : Dict[str, List[str]]
        A mapping of chunk IDs to their adjacent neighbors.
    """

    def __init__(
        self,
        root_dir: Path,
        tickers_of_interest: Optional[List[str]] = None,
        target_tokens: int = 750,
        overlap_tokens: int = 150,
        hard_ceiling: int = 1000,
        use_docker: bool = False,
        docker_port: int = 6333,
    ):
        if tickers_of_interest is None:
            tickers_of_interest = ["AAPL", "META", "TSLA", "NVDA", "AMZN"]

        self.root_dir = root_dir

        logger.info("Initializing RAG pipeline...")
        raw_data_path = self.root_dir / "data" / "raw" / "df_filings_full.parquet"
        self.document_store = DocumentStore(
            raw_data_path=raw_data_path, tickers_of_interest=tickers_of_interest
        )

        # Core components
        self.embedding_manager = EmbeddingManager()
        self.query_parser = QueryParser()
        self.answer_generator = AnswerGenerator()

        # Store chunking parameters
        self.target_tokens = target_tokens
        self.overlap_tokens = overlap_tokens
        self.hard_ceiling = hard_ceiling

        # --- Caching Logic ---
        cache_dir_name = (
            f"target_{target_tokens}_overlap_{overlap_tokens}_ceiling_{hard_ceiling}"
        )
        embedding_cache_dir = self.root_dir / "data" / "cache" / "embeddings"
        embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        chunk_cache_file = embedding_cache_dir / f"{cache_dir_name}.pkl"

        if chunk_cache_file.exists():
            logger.info(
                f"✅ Loading cached chunks and embeddings from {chunk_cache_file}"
            )
            with open(chunk_cache_file, "rb") as f:
                self.chunks = pickle.load(f)
        else:
            logger.info(
                f"💾 Cached chunks not found at {chunk_cache_file}. Generating fresh..."
            )

            logger.info("🔄 Generating chunks from sorted sentences...")
            df_sentences = (
                self.document_store.get_all_sentences()
            )  # This will trigger data loading

            # Add 'item' column for compatibility with chunker
            df_sentences["item"] = df_sentences["section"]

            chunker = SmartChunker(
                target_tokens=target_tokens,
                hard_ceiling=hard_ceiling,
                overlap_tokens=overlap_tokens,
            )
            chunk_objects = chunker.run(df_sentences)

            logger.info("🔄 Generating embeddings for chunks...")
            embedding_manager = EmbeddingManager()
            texts = [chunk.text for chunk in chunk_objects]
            embeddings = embedding_manager.embed_texts_in_batches(texts)
            logger.info(f"✅ Generated {len(embeddings)} embeddings")

            if len(chunk_objects) != len(embeddings):
                raise ValueError("Mismatch between number of chunks and embeddings")

            for i, chunk in enumerate(chunk_objects):
                chunk.embedding = embeddings[i]

            self.chunks = chunk_objects

            logger.info(
                f"💾 Saving {len(self.chunks)} chunks with embeddings to cache..."
            )
            with open(chunk_cache_file, "wb") as f:
                pickle.dump(self.chunks, f)

        # --- Vector Store Upsert ---
        logger.info("🧠 Initializing vector store...")

        if use_docker:
            self.vector_store = VectorStore(
                use_docker=use_docker,
                embedding_manager=self.embedding_manager,
                host="localhost",
                port=docker_port,
            )
        else:
            self.vector_store = VectorStore(
                use_docker=use_docker, embedding_manager=self.embedding_manager
            )

        chunk_dicts = [chunk.to_dict() for chunk in self.chunks]
        embeddings_list = [chunk.embedding for chunk in self.chunks]

        if any(e is None for e in embeddings_list):
            raise ValueError(
                "Some chunks are missing embeddings. Clear cache and re-run."
            )

        if self.vector_store.get_status().get("points_count", 0) != len(chunk_dicts):
            logger.info("Vector store is out of sync. Loading data...")
            self.vector_store.upsert_chunks(chunk_dicts, embeddings_list)
        else:
            logger.info("✅ Vector store is already up to date.")

        logger.info("✅ RAG Pipeline Initialized Successfully!")

        # --- Build adjacency map for evaluation ---
        self.adjacent_map = self._build_adjacent_map()

    def _build_adjacent_map(self):
        """Create a mapping from chunk_id to its immediate neighbours (prev, next)."""
        from collections import defaultdict

        section_groups = defaultdict(list)
        # Group chunks by (ticker, year, section)
        for ch in self.chunks:
            md = ch.metadata
            # Handle both old ('item') and new ('section') field names
            section = md.get("section") or md.get("item")
            key = (md["ticker"], md["fiscal_year"], section)

            # Use seq / slice_idx for ordering; fall back to parsing human_readable_id
            seq = md.get("seq")
            slice_idx = md.get("slice_idx", 0)
            if seq is None:
                # Try to parse from human_readable_id
                try:
                    parts = md.get("human_readable_id", "").split("_")
                    seq = int(parts[-1])
                except Exception:
                    seq = 0
            section_groups[key].append((seq, slice_idx, ch))

        adjacent_map = {}
        for group in section_groups.values():
            # sort by seq then slice_idx
            group.sort(key=lambda t: (t[0], t[1]))
            for i, (_, __, ch) in enumerate(group):
                neighbours = []
                if i > 0:
                    neighbours.append(group[i - 1][2].id)
                if i < len(group) - 1:
                    neighbours.append(group[i + 1][2].id)
                adjacent_map[ch.id] = neighbours

        return adjacent_map

    def answer(self, question: str, **kwargs: Any) -> Dict[str, Any]:
        """Answers a question by performing a semantic search and generating a response.

        Parameters
        ----------
        question : str
            The natural language question to answer.
        **kwargs : Any
            Additional keyword arguments to pass to the search function, such as `top_k`.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the generated answer, sources, and other metadata.
        """
        chunks = self.search(query=question, **kwargs)
        result = self.answer_generator.generate_answer(question=question, chunks=chunks)
        result["search_results_count"] = len(chunks)
        result["top_score"] = chunks[0]["score"] if chunks else 0.0
        return result

    def search(
        self, query: str, top_k: int = 10, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Performs a semantic search for a given query.

        This method parses the query, generates a vector embedding, and retrieves
        the most relevant document chunks from the vector store.

        Parameters
        ----------
        query : str
            The natural language query string.
        top_k : int, optional
            The number of top results to return, by default 10.
        **kwargs : Any
            Additional keyword arguments for filtering, e.g., `ticker`, `fiscal_year`.

        Returns
        -------
        List[Dict[str, Any]]
            A list of the top k retrieved chunks, including their metadata and scores.
        """
        parsed_query = self.query_parser.parse_query(query)
        ticker = kwargs.get("ticker") or parsed_query.get("ticker")
        fiscal_year = kwargs.get("fiscal_year") or parsed_query.get("fiscal_year")
        sections = kwargs.get("sections") or parsed_query.get("sections")

        query_vector = self.embedding_manager.embed_texts_in_batches([query])[0]

        return self.vector_store.search(
            query_vector=query_vector,
            ticker=ticker,
            fiscal_year=fiscal_year,
            sections=sections,
            top_k=top_k,
        )

    def retrieve_by_filter(self, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve documents by metadata filters."""
        return self.vector_store.retrieve_by_filter(**kwargs)

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
        chunks = self.search(
            query=topic,
            ticker=ticker,
            fiscal_year=fiscal_year,
            sections=sections,
            top_k=top_k,
        )

        if not chunks:
            return {
                "summary": f"No relevant information found about {topic}.",
                "sources": [],
                "chunks_used": 0,
            }

        result = self.answer(question=topic)
        return {
            "summary": result.get("answer", ""),
            "sources": result.get("sources", [])[:10],
            "chunks_used": len(chunks),
        }

    def generate_chunks_from_documents(
        self, tickers: List[str], fiscal_years: List[int]
    ) -> List[Any]:
        """
        Generates chunks from a filtered set of documents without creating embeddings.

        Args:
            tickers: A list of tickers to include.
            fiscal_years: A list of fiscal years to include.

        Returns:
            A list of chunk objects, without embeddings.
        """
        logger.info(
            f"🔄 Generating chunks for tickers {tickers} and years {fiscal_years}..."
        )

        # 1. Get filtered sentences from the document store
        df_sentences = self.document_store.get_sentences_by_filter(
            tickers, fiscal_years
        )
        if df_sentences.empty:
            logger.info("No documents found for the given filters.")
            return []

        # 2. Add 'item' column for compatibility with chunker
        df_sentences["item"] = df_sentences["section"]

        # 3. Initialize chunker from pipeline settings (or use defaults)
        #    This assumes the pipeline's chunking config is what you want to use.
        target_tokens = getattr(self, "target_tokens", 750)
        overlap_tokens = getattr(self, "overlap_tokens", 150)
        hard_ceiling = getattr(self, "hard_ceiling", 1000)

        chunker = SmartChunker(
            target_tokens=target_tokens,
            hard_ceiling=hard_ceiling,
            overlap_tokens=overlap_tokens,
        )

        # 4. Run chunking
        chunk_objects = chunker.run(df_sentences)
        logger.info(f"✅ Generated {len(chunk_objects)} chunks.")

        return chunk_objects

    def get_chunks(self) -> List[Dict[str, Any]]:
        return [chunk.to_dict() for chunk in self.chunks]

    def embed_chunks(self, chunks_to_embed: List[Any]) -> List[Any]:
        """
        Generates embeddings for a specific list of chunk objects.

        Args:
            chunks_to_embed: A list of chunk objects to be embedded.

        Returns:
            The list of chunk objects, updated with their embeddings.
        """
        logger.info(f"🔄 Generating embeddings for {len(chunks_to_embed)} chunks...")
        embedding_manager = EmbeddingManager()
        texts = [chunk.text for chunk in chunks_to_embed]
        embeddings = embedding_manager.embed_texts_in_batches(texts)
        logger.info(f"✅ Generated {len(embeddings)} embeddings")

        if len(chunks_to_embed) != len(embeddings):
            raise ValueError("Mismatch between number of chunks and embeddings")

        for i, chunk in enumerate(chunks_to_embed):
            chunk.embedding = embeddings[i]

        return chunks_to_embed
