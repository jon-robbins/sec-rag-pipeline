"""
The RAGPipeline class, which orchestrates the entire RAG system,
from data loading to answer generation.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, Optional, List
import pickle
from pathlib import Path

import pandas as pd

from .chunkers import SmartChunker
# from .config import RAW_DATA_PATH
from .document_store import DocumentStore
from .embedding import EmbeddingManager
from .generation import AnswerGenerator
from .parser import QueryParser
from .search import SearchManager
from .vector_store import VectorStore
from .config import CACHE_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrates the entire RAG pipeline from data loading to querying.
    """
    def __init__(self, tickers_of_interest=None, target_tokens=750, overlap_tokens=150):
        if tickers_of_interest is None:
            tickers_of_interest = ['AAPL', 'META', 'TSLA', 'NVDA', 'AMZN']

        logger.info("Initializing RAG pipeline...")
        self.document_store = DocumentStore(tickers_of_interest=tickers_of_interest)
        
        # --- Caching Logic ---
        hard_ceiling = 1000
        cache_dir_name = f"target_{target_tokens}_overlap_{overlap_tokens}_ceiling_{hard_ceiling}"
        embedding_cache_dir = CACHE_DIR / "embeddings"
        embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        chunk_cache_file = embedding_cache_dir / f"{cache_dir_name}.pkl"

        if chunk_cache_file.exists():
            logger.info(f"✅ Loading cached chunks and embeddings from {chunk_cache_file}")
            with open(chunk_cache_file, "rb") as f:
                self.chunks = pickle.load(f)
        else:
            logger.info(f"💾 Cached chunks not found at {chunk_cache_file}. Generating fresh...")
            
            logger.info("🔄 Generating chunks from sorted sentences...")
            df_sentences = self.document_store.get_all_sentences() # This will trigger data loading
            
            # Add 'item' column for compatibility with chunker
            df_sentences['item'] = df_sentences['section']
            
            chunker = SmartChunker(
                target_tokens=target_tokens,
                hard_ceiling=hard_ceiling,
                overlap_tokens=overlap_tokens
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
            
            logger.info(f"💾 Saving {len(self.chunks)} chunks with embeddings to cache...")
            with open(chunk_cache_file, "wb") as f:
                pickle.dump(self.chunks, f)

        # --- Vector Store Upsert ---
        logger.info("🧠 Initializing vector store...")
        self.vector_store = VectorStore(use_docker=False)
        
        chunk_dicts = [chunk.to_dict() for chunk in self.chunks]
        embeddings_list = [chunk.embedding for chunk in self.chunks]
        
        if any(e is None for e in embeddings_list):
            raise ValueError("Some chunks are missing embeddings. Clear cache and re-run.")

        if self.vector_store.get_status().get("points_count", 0) != len(chunk_dicts):
            logger.info("Vector store is out of sync. Loading data...")
            self.vector_store.upsert_chunks_with_embeddings(chunk_dicts, embeddings_list)
        else:
            logger.info("✅ Vector store is already up to date.")
            
        logger.info("✅ RAG Pipeline Initialized Successfully!")

    def answer(self, question: str, **kwargs) -> Dict[str, Any]:
        """Answer a question using the RAG pipeline."""
        return self.vector_store.answer(question=question, **kwargs)
    
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """Perform a semantic search."""
        return self.vector_store.search(query=query, **kwargs)

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
        
        # Generate summary using the answer generator
        summary = self.vector_store.generate_summary(topic, chunks)
        
        # Extract source information
        sources = []
        for chunk in chunks:
            payload = chunk.get("payload", {})
            ticker_info = payload.get("ticker", "UNKNOWN")
            year_info = payload.get("fiscal_year", "UNKNOWN")
            section_info = payload.get("item", "UNKNOWN")
            section_desc = payload.get("item_desc", "")
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
    
    def get_chunks(self) -> List[Dict[str, Any]]:
        return [chunk.to_dict() for chunk in self.chunks] 