"""
The RAGPipeline class, which orchestrates the entire RAG system,
from data loading to answer generation.
"""
from __future__ import annotations
import os
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

from .vector_store import VectorStore
from .document_store import DocumentStore
from .data_processing.prepare_data import prepare_and_save_data
from .data_processing.chunkers import SmartChunker
from .embedding import EmbeddingManager

# Define cache paths
CACHE_DIR = Path("cache")
PROCESSED_DATA_PATH = CACHE_DIR / "filings_processed.parquet"


class RAGPipeline:
    """
    Orchestrates the entire RAG pipeline from data loading to querying.
    The constructor handles the entire setup process, including data
    preparation, chunking, and embedding generation. Chunks are always
    generated fresh to ensure consistency.
    """
    def __init__(self):
        """
        Initializes the full RAG pipeline, processing data if necessary.
        """
        print("🔧 Initializing RAG Pipeline...")
        CACHE_DIR.mkdir(exist_ok=True)

        # 1. Prepare and load processed filings
        if not PROCESSED_DATA_PATH.exists():
            print(f"Processed data not found. Generating from source...")
            prepare_and_save_data(output_path=str(PROCESSED_DATA_PATH))
        else:
            print(f"✅ Found processed data at {PROCESSED_DATA_PATH}")
        self.document_store = DocumentStore(processed_path=str(PROCESSED_DATA_PATH))

        # 2. Always generate chunks fresh from processed data
        print("🔄 Generating chunks fresh from processed data...")
        df_processed = self.document_store.get_all_filings_for_chunking()
        
        exploded_data = []
        for _, row in df_processed.iterrows():
            report = row.get('report', {})
            for section, sentences in report.items():
                # Bug fix: Ensure sentences is a list before joining
                if isinstance(sentences, list):
                    text = ' '.join(sentences)
                    exploded_data.append({
                        'text': text,
                        'ticker': row['ticker'],
                        'fiscal_year': row['fiscal_year'],
                        'section': section,
                        'item': section
                    })
        df_exploded = pd.DataFrame(exploded_data)
        print(f"📄 Exploded into {len(df_exploded)} section documents for chunking.")

        chunker = SmartChunker()
        chunk_objects = chunker.run(df_exploded)
        chunks = [chunk.to_dict() for chunk in chunk_objects]
        self.chunks = chunks  # Store chunks for evaluation access
        print(f"✅ Generated {len(chunks)} chunks fresh")

        # 3. Generate embeddings fresh each run
        print("🔄 Generating embeddings for chunks...")
        embedding_manager = EmbeddingManager()
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedding_manager.embed_texts_in_batches(texts)
        print(f"✅ Generated {len(embeddings)} embeddings")

        # 4. Initialize and load Vector Store
        print("🧠 Initializing vector store...")
        self.vector_store = VectorStore(use_docker=False)
        if self.vector_store.get_status().get("points_count", 0) != len(chunks):
            print("Vector store is out of sync. Loading data...")
            
            # The chunk 'id' is now the UUID. 'human_readable_id' is in metadata.
            self.vector_store.upsert_chunks_with_embeddings(chunks, embeddings)
        else:
            print("✅ Vector store is already up to date.")
            
        print("✅ RAG Pipeline Initialized Successfully!")

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