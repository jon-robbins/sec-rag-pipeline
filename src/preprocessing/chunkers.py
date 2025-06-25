# src/chunkers.py
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from pydantic import BaseModel, Field

from src.utils.config import SEC_10K_SECTIONS

logger = logging.getLogger(__name__)

# A constant UUID namespace to ensure all chunk UUIDs are deterministic.
NAMESPACE = uuid.UUID("a9535359-4972-4363-8344-48618398e8d8")


class Chunk(BaseModel):
    """A Pydantic model for a data chunk, which is a piece of text from a document."""

    id: str = Field(
        description="A unique identifier for the chunk, generated as a UUID."
    )
    ticker: str = Field(
        description="The stock ticker symbol of the company, e.g., 'AAPL'."
    )
    fiscal_year: int = Field(description="The fiscal year of the document, e.g., 2023.")
    section: str = Field(description="The section of the document, e.g., '1A'.")
    text: str = Field(description="The actual text content of the chunk.")
    tokens: int = Field(description="The number of tokens in the chunk's text.")
    # The parent UUID for a given chunk
    parent_id: Optional[str] = Field(
        default=None, description="The UUID of the parent chunk."
    )
    # The starting character index of the chunk in the document.
    start_index: int = Field(description="The starting character index of the chunk.")
    # The ending character index of the chunk in the document.
    end_index: int = Field(description="The ending character index of the chunk.")
    # Vector embedding for semantic search
    embedding: Optional[List[float]] = Field(
        default=None, description="The vector embedding of the chunk's text."
    )

    @property
    def metadata(self) -> dict:
        """Return metadata dictionary for compatibility with QA generation."""
        return {
            "ticker": self.ticker,
            "fiscal_year": self.fiscal_year,
            "section": self.section,
            "item": self.section,  # Alias for backward compatibility
        }

    @classmethod
    def from_text(
        cls, text: str, ticker: str, fiscal_year: int, section: str, **kwargs: dict
    ) -> Chunk:
        """Creates a Chunk instance from raw text and metadata.

        A UUID is automatically generated for the chunk based on its content.

        Parameters
        ----------
        text : str
            The text content of the chunk.
        ticker : str
            The stock ticker symbol.
        fiscal_year : int
            The fiscal year.
        section : str
            The document section.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        Chunk
            A new Chunk instance.
        """
        id = str(uuid.uuid5(NAMESPACE, text))
        tokens = len(text.split())  # A simple whitespace-based token count
        return cls(
            id=id,
            text=text,
            ticker=ticker,
            fiscal_year=fiscal_year,
            section=section,
            tokens=tokens,
            **kwargs,
        )


class ChunkingConfig(BaseModel):
    """Configuration for the chunking process."""

    target_tokens: int = Field(
        default=150, description="The target number of tokens per chunk."
    )
    overlap_tokens: int = Field(
        default=50,
        description="The number of tokens to overlap between consecutive chunks.",
    )
    hard_ceiling: int = Field(
        default=500,
        description="A hard token limit to prevent excessively large chunks.",
    )


class Chunker:
    """A text chunker that uses a two-phase, semantic splitting strategy.

    This chunker first splits text by sentences and then groups them into
    chunks of a target token size. This strategy aims to preserve semantic
    meaning by keeping sentences intact.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None) -> None:
        self.config = config or ChunkingConfig()
        self._setup_splitters()

    def _setup_splitters(self) -> None:
        """Initializes the text splitters based on the configuration."""
        self.sentence_splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=self.config.overlap_tokens,
            tokens_per_chunk=self.config.target_tokens,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.hard_ceiling,
            chunk_overlap=self.config.overlap_tokens,
            length_function=len,
        )

    def _split_by_sentences(self, text: str) -> List[str]:
        """Phase 1: Split text into sentences."""
        return self.sentence_splitter.split_text(text)

    def _group_into_chunks(self, sentences: List[str]) -> List[str]:
        """Phase 2: Group sentences into chunks of the target token size."""
        chunks = []
        current_chunk_tokens = 0
        current_chunk: List[str] = []

        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            if current_chunk_tokens + sentence_tokens <= self.config.hard_ceiling:
                current_chunk.append(sentence)
                current_chunk_tokens += sentence_tokens
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_chunk_tokens = sentence_tokens
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def _apply_hard_ceiling(self, chunks: List[str]) -> List[str]:
        """Phase 3: Apply a hard token ceiling to each chunk."""
        final_chunks = []
        for chunk in chunks:
            if len(chunk.split()) > self.config.hard_ceiling:
                final_chunks.extend(self.recursive_splitter.split_text(chunk))
            else:
                final_chunks.append(chunk)
        return final_chunks

    def chunk(
        self, text: str, ticker: str, fiscal_year: int, section: str
    ) -> List[Chunk]:
        """Executes the three-phase chunking process.

        Parameters
        ----------
        text : str
            The input text to be chunked.
        ticker : str
            The stock ticker symbol.
        fiscal_year : int
            The fiscal year.
        section : str
            The document section.

        Returns
        -------
        List[Chunk]
            A list of Chunk objects.
        """
        sentences = self._split_by_sentences(text)
        grouped_chunks = self._group_into_chunks(sentences)
        final_chunks_text = self._apply_hard_ceiling(grouped_chunks)

        chunk_objects = []
        start_index = 0
        for chunk_text in final_chunks_text:
            end_index = start_index + len(chunk_text)
            id = str(uuid.uuid5(NAMESPACE, chunk_text))
            tokens = len(chunk_text.split())
            chunk_objects.append(
                Chunk(
                    id=id,
                    text=chunk_text,
                    ticker=ticker,
                    fiscal_year=fiscal_year,
                    section=section,
                    tokens=tokens,
                    start_index=start_index,
                    end_index=end_index,
                )
            )
            start_index = end_index

        return chunk_objects

    def chunk_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        """Chunks all documents in a DataFrame.

        This method iterates through a DataFrame of SEC filings, chunks each
        document, and returns a new DataFrame of chunks along with statistics.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame with columns for 'ticker', 'fiscal_year', and 'text'.

        Returns
        -------
        tuple[pd.DataFrame, dict[str, int]]
            A tuple containing a DataFrame of chunks and a dictionary of chunking statistics.
        """
        all_chunks: List[Chunk] = []
        for _, row in df.iterrows():
            doc_chunks = self.chunk(
                text=row["text"],
                ticker=row["ticker"],
                fiscal_year=row["fiscal_year"],
                section=row["section"],
            )
            all_chunks.extend(doc_chunks)

        chunk_df = pd.DataFrame([chunk.dict() for chunk in all_chunks])

        stats = {
            "total_documents": len(df),
            "total_chunks": len(chunk_df),
            "avg_tokens_per_chunk": int(chunk_df["tokens"].mean()),
            "median_tokens_per_chunk": int(chunk_df["tokens"].median()),
            "min_tokens_per_chunk": int(chunk_df["tokens"].min()),
            "max_tokens_per_chunk": int(chunk_df["tokens"].max()),
        }
        return chunk_df, stats


def chunk_document(
    document: Dict[str, Any], config: Optional[ChunkingConfig] = None
) -> List[Dict[str, Any]]:
    """Chunks a single document dictionary.

    Parameters
    ----------
    document : Dict[str, Any]
        A dictionary representing a single document.
    config : Optional[ChunkingConfig], optional
        Chunking configuration, by default None.

    Returns
    -------
    List[Dict[str, Any]]
        A list of chunk dictionaries.
    """
    chunker = Chunker(config)
    chunks = chunker.chunk(
        text=document["text"],
        ticker=document["ticker"],
        fiscal_year=document["fiscal_year"],
        section=document["section"],
    )
    return [chunk.dict() for chunk in chunks]


def get_chunking_configs() -> List[ChunkingConfig]:
    """Returns a predefined list of chunking configurations for comparison."""
    return [
        ChunkingConfig(target_tokens=150, overlap_tokens=50, hard_ceiling=500),  # Small
        ChunkingConfig(
            target_tokens=350, overlap_tokens=100, hard_ceiling=800
        ),  # Medium
        ChunkingConfig(
            target_tokens=500, overlap_tokens=100, hard_ceiling=800
        ),  # Large
        ChunkingConfig(
            target_tokens=750, overlap_tokens=150, hard_ceiling=1000
        ),  # X-Large
    ]


def get_sections_to_chunk() -> List[str]:
    """Returns the list of SEC sections to be chunked."""
    # Return all sections since they should all be chunkable
    return list(SEC_10K_SECTIONS.keys())
