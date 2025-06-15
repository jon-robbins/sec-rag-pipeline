"""
Data processing utilities for the SEC Vector Store.

This module provides utilities for processing and chunking SEC filing documents.

Components:
- SmartChunker: Intelligent text chunking with overlap and token management
- FilingExploder: Converts structured filings into processable chunks
- TextCleaning: Text preprocessing and normalization utilities
"""

from .filing_exploder import FilingExploder
from .chunkers import SmartChunker, Chunk
from .text_cleaning import remove_boilerplate

__all__ = [
    "FilingExploder",
    "SmartChunker",
    "Chunk",
    "remove_boilerplate"
] 