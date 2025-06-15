"""
SEC Vector Store - A modular vector database for SEC filings.
"""

from .vector_store import VectorStore, create_vector_store
from .config import VectorStoreConfig
from .generation import AnswerGenerator

# Main user-facing API
__all__ = [
    "VectorStore",
    "create_vector_store", 
    "VectorStoreConfig",
    "AnswerGenerator",
]

__version__ = "1.0.0" 