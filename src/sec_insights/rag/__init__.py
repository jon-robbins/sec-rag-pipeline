"""
RAG package for the SEC Insights project.
"""

from .chunkers import Chunk, SmartChunker
from .document_store import DocumentStore
from .embedding import EmbeddingManager
from .generation import AnswerGenerator
from .parser import QueryParser
from .pipeline import RAGPipeline
from .reranker import BGEReranker
from .vector_store import VectorStore

__all__ = [
    "RAGPipeline",
    "VectorStore",
    "Chunk",
    "SmartChunker",
    "DocumentStore",
    "EmbeddingManager",
    "AnswerGenerator",
    "QueryParser",
    "BGEReranker",
]

__version__ = "1.0.0"
