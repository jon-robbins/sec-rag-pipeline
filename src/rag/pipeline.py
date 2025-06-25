import logging
from typing import Any, Dict, List

from src.openai_functions.answer_question import AnswerGenerator
from src.vector_store.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    A lean RAG pipeline that orchestrates vector search and answer generation.
    """

    def __init__(self, vector_store: VectorStore, answer_generator: AnswerGenerator):
        """
        Initializes the pipeline with a vector store and an answer generator.
        """
        self.vector_store = vector_store
        self.answer_generator = answer_generator
        logger.info("RAGPipeline initialized.")

    def search(
        self,
        query: str,
        top_k: int = 10,
        ticker: str | None = None,
        fiscal_year: int | None = None,
        sections: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Embed the query and retrieve top_k chunks.

        Args:
            query: Natural-language question.
            top_k: Number of chunks to retrieve.
            ticker: Optional ticker filter.
            fiscal_year: Optional year filter.
            sections: Optional list of section codes.

        Returns:
            List of Qdrant search result dicts.
        """
        logger.debug("🔍 RAGPipeline.search: '%s' (top_k=%d)", query, top_k)

        # 1️⃣ Embed the query using the same EmbeddingManager as the vector store
        query_vector = self.vector_store.embedding_manager.embed_texts_in_batches(
            [query]
        )[0]

        # 2️⃣ Perform vector search with optional metadata filtering
        return self.vector_store.search(
            query_vector=query_vector,
            ticker=ticker,
            fiscal_year=fiscal_year,
            sections=sections,
            top_k=top_k,
        )

    def generate_answer(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """
        Generates an answer using the provided chunks.
        """
        return self.answer_generator.generate_answer(question=question, chunks=chunks)
