import logging
from typing import List, Tuple

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class BGEReranker:
    """
    A wrapper for the BAAI/bge-reranker-base model.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        logger.info("Initializing BGE Reranker with model: %s", model_name)
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, passages: List[str], top_k: int
    ) -> List[Tuple[int, float]]:
        """
        Reranks a list of passages based on a query.

        Args:
            query: The search query.
            passages: A list of passages to rerank.
            top_k: The number of results to return.

        Returns:
            A list of tuples, each containing the index and score of a passage.
        """
        scores = self.model.predict([(query, p) for p in passages])
        sorted_results = sorted(
            enumerate(scores), key=lambda item: item[1], reverse=True
        )
        return sorted_results[:top_k]
