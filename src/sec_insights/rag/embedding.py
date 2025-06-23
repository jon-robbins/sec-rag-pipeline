"""
Handles OpenAI embeddings with batching, retry logic, and L2-normalization.
"""

import logging
from typing import List

import numpy as np
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .client_utils import get_openai_client
from .config import EMBEDDING_MODEL_NAME, MAX_RETRIES

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Handles OpenAI embeddings with batching, retry logic, and L2-normalization."""

    def __init__(self):
        """Initializes the EmbeddingManager."""
        self.client = get_openai_client()
        self.model = EMBEDDING_MODEL_NAME

    def _normalize_embeddings(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Normalize the embeddings to unit length."""
        out: List[List[float]] = []
        for emb in embeddings:
            vec = np.asarray(emb, dtype="float32")
            norm = float(np.linalg.norm(vec)) or 1.0
            out.append((vec / norm).tolist())
        return out

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(MAX_RETRIES),
    )
    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        """Embed a single batch of texts."""
        response = self.client.embeddings.create(input=batch, model=self.model)
        return [item.embedding for item in response.data]

    def embed_texts_in_batches(
        self, texts: List[str], batch_size: int = 256
    ) -> List[List[float]]:
        """
        Embeds texts in batches and returns the normalized embeddings.
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                batch_embeddings = self._embed_batch(batch)
                normalized_embeddings = self._normalize_embeddings(batch_embeddings)
                all_embeddings.extend(normalized_embeddings)
            except Exception as e:
                logger.error(f"Error embedding batch {i // batch_size}: {e}")
                # Fill with zero vectors for the failed batch
                all_embeddings.extend([[0.0] * 1536 for _ in batch])
        return all_embeddings
