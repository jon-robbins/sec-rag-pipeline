"""
Handles OpenAI embeddings with batching, retry logic, and L2-normalization.
"""

import logging
from typing import List

import numpy as np
from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.utils.client_utils import get_openai_client
from src.utils.config import EMBEDDING_MODEL_NAME, MAX_RETRIES

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
        try:
            response = self.client.embeddings.create(input=batch, model=self.model)
            return [item.embedding for item in response.data]
        except Exception as e:
            # Log more details about the problematic batch
            logger.error(f"Embedding API error: {e}")
            logger.error(f"Batch size: {len(batch)}")
            for i, text in enumerate(batch[:3]):  # Show first 3 texts
                logger.error(f"  Text {i}: length={len(text)}, preview='{text[:100]}'")
            raise

    def _clean_text(self, text: str) -> str:
        """Clean text to prevent API errors."""
        if not text or not text.strip():
            return "empty text"

        # Remove excessive whitespace and normalize
        cleaned = " ".join(text.split())

        # Truncate if too long (OpenAI embedding models have ~8192 token limit)
        # Rough estimate: 1 token ≈ 4 characters
        max_chars = 8192 * 4  # Conservative estimate
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars]
            logger.warning(
                f"Truncated text that was {len(text)} chars to {len(cleaned)} chars"
            )

        return cleaned

    def embed_texts_in_batches(
        self, texts: List[str], batch_size: int = 100
    ) -> List[List[float]]:
        """
        Embeds texts in batches and returns the normalized embeddings.
        """
        logger.info(f"Generating embeddings for {len(texts)} chunks...")

        all_embeddings = []
        failed_batches = 0
        failed_individual_texts = 0

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size

            # Clean all texts in batch
            cleaned_batch = [self._clean_text(text) for text in batch]

            try:
                batch_embeddings = self._embed_batch(cleaned_batch)
                normalized_embeddings = self._normalize_embeddings(batch_embeddings)
                all_embeddings.extend(normalized_embeddings)

                if batch_num % 10 == 0:
                    logger.info(
                        f"✅ Processed batch {batch_num}/{(len(texts) - 1) // batch_size}"
                    )

            except Exception as e:
                failed_batches += 1
                logger.error(f"❌ Batch {batch_num} failed: {e}")
                logger.info(f"🔄 Retrying {len(batch)} texts individually...")

                # Try each text individually to isolate failures
                batch_embeddings = []
                for j, text in enumerate(cleaned_batch):
                    try:
                        individual_embedding = self._embed_batch([text])
                        normalized_embedding = self._normalize_embeddings(
                            individual_embedding
                        )
                        batch_embeddings.extend(normalized_embedding)
                    except Exception as individual_error:
                        failed_individual_texts += 1
                        logger.warning(
                            f"❌ Individual text failed: '{text[:50]}...', error: {individual_error}"
                        )
                        # Only this specific text gets zero embedding
                        batch_embeddings.append([0.0] * 1536)

                all_embeddings.extend(batch_embeddings)

        if failed_batches > 0:
            logger.warning(f"⚠️ {failed_batches} batches failed, retried individually")
        if failed_individual_texts > 0:
            logger.warning(
                f"⚠️ {failed_individual_texts} individual texts failed (filled with zeros)"
            )

        logger.info(f"✅ Generated embeddings for {len(all_embeddings)} chunks total")
        return all_embeddings
