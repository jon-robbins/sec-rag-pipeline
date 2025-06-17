"""
Embedding utilities for the SEC Vector Store.
"""

from typing import Iterable, List, Tuple, Any
import numpy as np
import tiktoken
import openai

from .config import MAX_TOKENS_PER_BATCH, DEFAULT_OPENAI_KEY
from .openai_helpers import retry_openai_call


class EmbeddingManager:
    """Handles OpenAI embeddings with batching, retry logic, and L2-normalization."""
    
    def __init__(self, model: str = "text-embedding-3-small", openai_key: str = None):
        self.model = model
        self.openai_client = openai.OpenAI(api_key=openai_key or DEFAULT_OPENAI_KEY)
        self._encoder = tiktoken.encoding_for_model(model)
    
    # ------------------------------------------------------------------
    # batching helpers
    # ------------------------------------------------------------------
    def batch_texts_by_tokens(
        self,
        texts: Iterable[str],
        max_tokens: int = MAX_TOKENS_PER_BATCH,
    ) -> Iterable[List[str]]:
        """Yield batches whose total token count stays below *max_tokens*."""
        buffer: list[str] = []
        token_sum = 0
        for text in texts:
            n_tok = len(self._encoder.encode(text))
            if buffer and token_sum + n_tok > max_tokens:
                yield buffer
                buffer, token_sum = [text], n_tok
            else:
                buffer.append(text)
                token_sum += n_tok
        if buffer:
            yield buffer

    # ------------------------------------------------------------------
    # embedding helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_vectors(vectors: List[list[float]]) -> List[List[float]]:
        arr = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        arr /= norms
        return arr.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts (single batch convenience)."""
        embeddings, _ = self.embed_texts_with_response(texts)
        return embeddings

    def embed_texts_with_response(
        self, texts: List[str]
    ) -> Tuple[List[List[float]], List[Any]]:
        """Return (embeddings, list_of_openai_response_objects)."""
        embeddings: list[list[float]] = []
        responses: list[Any] = []
        for batch in self.batch_texts_by_tokens(texts):
            response = retry_openai_call(
                self.openai_client.embeddings.create,
                input=batch,
                model=self.model,
            )
            embeddings.extend(r.embedding for r in response.data)
            responses.append(response)
        embeddings = self._normalize_vectors(embeddings)
        return embeddings, responses

    # Convenience for pipeline large datasets
    def embed_texts_in_batches(
        self, texts: List[str], *, batch_size: int = 100
    ) -> List[List[float]]:
        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            all_vecs.extend(self.embed_texts(chunk))
        return all_vecs 