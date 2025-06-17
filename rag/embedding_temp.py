"""
Embedding utilities for the SEC Vector Store.
"""

from typing import Iterable, List, Tuple, Any
import tiktoken
import openai

from .config import MAX_TOKENS_PER_BATCH, DEFAULT_OPENAI_KEY
from .openai_helpers import retry_openai_call


class EmbeddingManager:
    """Handles OpenAI embeddings with batching and token management."""
    
    def __init__(self, model: str = "text-embedding-3-small", openai_key: str = None):
        self.model = model
        self.openai_client = openai.OpenAI(api_key=openai_key or DEFAULT_OPENAI_KEY)
        self._encoder = tiktoken.encoding_for_model(model)
    
    def batch_texts_by_tokens(
        self, 
        texts: Iterable[str], 
        max_tokens: int = MAX_TOKENS_PER_BATCH
    ) -> Iterable[List[str]]:
        """
        Yield batches of texts that stay below `max_tokens`.
        
        Args:
            texts: Iterable of text strings to batch
            max_tokens: Maximum tokens per batch
            
        Yields:
            Lists of text strings that fit within token limit
        """
        buffer, token_sum = [], 0
        
        for text in texts:
            num_tokens = len(self._encoder.encode(text))
            
            if token_sum + num_tokens > max_tokens and buffer:
                yield buffer
                buffer, token_sum = [text], num_tokens
            else:
                buffer.append(text)
                token_sum += num_tokens
                
        if buffer:
            yield buffer
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Get OpenAI embeddings for a list of texts (batched, with retry).
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (lists of floats)
        """
        embeddings, _ = self.embed_texts_with_response(texts)
        return embeddings
    
    def embed_texts_with_response(self, texts: List[str]) -> Tuple[List[List[float]], List[Any]]:
        """
        Get OpenAI embeddings and return both embeddings and response objects.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            Tuple of (embedding_vectors, list_of_response_objects)
        """
        embeddings: List[List[float]] = []
        responses: List[Any] = []
        
        for batch in self.batch_texts_by_tokens(texts):
            response = retry_openai_call(
                self.openai_client.embeddings.create,
                input=batch,
                model=self.model,
            )
            embeddings.extend(r.embedding for r in response.data)
            responses.append(response)
            
        return embeddings, responses 