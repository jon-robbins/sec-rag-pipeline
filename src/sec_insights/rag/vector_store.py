"""
A pure data access layer for interacting with a Qdrant vector database.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient, models

from .config import VectorStoreConfig
from .embedding import EmbeddingManager

logger = logging.getLogger(__name__)


class VectorStore:
    """A wrapper for the Qdrant client that simplifies all vector database interactions.

    This class is responsible for all direct communication with the Qdrant
    database, including setting up connections, managing collections, and
    performing search and upsert operations. It is a data access layer and
    contains no business logic.

    Parameters
    ----------
    use_docker : bool, optional
        Whether to connect to a Docker-based Qdrant instance, by default False.
    collection_name : str, optional
        The name of the collection to use, by default "sec_filings".
    embedding_manager : EmbeddingManager
        The embedding manager instance to use for configuration.
    **kwargs
        Additional keyword arguments for the `QdrantClient`.
    """

    def __init__(
        self,
        *,
        use_docker: bool = False,
        collection_name: str = "sec_filings",
        embedding_manager: EmbeddingManager,
        **kwargs: Any,
    ) -> None:
        self.client = self._setup_client(use_docker=use_docker, **kwargs)
        self.config = VectorStoreConfig(collection_name=collection_name)
        self.embedding_manager = embedding_manager

        self.config.use_docker = use_docker
        if use_docker:
            self.config.docker_host = kwargs.get("host", "localhost")
            self.config.docker_port = kwargs.get("port", 6333)

        self._init_collection()

    def _setup_client(self, use_docker: bool, **kwargs: Any) -> QdrantClient:
        """Set up the Qdrant client based on configuration."""
        if use_docker:
            try:
                # Add timeout settings for Docker
                kwargs.setdefault("timeout", 300)
                client = QdrantClient(**kwargs)
                client.get_collections()
                logger.info("Docker Qdrant connection successful.")
                return client
            except Exception as e:
                logger.warning("Docker Qdrant connection failed: %s", e)
                if self.config.auto_fallback_to_memory:
                    logger.warning("Automatically falling back to in-memory Qdrant.")
                    self.config.use_docker = False
                    return QdrantClient(":memory:")
                else:
                    raise

        logger.info("Using in-memory Qdrant")
        return QdrantClient(":memory:")

    def search(
        self,
        query_vector: List[float],
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        sections: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Performs a vector search with optional metadata filtering.

        Parameters
        ----------
        query_vector : List[float]
            The vector embedding of the search query.
        ticker : Optional[str], optional
            A ticker symbol to filter on, by default None.
        fiscal_year : Optional[int], optional
            A fiscal year to filter on, by default None.
        sections : Optional[List[str]], optional
            A list of SEC section codes to filter on, by default None.
        top_k : int, optional
            The number of top results to return, by default 10.

        Returns
        -------
        List[Dict[str, Any]]
            A list of search results from Qdrant.
        """
        query_filter = self._build_filter(ticker, fiscal_year, sections)
        results = self.client.search(
            collection_name=self.config.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
        )
        return [result.model_dump() for result in results]

    def retrieve_by_filter(
        self,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        sections: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieves documents based on metadata filters only (no vector search).

        Parameters
        ----------
        ticker : Optional[str], optional
            A ticker symbol to filter on, by default None.
        fiscal_year : Optional[int], optional
            A fiscal year to filter on, by default None.
        sections : Optional[List[str]], optional
            A list of SEC section codes to filter on, by default None.
        limit : int, optional
            The maximum number of documents to return, by default 100.

        Returns
        -------
        List[Dict[str, Any]]
            A list of retrieved documents, including metadata and text.
        """
        query_filter = self._build_filter(ticker, fiscal_year, sections)
        points, _ = self.client.scroll(
            collection_name=self.config.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [point.model_dump()["payload"] for point in points]

    def _build_filter(
        self,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        sections: Optional[List[str]] = None,
    ) -> Optional[models.Filter]:
        """Build Qdrant filter from search parameters."""
        if not any([ticker, fiscal_year, sections]):
            return None
        conditions: List[models.FieldCondition] = []
        if ticker:
            conditions.append(
                models.FieldCondition(
                    key="ticker", match=models.MatchValue(value=ticker.upper())
                )
            )
        if fiscal_year:
            conditions.append(
                models.FieldCondition(
                    key="fiscal_year", match=models.MatchValue(value=int(fiscal_year))
                )
            )
        if sections:
            conditions.append(
                models.FieldCondition(
                    key="section", match=models.MatchAny(any=sections)
                )
            )
        return models.Filter(must=conditions) if conditions else None

    def upsert_chunks(
        self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]
    ) -> None:
        """Upserts a batch of chunks and their embeddings into the collection."""
        if len(chunks) != len(embeddings):
            raise ValueError("The number of chunks and embeddings must be the same.")

        # Use smaller batches for Docker to prevent timeouts
        batch_size = (
            self.config.docker_batch_size
            if self.config.use_docker
            else self.config.memory_batch_size
        )
        total_chunks = len(chunks)

        if total_chunks > batch_size:
            logger.info(
                "Processing %d chunks in batches of %d", total_chunks, batch_size
            )
            for i in range(0, total_chunks, batch_size):
                end_idx = min(i + batch_size, total_chunks)
                batch_chunks = chunks[i:end_idx]
                batch_embeddings = embeddings[i:end_idx]
                logger.info(
                    "Upserting batch %d/%d (%d chunks)",
                    i // batch_size + 1,
                    (total_chunks + batch_size - 1) // batch_size,
                    end_idx - i,
                )
                self._upsert_batch(batch_chunks, batch_embeddings)
        else:
            self._upsert_batch(chunks, embeddings)

    def _upsert_batch(
        self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]
    ) -> None:
        """Upserts a single batch of chunks."""
        ids = [chunk["id"] for chunk in chunks]
        payloads = [{**chunk["metadata"], "text": chunk["text"]} for chunk in chunks]

        try:
            self.client.upsert(
                collection_name=self.config.collection_name,
                points=models.Batch(ids=ids, vectors=embeddings, payloads=payloads),
                wait=True,
            )
        except Exception as e:
            logger.warning("Upsert failed: %s", e)
            if self.config.use_docker and self.config.auto_fallback_to_memory:
                logger.warning(
                    "Automatically falling back to in-memory Qdrant for upsert."
                )
                self.config.use_docker = False
                self.client = QdrantClient(":memory:")
                self._recreate_collection()
                # Retry the upsert with the new in-memory client
                self.client.upsert(
                    collection_name=self.config.collection_name,
                    points=models.Batch(ids=ids, vectors=embeddings, payloads=payloads),
                    wait=True,
                )
            else:
                raise

    def get_status(self) -> Dict[str, Any]:
        """Gets the current status of the collection.

        Returns
        -------
        Dict[str, Any]
            A dictionary with the collection name and number of points.
        """
        try:
            info = self.client.get_collection(self.config.collection_name)
            return {
                "collection_name": self.config.collection_name,
                "points_count": info.points_count,
            }
        except Exception:
            return {"collection_name": self.config.collection_name, "points_count": 0}

    def _init_collection(self) -> None:
        """(Re)create the collection if it doesn't exist or if dimensions change."""
        try:
            collection_info = self.client.get_collection(self.config.collection_name)
            if collection_info.vectors_config.params.size != self.config.dim:
                self._recreate_collection()
        except Exception:
            self._recreate_collection()

    def _recreate_collection(self) -> None:
        """Deletes and recreates the collection with the current configuration."""
        logger.info(f"Recreating collection '{self.config.collection_name}'...")
        self.client.recreate_collection(
            collection_name=self.config.collection_name,
            vectors_config=models.VectorParams(
                size=self.config.dim,
                distance=models.Distance.COSINE,
            ),
        )
