"""
Search functionality for the SEC Vector Store.
"""

from typing import List, Dict, Any, Optional, Union

from qdrant_client import models

from .config import VectorStoreConfig


class SearchManager:
    """Handles vector search operations with filtering."""
    
    def __init__(self, client, config: VectorStoreConfig, embedding_manager=None, query_parser=None):
        self.client = client
        self.config = config
        self.embedding_manager = embedding_manager
        self.query_parser = query_parser
    
    def search(
        self,
        query: str,
        *,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        sections: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents with optional filtering.
        
        Args:
            query: Search query string
            ticker: Optional ticker symbol filter
            fiscal_year: Optional fiscal year filter
            sections: Optional SEC sections filter
            top_k: Number of results to return
            
        Returns:
            List of search results with scores and metadata
        """
        # Auto-extract constraints from query if not provided
        if ticker is None and fiscal_year is None and sections is None:
            if self.query_parser:
                parsed = self.query_parser.parse_query(query)
                ticker = parsed.get("ticker")
                fiscal_year = parsed.get("fiscal_year")
                sections = parsed.get("sections")

        # Build query filter
        query_filter = self._build_filter(ticker, fiscal_year, sections)

        # Get query vector
        if self.embedding_manager is None:
            raise ValueError("EmbeddingManager required for search")
        
        query_vector = self.embedding_manager.embed_texts([query])[0]

        # Execute search
        result = self.client.query_points(
            collection_name=self.config.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        # Normalize result format
        if isinstance(result, tuple):  # ("points", list)
            result = result[1]
        elif hasattr(result, "points"):  # QueryResponse
            result = result.points

        # Format results
        return [
            {
                "score": point.score,
                **point.payload,
                "text": point.payload.get("text", ""),
            }
            for point in result
        ]
    
    def retrieve_by_filter(
        self,
        *,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        sections: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents based on metadata filters only, without vector search.
        
        Args:
            ticker: Optional ticker symbol filter
            fiscal_year: Optional fiscal year filter
            sections: Optional SEC sections filter
            limit: Number of results to return
            
        Returns:
            List of results with metadata
        """
        query_filter = self._build_filter(ticker, fiscal_year, sections)

        # Execute scroll to retrieve by filter
        points, _ = self.client.scroll(
            collection_name=self.config.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        # Format results
        return [
            {
                **point.payload,
                "text": point.payload.get("text", ""),
            }
            for point in points
        ]
    
    def _build_filter(
        self, 
        ticker: Optional[str] = None, 
        fiscal_year: Optional[int] = None, 
        sections: Optional[List[str]] = None
    ) -> Optional[models.Filter]:
        """
        Build Qdrant filter from search parameters.
        
        Args:
            ticker: Ticker symbol filter
            fiscal_year: Fiscal year filter
            sections: SEC sections filter
            
        Returns:
            Qdrant Filter object or None if no filters
        """
        if not any([ticker, fiscal_year, sections]):
            return None

        conditions: List[models.FieldCondition] = []

        if ticker:
            conditions.append(
                models.FieldCondition(
                    key="ticker",
                    match=models.MatchValue(value=ticker.upper()),
                )
            )

        if fiscal_year:
            conditions.append(
                models.FieldCondition(
                    key="fiscal_year",
                    match=models.MatchValue(value=int(fiscal_year)),
                )
            )

        if sections:
            conditions.append(
                models.FieldCondition(
                    key="item",
                    match=models.MatchAny(any=sections),
                )
            )

        return models.Filter(must=conditions) if conditions else None 