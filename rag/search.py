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
        query: Optional[str] = None,
        *,
        ticker: Optional[str] = None,
        fiscal_year: Optional[int] = None,
        sections: Optional[List[str]] = None,
        top_k: int = 10,
        query_vector: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents using a query string and/or metadata filters.
        """
        query_vector = None  # Initialize to None
        
        parsed_query = self.query_parser.parse_query(query or "")
        
        # Auto-extract constraints from query if not provided explicitly
        ticker = ticker or parsed_query.get("ticker")
        fiscal_year = fiscal_year or parsed_query.get("fiscal_year")
        sections = sections or parsed_query.get("sections")
        
        # Build query filter
        query_filter = self._build_filter(ticker, fiscal_year, sections)
        
        # Embed the query
        if query_vector is None:
            if not query:
                raise ValueError("Either `query` or `query_vector` must be provided.")
            
            # Use the batch embedding method for consistency
            query_vector = self.embedding_manager.embed_texts_in_batches([query])[0]
        
        # Execute search
        results = self.client.search(
            collection_name=self.config.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
        )
        
        return [result.model_dump() for result in results]
    
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
                    key="section",
                    match=models.MatchAny(any=sections),
                )
            )

        return models.Filter(must=conditions) if conditions else None 