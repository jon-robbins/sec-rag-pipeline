"""
Evaluation class for the SEC Vector Store system.
Handles end-to-end query processing with centralized token and cost tracking.
"""

import os
import time
from typing import Dict, Any, Optional, List

from rag import VectorStore, create_vector_store
from rag.openai_helpers import UsageCostCalculator


class SECQueryEvaluator:
    """
    End-to-end evaluation system for SEC Vector Store queries.
    
    Features:
    - Connects to Docker or in-memory vector database
    - Parses natural language queries to extract metadata
    - Retrieves relevant context based on parsed parameters
    - Generates answers using only the provided context
    - Centralized token usage and cost tracking from API responses
    """
    
    def __init__(
        self,
        *,
        use_docker: bool = False,
        collection_name: str = "sec_filings",
        docker_host: str = "localhost",
        docker_port: int = 6333,
        openai_key: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the evaluator.
        
        Args:
            use_docker: Whether to use Docker Qdrant or in-memory
            collection_name: Name of the Qdrant collection
            docker_host: Docker host address
            docker_port: Docker port number
            openai_key: OpenAI API key (uses env var if not provided)
            verbose: Whether to print detailed progress information
        """
        self.verbose = verbose
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        
        if not self.openai_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var or pass openai_key parameter.")
        
        # Initialize cost calculator
        self.cost_calculator = UsageCostCalculator()
        
        # Initialize vector store
        if self.verbose:
            print(f"🔧 Initializing vector store ({'Docker' if use_docker else 'Memory'})...")
        
        self.vector_store = create_vector_store(
            use_docker=use_docker,
            collection_name=collection_name,
            docker_host=docker_host,
            docker_port=docker_port,
            openai_key=self.openai_key,
            auto_fallback_to_memory=True
        )
        
        # Check if collection exists and has data
        status = self.vector_store.get_status()
        if self.verbose:
            print(f"📊 Vector store status: {status}")
        
        if status.get("points_count", 0) == 0:
            print("⚠️  Warning: Vector store appears to be empty. Make sure data is loaded.")
    
    def evaluate_query(
        self,
        query: str,
        *,
        top_k: int = 10,
        max_chunks: int = 10,
        max_context_length: int = 8000
    ) -> Dict[str, Any]:
        """
        Evaluate a single query end-to-end with centralized usage tracking.
        
        Args:
            query: Natural language query
            top_k: Number of chunks to retrieve
            max_chunks: Maximum chunks to use for answer generation
            max_context_length: Maximum context length for GPT
            
        Returns:
            Dictionary with answer, metadata, token usage, and costs
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\n🔍 Processing query: '{query}'")
        
        try:
            # Step 1: Parse the query
            if self.verbose:
                print("1️⃣ Parsing query...")
            
            parsed_params, parsing_response = self.vector_store.query_parser.parse_query_with_response(query)
            
            if self.verbose:
                print(f"   Parsed parameters: {parsed_params}")
            
            # Step 2: Search for relevant chunks
            if self.verbose:
                print("2️⃣ Searching for relevant chunks...")
            
            # Get embeddings with response objects
            _, embedding_responses = self.vector_store.embedding_manager.embed_texts_with_response([query])
            
            # Perform search
            chunks = self.vector_store.search(
                query=query,
                ticker=parsed_params.get("ticker"),
                fiscal_year=parsed_params.get("fiscal_year"),
                sections=parsed_params.get("sections"),
                top_k=top_k
            )
            
            if self.verbose:
                print(f"   Found {len(chunks)} chunks")
                if chunks:
                    print(f"   Top score: {chunks[0].get('score', 0):.3f}")
            
            # Step 3: Generate answer
            if self.verbose:
                print("3️⃣ Generating answer...")
            
            answer_result, generation_response = self.vector_store.answer_generator.generate_answer_with_response(
                question=query,
                chunks=chunks,
                max_chunks=max_chunks,
                max_context_length=max_context_length
            )
            
            if self.verbose:
                print(f"   Generated answer ({len(answer_result['answer'])} chars)")
            
            # Centralized usage and cost calculation
            usage_data = self._extract_all_usage(parsing_response, embedding_responses, generation_response)
            
            # Compile results
            total_time = time.time() - start_time
            
            result = {
                "query": query,
                "answer": answer_result["answer"],
                "sources": answer_result["sources"],
                "confidence": answer_result["confidence"],
                "chunks_used": answer_result["chunks_used"],
                "chunks_found": len(chunks),
                "parsed_params": parsed_params,
                "token_usage": usage_data["token_usage"],
                "cost_breakdown": usage_data["cost_breakdown"],
                "processing_time_seconds": total_time
            }
            
            if self.verbose:
                print(f"\n✅ Query processed successfully!")
                print(f"   Total tokens: {usage_data['token_usage']['total']}")
                print(f"   Total cost: ${usage_data['cost_breakdown']['total']:.4f}")
                print(f"   Processing time: {total_time:.2f}s")
            
            return result
            
        except Exception as e:
            error_result = {
                "query": query,
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "confidence": "error",
                "chunks_used": 0,
                "chunks_found": 0,
                "parsed_params": {},
                "token_usage": {"total": 0},
                "cost_breakdown": {"total": 0.0},
                "processing_time_seconds": time.time() - start_time,
                "error": str(e)
            }
            
            if self.verbose:
                print(f"❌ Error processing query: {e}")
            
            return error_result
    
    def _extract_all_usage(self, parsing_response, embedding_responses, generation_response) -> Dict[str, Any]:
        """Centrally extract all usage information and calculate costs."""
        
        # Extract usage from each response
        parsing_usage = self.cost_calculator.extract_usage_from_response(parsing_response, "parsing") if parsing_response else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        # Sum up embedding usage from all batches
        embedding_total_tokens = 0
        if embedding_responses:
            for resp in embedding_responses:
                embedding_usage = self.cost_calculator.extract_usage_from_response(resp, "embedding")
                embedding_total_tokens += embedding_usage["total_tokens"]
        
        generation_usage = self.cost_calculator.extract_usage_from_response(generation_response, "generation") if generation_response else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        # Calculate costs
        parsing_cost = self.cost_calculator.calculate_cost(parsing_usage, "gpt-4o-mini")
        embedding_cost = self.cost_calculator.calculate_cost({"prompt_tokens": embedding_total_tokens, "completion_tokens": 0}, "text-embedding-3-small")
        generation_cost = self.cost_calculator.calculate_cost(generation_usage, "gpt-4o-mini")
        
        return {
            "token_usage": {
                "parsing": {"input": parsing_usage["prompt_tokens"], "output": parsing_usage["completion_tokens"]},
                "embedding": embedding_total_tokens,
                "generation": {"input": generation_usage["prompt_tokens"], "output": generation_usage["completion_tokens"]},
                "total": parsing_usage["total_tokens"] + embedding_total_tokens + generation_usage["total_tokens"]
            },
            "cost_breakdown": {
                "parsing": parsing_cost,
                "embedding": embedding_cost,
                "generation": generation_cost,
                "total": parsing_cost + embedding_cost + generation_cost
            }
        }
    
    def get_database_status(self) -> Dict[str, Any]:
        """Get current status of the vector database."""
        return self.vector_store.get_status()


if __name__ == "__main__":
    from rag.openai_helpers import UsageCostCalculator
    evaluator = SECQueryEvaluator(use_docker=True, verbose=True)
    print(evaluator.get_database_status())

    test_queries = [
        "What were Tesla's main risks in 2020?",
        "How did Apple's revenue change in 2021?",
        "What competitors does Microsoft mention in their 2019 filing?",
    ]

    print("🚀 Running evaluation on test queries...\n")
    results = [evaluator.evaluate_query(q) for q in test_queries]


    # Calculate summary using the cost calculator
    calc = UsageCostCalculator()
    summary = calc.summarize(results)

    print("📊 EVALUATION SUMMARY")
    print(f"   Model: {summary['model']}")
    print(f"   Queries processed: {len(results)}")
    print(f"   Total tokens used: {summary['total_tokens']:,}")
    print(f"   Total cost: ${summary['total_cost']:.4f}")
    print(f"   Avg cost / query: ${summary['avg_cost_per_query']:.4f}")
    print("   Breakdown:")
    for part, cost in summary["breakdown"].items():
        print(f"     {part.capitalize():<10}: ${cost:.4f}")
