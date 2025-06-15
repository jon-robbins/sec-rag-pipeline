"""
OpenAI API helpers with retry logic and usage tracking.
"""

import time
import random
from typing import Dict, Any, List, Callable
from dataclasses import dataclass, field


def retry_openai_call(func: Callable, max_retries: int = 3, **kwargs) -> Any:
    """
    Retry OpenAI API calls with exponential backoff.
    
    Args:
        func: OpenAI API function to call
        max_retries: Maximum number of retries
        **kwargs: Arguments to pass to the API function
        
    Returns:
        API response object
    """
    for attempt in range(max_retries):
        try:
            return func(**kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            
            # Exponential backoff with jitter
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
            print(f"Retrying in {wait_time:.1f} seconds...")
            time.sleep(wait_time)


@dataclass
class UsageStats:
    """Tracks usage statistics for different API operations."""
    parsing_prompt_tokens: int = 0
    parsing_completion_tokens: int = 0
    embedding_tokens: int = 0
    generation_prompt_tokens: int = 0
    generation_completion_tokens: int = 0
    
    @property
    def total_tokens(self) -> int:
        return (self.parsing_prompt_tokens + self.parsing_completion_tokens + 
                self.embedding_tokens + self.generation_prompt_tokens + 
                self.generation_completion_tokens)


class UsageCostCalculator:
    """Centralized token usage and cost calculation."""
    
    # OpenAI Pricing (USD per 1K tokens) - Updated December 2024
    PRICING = {
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015},
        "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
        "text-embedding-3-large": {"input": 0.00013, "output": 0.0}
    }
    
    def __init__(self, default_model: str = "gpt-4o-mini"):
        self.default_model = default_model
    
    def extract_usage_from_response(self, response: Any, operation_type: str) -> Dict[str, int]:
        """
        Extract usage information from OpenAI API response.
        
        Args:
            response: OpenAI API response object
            operation_type: Type of operation ("parsing", "generation", "embedding")
            
        Returns:
            Dictionary with usage information
        """
        if not hasattr(response, 'usage'):
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        usage = response.usage
        
        if operation_type == "embedding":
            # Embeddings only have total_tokens
            return {
                "prompt_tokens": getattr(usage, 'prompt_tokens', usage.total_tokens),
                "completion_tokens": 0,
                "total_tokens": usage.total_tokens
            }
        else:
            # Chat completions have prompt_tokens and completion_tokens
            return {
                "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
                "completion_tokens": getattr(usage, 'completion_tokens', 0),
                "total_tokens": usage.total_tokens
            }
    
    def calculate_cost(self, usage: Dict[str, int], model: str = None) -> float:
        """
        Calculate cost based on token usage and model pricing.
        
        Args:
            usage: Dictionary with token usage (prompt_tokens, completion_tokens)
            model: Model name (uses default if not provided)
            
        Returns:
            Cost in USD
        """
        model = model or self.default_model
        
        if model not in self.PRICING:
            print(f"Warning: Unknown model {model}, using default pricing")
            model = self.default_model
        
        pricing = self.PRICING[model]
        
        prompt_cost = usage.get("prompt_tokens", 0) * pricing["input"] / 1000
        completion_cost = usage.get("completion_tokens", 0) * pricing["output"] / 1000
        
        return prompt_cost + completion_cost
    
    def summarize(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Summarize token usage and costs across multiple evaluation results.
        
        Args:
            results: List of evaluation results from SECQueryEvaluator
            
        Returns:
            Summary statistics
        """
        if not results:
            return {
                "model": self.default_model,
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_cost_per_query": 0.0,
                "breakdown": {}
            }
        
        total_cost = sum(r.get("cost_breakdown", {}).get("total", 0) for r in results)
        total_tokens = sum(r.get("token_usage", {}).get("total", 0) for r in results)
        
        # Calculate breakdown by operation type
        breakdown = {
            "parsing": sum(r.get("cost_breakdown", {}).get("parsing", 0) for r in results),
            "embedding": sum(r.get("cost_breakdown", {}).get("embedding", 0) for r in results),
            "generation": sum(r.get("cost_breakdown", {}).get("generation", 0) for r in results)
        }
        
        return {
            "model": self.default_model,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "avg_cost_per_query": total_cost / len(results) if results else 0,
            "breakdown": breakdown
        }
