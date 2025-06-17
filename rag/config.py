"""
Configuration and constants for the SEC Vector Store.
"""

import os
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


@dataclass
class VectorStoreConfig:
    """Configuration for VectorStore instances."""
    collection_name: str = "sec_filings"
    dim: int = 1536  # text-embedding-3-small
    model: str = "text-embedding-3-small"
    openai_key: Optional[str] = None
    use_docker: bool = False
    docker_host: str = "localhost"
    docker_port: int = 6333
    auto_fallback_to_memory: bool = True
    docker_timeout: int = 120  # seconds
    docker_batch_size: int = 500  # smaller batches for Docker
    memory_batch_size: int = 1024  # larger batches for memory


# Constants
DEFAULT_OPENAI_KEY = os.getenv("OPENAI_KEY")
MAX_TOKENS_PER_BATCH = 150_000  # Reduced for safety
MAX_RETRIES = 3

#openai pricing
PRICING = {
    "gpt-4o-mini": {"input": 0.15/1_000_000, "output": 0.60/1_000_000},
    "text-embedding-3-small": {"input": 0.02/1_000_000, "output": 0.0}
}