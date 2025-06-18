"""
Configuration and constants for the SEC Vector Store.
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

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

# --- Directories ---
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"  # For embeddings and other caches
DB_DIR = CACHE_DIR / "database"
RESULTS_DIR = DATA_DIR / "results"

# Create directories if they don't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)


# --- Files ---
RAW_DATA_PATH = RAW_DATA_DIR / "df_filings_full.parquet"
QA_DATASET_PATH = PROCESSED_DATA_DIR / "qa_dataset.jsonl"
EVAL_RESULTS_PATH = ROOT_DIR / "evaluation_results.json"

# --- Vector Store ---
VECTOR_DB_PATH = str(DB_DIR)
COLLECTION_NAME = "sec_filings_2"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
# Check if running in a Docker container
IN_DOCKER = os.getenv("IN_DOCKER") == "true"
QDRANT_HOST = "qdrant" if IN_DOCKER else "localhost"
QDRANT_PORT = 6333

# --- Models ---
GENERATION_MODEL_NAME = "gpt-3.5-turbo-0125"
# MAX_CONTEXT_LENGTH = 16385 # For gpt-3.5-turbo-0125
# Update: Use a smaller context length to be safe
MAX_CONTEXT_LENGTH = 8192

# --- Evaluation ---
EVAL_NUM_QUESTIONS = 50
EVAL_TARGET_LLM = "gpt-4-0125-preview"
EVAL_JUDGE_LLM = "gpt-4-turbo-preview"

# --- Chunking ---
CHUNK_TARGET_TOKENS = 350
CHUNK_OVERLAP_TOKENS = 50
CHUNK_HARD_CEILING = CHUNK_TARGET_TOKENS + 300