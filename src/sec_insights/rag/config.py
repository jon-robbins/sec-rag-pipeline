"""
Configuration and constants for the SEC Vector Store.
"""

import os
from dataclasses import dataclass
from pathlib import Path
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

# openai pricing
PRICING_PER_TOKEN = {
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "text-embedding-3-small": {"input": 0.02 / 1_000_000, "output": 0.0},
}
PRICING_PER_CALL = {
    "gpt-4o-mini-search-preview": 27.5 / 1_000,
}

# --- Directories ---
# These are now constructed dynamically in the relevant classes
# to avoid pathing issues related to the src layout.
RESULTS_DIR = Path.cwd() / "data" / "results"
CACHE_DIR = Path.cwd() / "data" / "cache"

# --- Files ---
QA_DATASET_PATH = Path.cwd() / "data" / "processed" / "qa_dataset.jsonl"
EVAL_RESULTS_PATH = Path.cwd() / "evaluation_results.json"

# --- Vector Store ---
COLLECTION_NAME = "sec_filings_2"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
# Check if running in a Docker container
IN_DOCKER = os.getenv("IN_DOCKER") == "true"
QDRANT_HOST = "qdrant" if IN_DOCKER else "localhost"
QDRANT_PORT = 6333

# --- Models ---

MAX_CONTEXT_LENGTH = 8192

# --- Section metadata ---
# ──────────────────────────────── SEC mapping ────────────────────────────────
SEC_10K_SECTIONS = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity, Related Stockholder Matters "
    "and Issuer Purchases of Equity Securities",
    "6": "Selected Financial Data",
    "7": "Management's Discussion and Analysis of Financial Condition and "
    "Results of Operations (MD&A)",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
    "9": "Changes in and Disagreements with Accountants on Accounting and "
    "Financial Disclosure",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and "
    "Related Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director "
    "Independence",
    "14": "Principal Accounting Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
}
