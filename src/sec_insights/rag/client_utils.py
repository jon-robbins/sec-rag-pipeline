"""
Utilities for creating API clients.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

# Load .env file if it exists
load_dotenv()

_openai_client = None


def get_openai_client() -> OpenAI:
    """
    Returns a singleton instance of the OpenAI client.

    Initializes the client with the OPENAI_API_KEY from environment variables.
    """
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client
