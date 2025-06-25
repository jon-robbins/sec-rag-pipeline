"""
Utilities for creating API clients.
"""

import os
from typing import Any, Callable

import backoff
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# Load .env file if it exists
load_dotenv()

_openai_client = None


# Decorator for retrying OpenAI API calls with exponential backoff
@backoff.on_exception(backoff.expo, OpenAIError, max_tries=5, factor=1.5)
def retry_openai_call(api_call: Callable[..., Any], **kwargs) -> Any:
    """
    Retries an OpenAI API call with exponential backoff.

    Args:
        api_call: The OpenAI API function to call.
        **kwargs: Arguments to pass to the API call.

    Returns:
        The result of the API call.
    """
    return api_call(**kwargs)


def get_openai_client() -> OpenAI:
    """
    Returns an OpenAI client, caching it for reuse.
    Expects the OPENAI_API_KEY environment variable to be set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return OpenAI(api_key=api_key)
