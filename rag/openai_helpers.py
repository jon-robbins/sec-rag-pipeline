"""
Collection of helper functions for interacting with the OpenAI API.
"""

import backoff
import openai
from openai import OpenAIError
from typing import Callable, Any, List, Dict


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
