"""
OpenAI helper functions for the SEC Vector Store.
"""

import time
import openai


def retry_openai_call(call, *args, **kwargs):
    """
    Retry helper for transient OpenAI errors / rate-limits.
    
    Args:
        call: The OpenAI API function to call
        *args: Positional arguments for the API call
        **kwargs: Keyword arguments for the API call
        
    Returns:
        The result of the successful API call
        
    Raises:
        RuntimeError: If all retries failed
    """
    for attempt in range(5):
        try:
            return call(*args, **kwargs)
        except openai.RateLimitError:
            time.sleep(2 ** attempt)  # 1s, 2s, 4s, 8s, 16s
        except openai.APIStatusError as e:  # 5xx server errors
            if 500 <= e.status_code < 600:
                time.sleep(2 ** attempt)
            else:
                raise
    raise RuntimeError("OpenAI request failed after 5 retries") 