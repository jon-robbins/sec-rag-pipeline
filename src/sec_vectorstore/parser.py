"""
Query parsing utilities for the SEC Vector Store.
"""

import json
from typing import Dict, Any

from .openai_helpers import retry_openai_call
from .config import DEFAULT_OPENAI_KEY
import openai


class QueryParser:
    """Handles parsing of natural language queries into structured parameters."""
    
    def __init__(self, openai_key: str = None):
        self.openai_client = openai.OpenAI(api_key=openai_key or DEFAULT_OPENAI_KEY)
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse a natural language query into ticker, fiscal_year, and sections.
        
        Args:
            query: Natural language query string
            
        Returns:
            Dictionary with parsed parameters: ticker, fiscal_year, sections
        """
        system_message = """
        ## ROLE
        You are a financial-query parser.
        Return only ONE JSON object, nothing else.

        ## OUTPUT KEYS
        "ticker"       – 2-5 letter US stock symbol (uppercase)
        "fiscal_year"  – 4-digit year 1995-2030
        "sections"     – array (≤3) of SEC 10-K items as strings

        Keys are optional; omit if unknown.

        ## SECTION HINTS
        "1"  Business / competitors
        "1A" Risk factors / competition
        "7"  MD&A / revenue, profit
        "7A" Market risk
        "8"  Financial statements
        "3"  Legal proceedings

        ## TICKER MAP
        Apple→AAPL  Meta→META  Microsoft→MSFT
        Tesla→TSLA  Amazon→AMZN  Netflix→NFLX

        ## EXAMPLES
        Q: how did Tesla's revenue change in FY 2019?  
        A: {"ticker":"TSLA","fiscal_year":2019,"sections":["7","8"]}

        Q: who were Tesla's main competitors in 2019?  
        A: {"ticker":"TSLA","fiscal_year":2019,"sections":["1","1A"]}

        If nothing can be parsed, output {}.
        """
        
        # Try GPT-4o-mini first for better accuracy
        try:
            response = retry_openai_call(
                self.openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,  # Lower temperature for consistent parsing
                max_tokens=100,   # Short response expected
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            print(f"GPT-4o-mini parsed: {result}")
            return result
            
        except Exception as e:
            print(f"GPT-4o-mini failed: {e}, falling back to GPT-3.5-turbo")
            
            # Fallback to GPT-3.5-turbo  
            try:
                response = retry_openai_call(
                    self.openai_client.chat.completions.create,
                    model="gpt-3.5-turbo-0125",
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": query},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=100,
                )
                content = response.choices[0].message.content
                result = json.loads(content)
                print(f"GPT-3.5-turbo parsed: {result}")
                return result
                
            except Exception as fallback_e:
                print(f"Both models failed: {fallback_e}")
                return {} 