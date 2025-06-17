"""
Query parsing utilities for the SEC Vector Store.
"""

import json
from typing import Dict, Any, Tuple, Union

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
        response = self._make_parsing_request(query)
        try:
            content = response.choices[0].message.content
            result = json.loads(content)
            print(f"Parsed query: {result}")
            return result
        except Exception as e:
            print(f"Failed to parse response: {e}")
            return {}
    
    def parse_query_with_response(self, query: str) -> Tuple[Dict[str, Any], Any]:
        """
        Parse query and return both result and full response object.
        
        Args:
            query: Natural language query string
            
        Returns:
            Tuple of (parsed_parameters, response_object)
        """
        response = self._make_parsing_request(query)
        try:
            content = response.choices[0].message.content
            result = json.loads(content)
            print(f"Parsed query: {result}")
            return result, response
        except Exception as e:
            print(f"Failed to parse response: {e}")
            return {}, response
    
    def _make_parsing_request(self, query: str) -> Any:
        """Make the actual API request for parsing."""
        system_message = """
## ROLE
You are an expert financial-query parser.  
Respond exclusively with ONE JSON object matching the specified keys.  
Provide NO additional text or explanations.

## OUTPUT FORMAT
Return a JSON object with these optional keys (omit if unknown):

- `"ticker"`: 2–5 letter uppercase US stock symbol.
- `"fiscal_year"`: 4-digit year between 1995–2030.
- `"sections"`: An array of up to 3 SEC 10-K item codes as strings (see below).

## SECTION CODES
Use these SEC 10-K item codes when identifying relevant sections:

{
  "Item 1": "Business Operations, Products, Services (including competitors, market details)",
  "Item 1A": "Risk Factors",
  "Item 1B": "Unresolved Staff Comments",
  "Item 2": "Properties",
  "Item 3": "Legal Proceedings",
  "Item 4": "Mine Safety Disclosures (mining companies only)",
  "Item 5": "Market for Registrant’s Common Equity and Related Stockholder Matters",
  "Item 6": "[Removed and Reserved]",
  "Item 7": "Management’s Discussion and Analysis (MD&A)",
  "Item 7A": "Quantitative and Qualitative Disclosures About Market Risk",
  "Item 8": "Financial Statements and Supplementary Data",
  "Item 9": "Changes in and Disagreements with Accountants on Accounting/Financial Disclosure",
  "Item 9A": "Controls and Procedures",
  "Item 9B": "Other Information",
  "Item 9C": "Disclosure Regarding Foreign Jurisdictions (HFCAA)",
  "Item 10": "Directors, Executive Officers, and Corporate Governance",
  "Item 11": "Executive Compensation",
  "Item 12": "Security Ownership of Certain Beneficial Owners, Management, and Related Matters",
  "Item 13": "Certain Relationships, Related Transactions, Director Independence",
  "Item 14": "Principal Accounting Fees and Services",
  "Item 15": "Exhibits and Financial Statement Schedules",
  "Item 16": "Form 10-K Summary (optional)"
}

## COMMON COMPANY NAMES (Ticker Reference)
Apple→AAPL  
Meta→META  
Microsoft→MSFT  
Tesla→TSLA  
Amazon→AMZN  
Netflix→NFLX  

## EXAMPLES

**Q:** How did Tesla's revenue change in FY 2019?  
**A:** `{"ticker":"TSLA","fiscal_year":2019,"sections":["7","8"]}`

**Q:** Who were Tesla's main competitors in 2019?  
**A:** `{"ticker":"TSLA","fiscal_year":2019,"sections":["1","1A"]}`

If the input doesn't match any known format, output an empty JSON object `{}`.

        """
        
        # Try GPT-4o-mini first for better accuracy
        try:
            return retry_openai_call(
                self.openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": query},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=100,
            )
        except Exception as e:
            print(f"GPT-4o-mini failed: {e}, falling back to GPT-3.5-turbo")
            
            # Fallback to GPT-3.5-turbo  
            return retry_openai_call(
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