"""
Query parsing utilities for the SEC Vector Store.
"""

import json
import logging
from typing import Any, Callable, Dict, Tuple

import backoff
from openai import OpenAIError

from .client_utils import get_openai_client

logger = logging.getLogger(__name__)


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


class QueryParser:
    """Handles parsing of natural language queries into structured parameters."""

    def __init__(self):
        self.openai_client = get_openai_client()

    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse a natural language query into ticker, fiscal_year, and sections.

        Args:
            query: Natural language query string

        Returns:
            Dictionary with parsed parameters: ticker, fiscal_year, sections
        """
        result, _ = self.parse_query_with_response(query)
        return result

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
            logger.info("Parsed query: %s", result)
            return result, response
        except Exception as e:
            logger.error("Failed to parse response: %s", e)
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
- `"sections"`: An array of up to 3 SEC 10-K section codes as strings (see below).

## SECTION CODES
Use these SEC 10-K section codes when identifying relevant sections:

{
  "Section 1": "Business Operations, Products, Services (including competitors, market details)",
  "Section 1A": "Risk Factors",
  "Section 1B": "Unresolved Staff Comments",
  "Section 2": "Properties",
  "Section 3": "Legal Proceedings",
  "Section 4": "Mine Safety Disclosures (mining companies only)",
  "Section 5": "Market for Registrant's Common Equity and Related Stockholder Matters",
  "Section 6": "[Removed and Reserved]",
  "Section 7": "Management's Discussion and Analysis (MD&A)",
  "Section 7A": "Quantitative and Qualitative Disclosures About Market Risk",
  "Section 8": "Financial Statements and Supplementary Data",
  "Section 9": "Changes in and Disagreements with Accountants on Accounting/Financial Disclosure",
  "Section 9A": "Controls and Procedures",
  "Section 9B": "Other Information",
  "Section 9C": "Disclosure Regarding Foreign Jurisdictions (HFCAA)",
  "Section 10": "Directors, Executive Officers, and Corporate Governance",
  "Section 11": "Executive Compensation",
  "Section 12": "Security Ownership of Certain Beneficial Owners, Management, and Related Matters",
  "Section 13": "Certain Relationships, Related Transactions, Director Independence",
  "Section 14": "Principal Accounting Fees and Services",
  "Section 15": "Exhibits and Financial Statement Schedules",
  "Section 16": "Form 10-K Summary (optional)"
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
            logger.error("GPT-4o-mini failed: %s, falling back to GPT-3.5-turbo", e)

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
