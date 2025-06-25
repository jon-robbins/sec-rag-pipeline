import logging
from typing import Any, Dict, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)


def format_question_with_context(question: str, ticker: str, fiscal_year: int) -> str:
    """Add company and year context to the question for better parsing."""
    # Format with explicit ticker and fiscal year for optimal parsing
    formatted = (
        f"Question about ticker {ticker} for fiscal year {fiscal_year}: {question}"
    )
    logger.debug(f"🔍 Formatted question: {formatted}")
    return formatted


def run_web_search_scenario(
    openai_client: OpenAI, qa_item: Dict[str, Any]
) -> Tuple[str, Dict[str, int]]:
    """Scenario 2: GPT-4 with web search."""
    question = qa_item["question"]
    ticker = qa_item["ticker"]
    year = qa_item["year"]
    formatted_question = format_question_with_context(question, ticker, year)

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini-search-preview",
        web_search_options={},
        messages=[
            {
                "role": "system",
                "content": """
You are a financial analyst assistant with web search capabilities. Your job is to answer questions about SEC filings using real-time web search.

IMPORTANT GUIDELINES:
1. Use web search to find current and accurate information about the company's SEC filings
2. Be concise and direct - aim for 2-4 sentences unless more detail is needed
3. Search for official SEC filings, company reports, and reliable financial sources
4. Include specific details like numbers, dates, or company names when available
5. Cite or reference the sources you find when possible
6. Focus on factual, verifiable information from authoritative sources
7. For financial data, be precise with numbers and units

RESPONSE FORMAT:
- Start directly with the answer
- Be professional but conversational
- If search results are insufficient, mention what you were unable to find
- Prioritize official SEC documents and filings in your search""",
            },
            {"role": "user", "content": formatted_question},
        ],
        max_tokens=60,
    )

    token_usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return response.choices[0].message.content.strip(), token_usage
