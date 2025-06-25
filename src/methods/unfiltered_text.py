import logging
from typing import Any, Dict, Tuple

from openai import OpenAI

from src.vector_store.document_store import DocumentStore

logger = logging.getLogger(__name__)


def format_question_with_context(question: str, ticker: str, fiscal_year: int) -> str:
    """Add company and year context to the question for better parsing."""
    # Format with explicit ticker and fiscal year for optimal parsing
    formatted = (
        f"Question about ticker {ticker} for fiscal year {fiscal_year}: {question}"
    )
    logger.debug(f"🔍 Formatted question: {formatted}")
    return formatted


def run_unfiltered_context_scenario(
    doc_store: "DocumentStore", openai_client: OpenAI, qa_item: Dict[str, Any]
) -> Tuple[str, Dict[str, int]]:
    """Scenario 1: GPT-4 with entire SEC filing as context."""
    question = qa_item["question"]
    ticker = qa_item["ticker"]
    year = qa_item["year"]

    full_text = doc_store.get_full_filing_text(ticker, year)

    if not full_text:
        return f"[No SEC filing data available for {ticker} {year}]", {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    system_prompt = """You are a financial analyst assistant. Your job is to answer questions about SEC filings based ONLY on the complete SEC filing context provided.

IMPORTANT GUIDELINES:
1. Answer based ONLY on the information in the complete SEC filing provided
2. Be concise and direct - aim for 2-4 sentences unless more detail is needed
3. You have access to the ENTIRE SEC filing, so search thoroughly for relevant information
4. Include specific details like numbers, dates, or company names when available
5. Do not make assumptions or add information not in the filing
6. If multiple sections provide conflicting information, mention this and cite the sections
7. For financial data, be precise with numbers and units

RESPONSE FORMAT:
- Start directly with the answer
- Don't say "Based on the context" or similar phrases
- Be professional but conversational
- If you're unsure after reviewing the entire filing, express appropriate uncertainty"""

    formatted_question = format_question_with_context(question, ticker, year)
    user_prompt = f"Context: {full_text}\n\nQuestion: {formatted_question}\n\nAnswer:"

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=60,
    )

    token_usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }

    return response.choices[0].message.content.strip(), token_usage
