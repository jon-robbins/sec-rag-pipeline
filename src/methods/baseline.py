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


def run_baseline_scenario(
    openai_client: OpenAI, qa_item: Dict[str, Any]
) -> Tuple[str, Dict[str, int]]:
    """Scenario 5: Direct GPT-4o-mini without any additional context."""
    question = qa_item["question"]
    ticker = qa_item["ticker"]
    year = qa_item["year"]
    formatted_question = format_question_with_context(question, ticker, year)

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a financial analyst assistant. Your job is to answer questions about SEC filings using only your training knowledge.

IMPORTANT GUIDELINES:
1. Answer based ONLY on your existing knowledge from training data
2. Be concise and direct - aim for 2-4 sentences unless more detail is needed
3. You do NOT have access to current documents, web search, or real-time data
4. Include specific details like numbers, dates, or company names only if you're confident from training
5. Be honest about limitations - if information might be outdated or unavailable, say so
6. Do not make up specific financial figures or dates you're uncertain about
7. Focus on general knowledge about companies and typical SEC filing practices

RESPONSE FORMAT:
- Start directly with the answer
- Be professional but conversational
- If your training data is insufficient for specifics, explain this limitation
- Distinguish between general knowledge and specific facts you cannot verify""",
            },
            {"role": "user", "content": formatted_question},
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
