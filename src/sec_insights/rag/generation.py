"""
Answer generation using GPT based on retrieved document chunks.
"""

from typing import Any, Callable, Dict, List, Tuple

import backoff
from openai import OpenAIError

from .client_utils import get_openai_client


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


class AnswerGenerator:
    """Generates answers to questions using GPT based on retrieved document chunks."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.openai_client = get_openai_client()
        self.model = model

    def generate_answer(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        max_chunks: int = 10,
        max_context_length: int = 8000,
    ) -> Dict[str, Any]:
        """
        Generate an answer based on the question and retrieved chunks.

        Args:
            question: The user's question
            chunks: List of retrieved chunks with metadata
            max_chunks: Maximum number of chunks to include
            max_context_length: Maximum character length for context

        Returns:
            Dictionary containing the answer and metadata
        """
        result, _ = self.generate_answer_with_response(
            question, chunks, max_chunks, max_context_length
        )
        return result

    def generate_answer_with_response(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        max_chunks: int = 10,
        max_context_length: int = 8000,
    ) -> Tuple[Dict[str, Any], Any]:
        """
        Generate an answer and return both result and full response object.

        Args:
            question: The user's question
            chunks: List of retrieved chunks with metadata
            max_chunks: Maximum number of chunks to include
            max_context_length: Maximum character length for context

        Returns:
            Tuple of (answer_dict, response_object)
        """
        if not chunks:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "chunks_used": 0,
                "confidence": "low",
            }, None

        # Prepare context from chunks
        context = self._prepare_context(chunks, max_chunks, max_context_length)

        # Generate the answer
        try:
            response = self._make_generation_request(question, context["text"])
            answer = response.choices[0].message.content.strip()

            result = {
                "answer": answer,
                "sources": context["sources"],
                "chunks_used": context["chunks_used"],
                "confidence": self._assess_confidence(chunks, context["chunks_used"]),
            }
            return result, response
        except Exception as e:
            result = {
                "answer": f"I encountered an error while generating the answer: {str(e)}",
                "sources": [],
                "chunks_used": 0,
                "confidence": "error",
            }
            return result, None

    def _prepare_context(
        self, chunks: List[Dict[str, Any]], max_chunks: int, max_context_length: int
    ) -> Dict[str, Any]:
        """
        Prepare context text from chunks, respecting length limits.

        Args:
            chunks: Retrieved chunks
            max_chunks: Maximum chunks to use
            max_context_length: Maximum character length

        Returns:
            Dictionary with prepared context text and metadata
        """
        context_parts: List[str] = []
        sources = []
        current_length = 0
        chunks_used = 0

        for i, chunk in enumerate(chunks[:max_chunks]):
            # Extract text - try top level first, then payload
            chunk_text = chunk.get("text") or chunk.get("payload", {}).get("text", "")

            # Extract metadata - try top level first, then payload
            ticker = chunk.get("ticker") or chunk.get("payload", {}).get(
                "ticker", "UNKNOWN"
            )
            year = chunk.get("fiscal_year") or chunk.get("payload", {}).get(
                "fiscal_year", "UNKNOWN"
            )
            section = chunk.get("item") or chunk.get("payload", {}).get(
                "item", "UNKNOWN"
            )
            section_desc = chunk.get("item_desc") or chunk.get("payload", {}).get(
                "item_desc", ""
            )
            score = chunk.get("score", 0.0)

            source_info = f"{ticker} {year} Section {section}"
            if section_desc:
                source_info += f" ({section_desc})"
            source_info += f" [Score: {score:.3f}]"

            # Check if adding this chunk would exceed length limit
            chunk_addition = f"\n\n--- Source {i+1}: {source_info} ---\n{chunk_text}"

            if (
                current_length + len(chunk_addition) > max_context_length
                and context_parts
            ):
                break

            context_parts.append(chunk_addition)
            sources.append(source_info)
            current_length += len(chunk_addition)
            chunks_used += 1

        return {
            "text": "".join(context_parts),
            "sources": sources,
            "chunks_used": chunks_used,
        }

    def _make_generation_request(self, question: str, context: str) -> Any:
        """
        Make the actual API request for answer generation.

        Args:
            question: User's question
            context: Prepared context from retrieved chunks

        Returns:
            Full OpenAI API response object
        """
        system_prompt = """
You are a financial analyst assistant. Your job is to answer questions about SEC filings based ONLY on the provided context.

IMPORTANT GUIDELINES:
1. Answer based ONLY on the information provided in the context
2. Be concise and direct - aim for 2-4 sentences unless more detail is needed
3. If the context doesn't contain enough information to answer the question, say so clearly
4. Include specific details like numbers, dates, or company names when available
5. Do not make assumptions or add information not in the context
6. If multiple sources provide conflicting information, mention this
7. For financial data, be precise with numbers and units

RESPONSE FORMAT:
- Start directly with the answer
- Don't say "Based on the context" or similar phrases
- Be professional but conversational
- If you're unsure, express appropriate uncertainty"""

        user_prompt = f"""
Question: {question}

Context from SEC filings:
{context}

Please provide a brief, accurate answer based on the above context."""

        return retry_openai_call(
            self.openai_client.chat.completions.create,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for factual accuracy
            max_tokens=60,  # Limit response length
        )

    def _assess_confidence(self, chunks: List[Dict[str, Any]], chunks_used: int) -> str:
        """
        Assess confidence level based on search results quality.

        Args:
            chunks: All retrieved chunks
            chunks_used: Number of chunks actually used

        Returns:
            Confidence level string: "high", "medium", "low"
        """
        if not chunks:
            return "low"

        # Get the top score
        top_score = chunks[0].get("score", 0.0) if chunks else 0.0

        # Assess based on top score and number of relevant chunks
        if top_score > 0.8 and chunks_used >= 3:
            return "high"
        elif top_score > 0.6 and chunks_used >= 2:
            return "medium"
        else:
            return "low"
