"""
Answer generation using GPT based on retrieved document chunks.
"""

import json
from typing import List, Dict, Any, Optional

from .openai_helpers import retry_openai_call
from .config import DEFAULT_OPENAI_KEY
import openai


class AnswerGenerator:
    """Generates answers to questions using GPT based on retrieved document chunks."""
    
    def __init__(self, openai_key: str = None, model: str = "gpt-4o-mini"):
        self.openai_client = openai.OpenAI(api_key=openai_key or DEFAULT_OPENAI_KEY)
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
        if not chunks:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "chunks_used": 0,
                "confidence": "low"
            }
        
        # Prepare context from chunks
        context = self._prepare_context(chunks, max_chunks, max_context_length)
        
        # Generate the answer
        try:
            answer = self._call_gpt(question, context["text"])
            
            return {
                "answer": answer,
                "sources": context["sources"],
                "chunks_used": context["chunks_used"],
                "confidence": self._assess_confidence(chunks, context["chunks_used"])
            }
        except Exception as e:
            return {
                "answer": f"I encountered an error while generating the answer: {str(e)}",
                "sources": [],
                "chunks_used": 0,
                "confidence": "error"
            }
    
    def _prepare_context(
        self, 
        chunks: List[Dict[str, Any]], 
        max_chunks: int, 
        max_context_length: int
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
        context_parts = []
        sources = []
        current_length = 0
        chunks_used = 0
        
        for i, chunk in enumerate(chunks[:max_chunks]):
            chunk_text = chunk.get("text", "")
            
            # Create source reference
            ticker = chunk.get("ticker", "UNKNOWN")
            year = chunk.get("fiscal_year", "UNKNOWN")
            section = chunk.get("item", "UNKNOWN")
            section_desc = chunk.get("item_desc", "")
            score = chunk.get("score", 0.0)
            
            source_info = f"{ticker} {year} Section {section}"
            if section_desc:
                source_info += f" ({section_desc})"
            source_info += f" [Score: {score:.3f}]"
            
            # Check if adding this chunk would exceed length limit
            chunk_addition = f"\n\n--- Source {i+1}: {source_info} ---\n{chunk_text}"
            
            if current_length + len(chunk_addition) > max_context_length and context_parts:
                break
                
            context_parts.append(chunk_addition)
            sources.append(source_info)
            current_length += len(chunk_addition)
            chunks_used += 1
        
        return {
            "text": "".join(context_parts),
            "sources": sources,
            "chunks_used": chunks_used
        }
    
    def _call_gpt(self, question: str, context: str) -> str:
        """
        Call GPT to generate an answer based on the question and context.
        
        Args:
            question: User's question
            context: Prepared context from retrieved chunks
            
        Returns:
            Generated answer string
        """
        system_prompt = """You are a financial analyst assistant. Your job is to answer questions about SEC filings based ONLY on the provided context.

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

        user_prompt = f"""Question: {question}

Context from SEC filings:
{context}

Please provide a brief, accurate answer based on the above context."""

        response = retry_openai_call(
            self.openai_client.chat.completions.create,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for factual accuracy
            max_tokens=500,   # Limit response length
        )
        
        return response.choices[0].message.content.strip()
    
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
    
    def generate_summary(self, chunks: List[Dict[str, Any]], topic: str = None) -> str:
        """
        Generate a summary of the retrieved chunks.
        
        Args:
            chunks: Retrieved chunks
            topic: Optional topic focus for the summary
            
        Returns:
            Generated summary string
        """
        if not chunks:
            return "No relevant information found."
        
        context = self._prepare_context(chunks, max_chunks=15, max_context_length=12000)
        
        topic_clause = f" about {topic}" if topic else ""
        
        summary_prompt = f"""Please provide a comprehensive summary of the key information{topic_clause} from the following SEC filing excerpts:

{context['text']}

Focus on the most important facts, figures, and insights. Organize the information logically and be specific with details."""

        try:
            response = retry_openai_call(
                self.openai_client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst. Provide clear, well-organized summaries of SEC filing information."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.2,
                max_tokens=800,
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating summary: {str(e)}" 