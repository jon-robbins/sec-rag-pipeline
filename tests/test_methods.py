"""
Tests for all evaluation methods in src.methods.
"""

from unittest.mock import ANY, MagicMock, patch

import numpy as np

from src.methods import (
    baseline,
    ensemble_rerank_rag,
    rerank_rag,
    unfiltered_text,
    vanilla_rag,
    web_search,
)


class TestBaseline:
    """Tests for baseline.py"""

    def test_format_question_with_context(self):
        """Test question formatting with context."""
        question = "What is the revenue?"
        ticker = "AAPL"
        year = 2021

        result = baseline.format_question_with_context(question, ticker, year)

        assert "AAPL" in result
        assert "2021" in result
        assert question in result

    @patch("src.methods.baseline.OpenAI")
    def test_run_baseline_scenario(self, mock_openai_class):
        """Test baseline scenario execution."""
        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test answer"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_client.chat.completions.create.return_value = mock_response

        qa_item = {"question": "What is the revenue?", "ticker": "AAPL", "year": 2021}

        answer, tokens = baseline.run_baseline_scenario(mock_client, qa_item)

        assert answer == "Test answer"
        assert tokens["prompt_tokens"] == 100
        assert tokens["completion_tokens"] == 50
        assert tokens["total_tokens"] == 150
        mock_client.chat.completions.create.assert_called_once()


class TestWebSearch:
    """Tests for web_search.py"""

    def test_format_question_with_context(self):
        """Test question formatting with context."""
        question = "What is the revenue?"
        ticker = "AAPL"
        year = 2021

        result = web_search.format_question_with_context(question, ticker, year)

        assert "AAPL" in result
        assert "2021" in result
        assert question in result

    @patch("src.methods.web_search.OpenAI")
    def test_run_web_search_scenario(self, mock_openai_class):
        """Test web search scenario execution."""
        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test web search answer"
        mock_response.usage.prompt_tokens = 120
        mock_response.usage.completion_tokens = 60
        mock_response.usage.total_tokens = 180
        mock_client.chat.completions.create.return_value = mock_response

        qa_item = {"question": "What is the revenue?", "ticker": "AAPL", "year": 2021}

        answer, tokens = web_search.run_web_search_scenario(mock_client, qa_item)

        assert answer == "Test web search answer"
        assert tokens["prompt_tokens"] == 120
        assert tokens["completion_tokens"] == 60
        assert tokens["total_tokens"] == 180
        mock_client.chat.completions.create.assert_called_once()


class TestUnfilteredText:
    """Tests for unfiltered_text.py"""

    def test_format_question_with_context(self):
        """Test question formatting with context."""
        question = "What is the revenue?"
        ticker = "AAPL"
        year = 2021

        result = unfiltered_text.format_question_with_context(question, ticker, year)

        assert "AAPL" in result
        assert "2021" in result
        assert question in result

    @patch("src.methods.unfiltered_text.OpenAI")
    def test_run_unfiltered_context_scenario(self, mock_openai_class):
        """Test unfiltered context scenario execution."""
        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test unfiltered answer"
        mock_response.usage.prompt_tokens = 5000
        mock_response.usage.completion_tokens = 100
        mock_response.usage.total_tokens = 5100
        mock_client.chat.completions.create.return_value = mock_response

        # Mock DocumentStore
        mock_doc_store = MagicMock()
        mock_doc_store.get_full_filing_text.return_value = (
            "Full SEC filing text content..."
        )

        qa_item = {"question": "What is the revenue?", "ticker": "AAPL", "year": 2021}

        answer, tokens = unfiltered_text.run_unfiltered_context_scenario(
            mock_doc_store, mock_client, qa_item
        )

        assert answer == "Test unfiltered answer"
        assert tokens["prompt_tokens"] == 5000
        assert tokens["completion_tokens"] == 100
        assert tokens["total_tokens"] == 5100
        mock_doc_store.get_full_filing_text.assert_called_once_with("AAPL", 2021)
        mock_client.chat.completions.create.assert_called_once()

    @patch("src.methods.unfiltered_text.OpenAI")
    def test_run_unfiltered_context_scenario_no_filing(self, mock_openai_class):
        """Test unfiltered context scenario when no filing is found."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock DocumentStore with no filing found
        mock_doc_store = MagicMock()
        mock_doc_store.get_full_filing_text.return_value = None

        qa_item = {"question": "What is the revenue?", "ticker": "AAPL", "year": 2021}

        answer, tokens = unfiltered_text.run_unfiltered_context_scenario(
            mock_doc_store, mock_client, qa_item
        )

        assert "No SEC filing data available" in answer
        assert tokens["prompt_tokens"] == 0
        assert tokens["completion_tokens"] == 0
        assert tokens["total_tokens"] == 0
        mock_client.chat.completions.create.assert_not_called()


class TestVanillaRAG:
    """Tests for vanilla_rag.py"""

    def test_format_question_with_context(self):
        """Test question formatting with context."""
        question = "What is the revenue?"
        ticker = "AAPL"
        year = 2021

        result = vanilla_rag.format_question_with_context(question, ticker, year)

        assert "AAPL" in result
        assert "2021" in result
        assert question in result

    def test_run_rag_scenario(self):
        """Test vanilla RAG scenario execution."""
        # Mock RAGPipeline
        mock_pipeline = MagicMock()
        mock_chunks = [
            {"id": "chunk1", "payload": {"text": "Revenue was $100B"}},
            {"id": "chunk2", "payload": {"text": "Expenses were $50B"}},
        ]
        mock_pipeline.search.return_value = mock_chunks

        # Mock answer generator response
        mock_result = {"answer": "Revenue was $100B according to the filing."}
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 200
        mock_response.usage.completion_tokens = 75
        mock_response.usage.total_tokens = 275
        mock_pipeline.answer_generator.generate_answer_with_response.return_value = (
            mock_result,
            mock_response,
        )

        qa_item = {"question": "What is the revenue?", "ticker": "AAPL", "year": 2021}

        answer, retrieved_ids, tokens = vanilla_rag.run_rag_scenario(
            mock_pipeline, qa_item
        )

        assert answer == "Revenue was $100B according to the filing."
        assert retrieved_ids == ["chunk1", "chunk2"]
        assert tokens["prompt_tokens"] == 200
        assert tokens["completion_tokens"] == 75
        assert tokens["total_tokens"] == 275
        mock_pipeline.search.assert_called_once()
        mock_pipeline.answer_generator.generate_answer_with_response.assert_called_once()


class TestRerankRAG:
    """Tests for rerank_rag.py"""

    def test_format_question_with_context(self):
        """Test question formatting with context."""
        question = "What is the revenue?"
        ticker = "AAPL"
        year = 2021

        result = rerank_rag.format_question_with_context(question, ticker, year)

        assert "AAPL" in result
        assert "2021" in result
        assert question in result

    def test_run_reranked_rag_scenario(self):
        """Test reranked RAG scenario execution."""
        # Mock RAGPipeline
        mock_pipeline = MagicMock()
        mock_chunks = [
            {"id": "chunk1", "payload": {"text": "Revenue was $100B"}},
            {"id": "chunk2", "payload": {"text": "Expenses were $50B"}},
            {"id": "chunk3", "payload": {"text": "Profit was $50B"}},
        ]
        mock_pipeline.search.return_value = mock_chunks

        # Mock reranker
        mock_reranker = MagicMock()
        # Reranker returns (index, score) tuples - reorder chunks
        mock_reranker.rerank.return_value = [(1, 0.9), (0, 0.8), (2, 0.7)]

        # Mock answer generator response
        mock_result = {"answer": "Expenses were $50B and revenue was $100B."}
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 250
        mock_response.usage.completion_tokens = 80
        mock_response.usage.total_tokens = 330
        mock_pipeline.answer_generator.generate_answer_with_response.return_value = (
            mock_result,
            mock_response,
        )

        qa_item = {"question": "What is the revenue?", "ticker": "AAPL", "year": 2021}

        answer, retrieved_ids, tokens = rerank_rag.run_reranked_rag_scenario(
            pipeline=mock_pipeline,
            qa_item=qa_item,
            reranker=mock_reranker,
            phase_1_k=30,
            phase_2_k=10,
        )

        assert answer == "Expenses were $50B and revenue was $100B."
        # Should return all reranked IDs for evaluation
        assert retrieved_ids == ["chunk2", "chunk1", "chunk3"]
        assert tokens["prompt_tokens"] == 250
        assert tokens["completion_tokens"] == 80
        assert tokens["total_tokens"] == 330
        mock_pipeline.search.assert_called_once_with(query=ANY, top_k=30)
        mock_reranker.rerank.assert_called_once()
        mock_pipeline.answer_generator.generate_answer_with_response.assert_called_once()

    def test_run_reranked_rag_scenario_no_documents(self):
        """Test reranked RAG scenario when no documents are found."""
        # Mock RAGPipeline with no results
        mock_pipeline = MagicMock()
        mock_pipeline.search.return_value = []

        mock_reranker = MagicMock()

        qa_item = {"question": "What is the revenue?", "ticker": "AAPL", "year": 2021}

        answer, retrieved_ids, tokens = rerank_rag.run_reranked_rag_scenario(
            pipeline=mock_pipeline,
            qa_item=qa_item,
            reranker=mock_reranker,
            phase_1_k=30,
            phase_2_k=10,
        )

        assert "[No documents found to rerank]" in answer
        assert retrieved_ids == []
        assert tokens["prompt_tokens"] == 0
        assert tokens["completion_tokens"] == 0
        assert tokens["total_tokens"] == 0
        mock_reranker.rerank.assert_not_called()


class TestEnsembleRerankRAG:
    """Tests for ensemble_rerank_rag.py"""

    @patch("src.methods.ensemble_rerank_rag.expand_query")
    @patch("src.methods.ensemble_rerank_rag.get_rerankers")
    def test_run_ensemble_rerank_rag(self, mock_get_rerankers, mock_expand_query):
        """Test ensemble rerank RAG scenario execution."""
        # Mock query expansion
        mock_expand_query.return_value = ("expanded query about revenue", 50, 25)

        # Mock rerankers
        mock_jina_reranker = MagicMock()
        mock_bge_reranker = MagicMock()
        # Return numpy arrays instead of lists
        mock_jina_reranker.predict.return_value = np.array([0.9, 0.7, 0.8])
        mock_bge_reranker.predict.return_value = np.array([0.8, 0.9, 0.6])
        mock_get_rerankers.return_value = {
            "jina": mock_jina_reranker,
            "bge": mock_bge_reranker,
        }

        # Mock RAGPipeline
        mock_pipeline = MagicMock()
        mock_chunks = [
            {"id": "chunk1", "payload": {"text": "Revenue was $100B"}},
            {"id": "chunk2", "payload": {"text": "Expenses were $50B"}},
            {"id": "chunk3", "payload": {"text": "Profit was $50B"}},
        ]
        mock_pipeline.search.return_value = mock_chunks

        # Mock answer generator response
        mock_result = {"answer": "Revenue was $100B according to ensemble analysis."}
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 300
        mock_response.usage.completion_tokens = 90
        mock_response.usage.total_tokens = 390
        mock_pipeline.answer_generator.generate_answer_with_response.return_value = (
            mock_result,
            mock_response,
        )

        question = "What is the revenue?"

        answer, retrieved_ids, tokens = ensemble_rerank_rag.run_ensemble_rerank_rag(
            rag_pipeline=mock_pipeline,
            question=question,
            phase_1_k=30,
            phase_2_k=10,
            use_rrf=False,
        )

        assert answer == "Revenue was $100B according to ensemble analysis."
        assert len(retrieved_ids) > 0
        assert tokens["prompt_tokens"] == 350  # 50 (expansion) + 300 (generation)
        assert tokens["completion_tokens"] == 115  # 25 (expansion) + 90 (generation)
        assert tokens["total_tokens"] == 465

        mock_expand_query.assert_called_once_with(question)
        mock_get_rerankers.assert_called_once()
        mock_pipeline.search.assert_called_once()
        mock_jina_reranker.predict.assert_called_once()
        mock_bge_reranker.predict.assert_called_once()
        mock_pipeline.answer_generator.generate_answer_with_response.assert_called_once()

    def test_get_rerankers(self):
        """Test reranker initialization."""
        with patch(
            "src.methods.ensemble_rerank_rag.CrossEncoder"
        ) as mock_cross_encoder:
            mock_cross_encoder.return_value = MagicMock()

            rerankers = ensemble_rerank_rag.get_rerankers()

            assert "jina" in rerankers
            assert "bge" in rerankers
            assert mock_cross_encoder.call_count == 2  # Called for both rerankers
