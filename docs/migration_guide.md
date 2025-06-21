# Migration Guide

This document outlines the key changes introduced during the comprehensive refactoring of the `sec-insights` library. The goal of the refactor was to improve code quality, clarity, and maintainability without changing external behavior.

## 1. Project Structure (`src/` Layout)

The entire library has been moved into a `src/` directory to follow modern Python packaging standards. All imports must be updated to reflect this change.

**Old Structure:**
```
.
├── rag/
├── evaluation/
└── ...
```

**New Structure:**
```
.
├── src/
│   └── sec_insights/
│       ├── rag/
│       ├── evaluation/
│       └── ...
└── ...
```

**Example Import Change:**

- **Old**: `from rag.pipeline import RAGPipeline`
- **New**: `from sec_insights.rag.pipeline import RAGPipeline`

## 2. Public API: `RAGPipeline`

The primary entry point for all library functionality is now the `sec_insights.rag.pipeline.RAGPipeline` class. It orchestrates all high-level operations like searching and answering questions.

Direct instantiation of lower-level classes like `VectorStore`, `QueryParser`, or `AnswerGenerator` is discouraged for typical use.

## 3. `VectorStore` Refactoring

The `VectorStore` class has been significantly refactored. It is now a pure **data access layer** responsible only for direct communication with the Qdrant vector database.

- **No More Business Logic**: The `VectorStore` no longer contains methods like `answer()` or logic for parsing queries. This responsibility now lives in `RAGPipeline`.
- **Consolidated Managers**: The internal `CollectionManager` and `SearchManager` classes have been removed, and their logic has been merged directly into the `VectorStore`. This simplifies the internal architecture.
- **Search Method Change**: The `search()` method now requires a pre-computed `query_vector`. The embedding of the query string is handled by `RAGPipeline`.

**Old Usage (Discouraged):**
```python
# Old way, multiple classes and steps visible to user
vs = VectorStore()
parsed_query = vs.query_parser.parse_query("some question")
# ... manual steps ...
```

**New Usage:**
```python
# New, simplified API via RAGPipeline
from sec_insights.rag.pipeline import RAGPipeline

pipeline = RAGPipeline()
answer = pipeline.answer("some question")
```

## 4. Client Instantiation

All OpenAI API client instantiations have been centralized into a single utility function: `sec_insights.rag.client_utils.get_openai_client()`. This ensures consistent configuration (e.g., API keys from `.env`) across the library.

## 5. Bug Fixes

- The `summarize()` method in `RAGPipeline` now correctly generates summaries.
- The evaluation scenarios have been made more efficient by removing redundant API calls.
