"""
Backward compatibility wrapper for the refactored SEC Vector Store.

The original vector_db.py has been refactored into multiple modules under
src/sec_vectorstore/. This file maintains backward compatibility by re-exporting
the main classes and functions.

For new code, import directly from src.sec_vectorstore instead.

NEW RAG CAPABILITIES:
- vs.answer(question) - Get AI-generated answers based on retrieved chunks
- vs.summarize(topic) - Get AI-generated summaries of topics
- Automatic query parsing and context-aware responses
"""

# Re-export main classes for backward compatibility
from .sec_vectorstore import (
    VectorStore,
    create_vector_store,
    VectorStoreConfig,
    AnswerGenerator
)

# Legacy function name mapping
def _retry_openai(*args, **kwargs):
    """Legacy function - now handled internally by openai_helpers."""
    raise NotImplementedError(
        "_retry_openai is now handled internally. "
        "Use VectorStore class or import from sec_vectorstore.openai_helpers"
    )

# For any code that was importing the smoke test functionality
if __name__ == "__main__":
    # Redirect to the new CLI smoke test
    import sys
    import os
    
    print("🔄 Redirecting to refactored smoke test...")
    print("💡 New RAG capabilities available: vs.answer() and vs.summarize()")
    
    # Import and run the smoke test
    from .sec_vectorstore.cli.smoke_test import main
    sys.exit(main()) 