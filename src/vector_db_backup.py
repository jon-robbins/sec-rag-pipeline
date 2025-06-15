"""
Backward compatibility wrapper for the refactored SEC Vector Store.

The original vector_db.py has been refactored into multiple modules under
src/sec_vectorstore/. This file maintains backward compatibility by re-exporting
the main classes and functions.

For new code, import directly from src.sec_vectorstore instead.
"""

# Re-export main classes for backward compatibility
from .sec_vectorstore import (
    VectorStore,
    create_vector_store,
    VectorStoreConfig
)

# Legacy aliases for full backward compatibility
_retry_openai = None  # Now handled internally by openai_helpers

# For any code that was importing the smoke test functionality
if __name__ == "__main__":
    # Redirect to the new CLI smoke test
    import sys
    import os
    
    # Import and run the smoke test
    from .sec_vectorstore.cli.smoke_test import main
    sys.exit(main())
