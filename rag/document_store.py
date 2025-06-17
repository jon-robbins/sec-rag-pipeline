"""
A store for accessing the processed SEC filing documents.
"""
from __future__ import annotations
import pandas as pd
import json
from typing import Optional
from pathlib import Path

class DocumentStore:
    """Provides access to the processed SEC filings data."""

    def __init__(self, processed_path: str = "data/filings_processed.parquet"):
        """
        Initializes the DocumentStore by loading the processed parquet file.
        """
        self.processed_path = Path(processed_path)
        if not self.processed_path.exists():
            raise FileNotFoundError(
                f"Processed filings not found at {self.processed_path}.\n"
                "Please run the data preparation step first."
            )
        # Use pyarrow engine for better complex object support
        self.df = pd.read_parquet(self.processed_path, engine='pyarrow')
        
        # Ensure the 'report' column is parsed as a dictionary
        if 'report' in self.df.columns and isinstance(self.df['report'].iloc[0], str):
            print("Detected 'report' column as string, parsing JSON...")
            self.df['report'] = self.df['report'].apply(json.loads)

        print(f"📁 DocumentStore initialized with {len(self.df)} documents from {self.processed_path}")

    def get_all_filings_for_chunking(self) -> pd.DataFrame:
        """Retrieves all filings with the structured 'report' column."""
        return self.df

    def get_full_filing_text(self, ticker: str, fiscal_year: int) -> Optional[str]:
        """Get the complete, concatenated SEC filing text for a given ticker and year."""
        filing = self.df[
            (self.df['ticker'] == ticker) & (self.df['fiscal_year'] == fiscal_year)
        ]
        if not filing.empty:
            return filing.iloc[0]['full_text']
        return None 