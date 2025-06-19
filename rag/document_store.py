"""
A store for accessing the raw SEC filing documents and providing on-the-fly text generation.
"""
from __future__ import annotations
import pandas as pd
import duckdb
from typing import Optional, List, Dict
from pathlib import Path
import re
import html
import tiktoken

from .config import RAW_DATA_PATH, DB_DIR

# Text processing functions (moved from chunkers.py for logical grouping)
_BULLETS = re.compile(r"^[\s»\-–•\*]+\s*", re.MULTILINE)
_MULTI_WS = re.compile(r"\s+")
_ENC = tiktoken.encoding_for_model("gpt-3.5-turbo")

def _preprocess_text(raw: str) -> str:
    """Preprocesses raw text by unescaping HTML, removing bullets, and normalizing whitespace."""
    txt = html.unescape(raw or "")
    txt = _BULLETS.sub("", txt)
    txt = _MULTI_WS.sub(" ", txt)
    return txt.strip()

def _count_tokens(text: str) -> int:
    """Counts the number of tokens in a string."""
    return len(_ENC.encode(text))

class DocumentStore:
    """
    Handles loading and accessing sentence-level data from SEC filings.
    """
    def __init__(self, tickers_of_interest=None):
        if tickers_of_interest is None:
            # Default tickers of interest
            self.tickers = ['AAPL', 'META', 'TSLA', 'NVDA', 'AMZN']
        else:
            self.tickers = tickers_of_interest
            
        self.raw_data_path = RAW_DATA_PATH
        self.db_path = str(DB_DIR / "sec_filings.duckdb")
        self.df = None
        self._full_texts = None # Cache for full document texts

    def _load_data_if_needed(self):
        """Loads and processes the raw data if it hasn't been loaded yet."""
        if self.df is not None:
            return

        ticker_list_str = "','".join(self.tickers)
        query = f"""
        WITH unnested AS (
            SELECT *, UNNEST(tickers) AS ticker FROM read_parquet('{self.raw_data_path}')
        )
        SELECT 
            ticker,
            CAST(RIGHT(docID, 4) AS INTEGER) AS fiscal_year,
            docID,
            sentenceID,
            sentence,
            regexp_extract(sentenceID, 'section_([^_]+)', 1) AS section
        FROM unnested
        WHERE 1=1
        and ticker IN ('{ticker_list_str}')
        and cast(RIGHT(docID, 4) AS INTEGER) between 2012 and 2020
        """
        print("📁 DocumentStore: Loading and processing raw sentence data...")
        self.df = duckdb.query(query).to_df()
        print(f"✅ Loaded {len(self.df)} sentences for {len(self.tickers)} tickers.")
        
        # Preprocess sentences and calculate token counts before sorting
        print("⚙️  Preprocessing sentences and counting tokens...")
        self.df['sentence'] = self.df['sentence'].apply(_preprocess_text)
        self.df['sentence_token_count'] = self.df['sentence'].apply(_count_tokens)
        
        # Remove empty sentences that might result from preprocessing
        self.df = self.df[self.df['sentence'].str.len() > 0].copy()
        
        # Sort dataframe to ensure correct order for full_text generation
        self.df['section_num'] = pd.to_numeric(self.df['section'].str.extract(r'(\d+)')[0], errors='coerce')
        self.df['section_letter'] = self.df['section'].str.extract(r'\d+([A-Z]?)').fillna('')
        self.df = self.df.sort_values(by=['docID', 'section_num', 'section_letter']).drop(columns=['section_num', 'section_letter'])

        print("Pre-calculating full texts for each document...")
        self._full_texts = self.df.groupby('docID')['sentence'].apply(' '.join).to_dict()
        print("✅ Pre-calculation of full texts complete.")

    def get_all_sentences(self) -> pd.DataFrame:
        """Retrieves all filtered sentences, loading data if necessary."""
        self._load_data_if_needed()
        return self.df

    def get_sentences_by_filter(self, tickers: List[str], fiscal_years: List[int]) -> pd.DataFrame:
        """
        Retrieves sentences based on a filter of tickers and fiscal years.
        """
        self._load_data_if_needed()
        
        filtered_df = self.df[
            self.df['ticker'].isin(tickers) &
            self.df['fiscal_year'].isin(fiscal_years)
        ]
        
        return filtered_df.copy()

    def get_full_filing_text(self, ticker: str, fiscal_year: int) -> Optional[str]:
        """
        Get the complete, concatenated SEC filing text for a given ticker and year.
        This method can query the data on-demand if it wasn't pre-loaded.
        """
        self._load_data_if_needed()
        
        # Find the docID for the given ticker and year. Assumes one 10-K per ticker per year.
        filtered_df = self.df[(self.df['ticker'] == ticker) & (self.df['fiscal_year'] == fiscal_year)]
        if not filtered_df.empty:
            docID = filtered_df['docID'].iloc[0]
            return self._full_texts.get(docID)
        return None 