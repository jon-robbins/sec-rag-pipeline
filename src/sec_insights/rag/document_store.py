"""
A store for accessing the raw SEC filing documents and providing on-the-fly text generation.
"""

from __future__ import annotations

import html
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

import duckdb
import pandas as pd
import tiktoken

logger = logging.getLogger(__name__)

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
    """Handles loading and accessing sentence-level data from SEC filings.

    This class connects to a DuckDB database generated from raw Parquet files
    and provides methods to retrieve sentences or full document texts based
    on specified filters.

    Parameters
    ----------
    raw_data_path : Path
        The path to the raw data Parquet file.
    tickers_of_interest : Optional[List[str]], optional
        A list of ticker symbols to focus on. If None, a default list is used,
        by default None.
    """

    def __init__(
        self, raw_data_path: Path, tickers_of_interest: Optional[List[str]] = None
    ):
        if tickers_of_interest is None:
            self.tickers = ["AAPL", "META", "TSLA", "NVDA", "AMZN"]
        else:
            self.tickers = tickers_of_interest

        self.raw_data_path = raw_data_path
        if not self.raw_data_path.exists():
            raise FileNotFoundError(
                f"Raw data parquet file not found at: {self.raw_data_path}"
            )

        self.df: Optional[pd.DataFrame] = None
        self._full_texts: Optional[Dict[str, str]] = None

    def _load_data_if_needed(self) -> None:
        """Loads and processes the raw data if it hasn't been loaded yet."""
        if self.df is not None:
            return

        ticker_list_str = "','".join(self.tickers)
        query = f"""
        WITH unnested AS (
            SELECT *, UNNEST(tickers) AS ticker FROM read_parquet('{str(self.raw_data_path)}')
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
        logger.info("DocumentStore: Loading and processing raw sentence data...")
        self.df = duckdb.query(query).to_df()
        logger.info(
            "Loaded %d sentences for %d tickers.", len(self.df), len(self.tickers)
        )

        # Preprocess sentences and calculate token counts before sorting
        logger.info("Preprocessing sentences and counting tokens...")
        self.df["sentence"] = self.df["sentence"].apply(_preprocess_text)
        self.df["sentence_token_count"] = self.df["sentence"].apply(_count_tokens)

        # Remove empty sentences that might result from preprocessing
        self.df = self.df[self.df["sentence"].str.len() > 0].copy()

        # Sort dataframe to ensure correct order for full_text generation
        self.df["section_num"] = pd.to_numeric(
            self.df["section"].str.extract(r"(\d+)")[0], errors="coerce"
        )
        self.df["section_letter"] = (
            self.df["section"].str.extract(r"\d+([A-Z]?)").fillna("")
        )
        self.df = self.df.sort_values(
            by=["docID", "section_num", "section_letter"]
        ).drop(columns=["section_num", "section_letter"])

        logger.info("Pre-calculating full texts for each document...")
        self._full_texts = (
            self.df.groupby("docID")["sentence"].apply(" ".join).to_dict()
        )
        logger.info("Pre-calculation of full texts complete.")

    def get_all_sentences(self) -> pd.DataFrame:
        """Retrieves all filtered sentences as a DataFrame.

        This method will trigger the initial data loading and processing from
        the raw Parquet file if it has not been loaded yet.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing all sentences for the configured tickers and years.
        """
        self._load_data_if_needed()
        if self.df is None:
            return pd.DataFrame()
        return self.df

    def get_sentences_by_filter(
        self, tickers: List[str], fiscal_years: List[int]
    ) -> pd.DataFrame:
        """Retrieves sentences based on a filter of tickers and fiscal years.

        Parameters
        ----------
        tickers : List[str]
            A list of ticker symbols to include.
        fiscal_years : List[int]
            A list of fiscal years to include.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the filtered sentences.
        """
        self._load_data_if_needed()
        if self.df is None:
            return pd.DataFrame()

        filtered_df = self.df[
            self.df["ticker"].isin(tickers) & self.df["fiscal_year"].isin(fiscal_years)
        ]

        return filtered_df.copy()

    def get_full_filing_text(self, ticker: str, fiscal_year: int) -> Optional[str]:
        """Gets the complete, concatenated text for a single SEC filing.

        Parameters
        ----------
        ticker : str
            The ticker symbol for the company.
        fiscal_year : int
            The fiscal year of the filing.

        Returns
        -------
        Optional[str]
            The full text of the filing, or None if not found.
        """
        self._load_data_if_needed()
        if self._full_texts is None or self.df is None:
            return None

        # Find the docID for the given ticker and year. Assumes one 10-K per ticker per year.
        filtered_df = self.df[
            (self.df["ticker"] == ticker) & (self.df["fiscal_year"] == fiscal_year)
        ]
        if not filtered_df.empty:
            docID = filtered_df["docID"].iloc[0]
            return self._full_texts.get(docID)
        return None
