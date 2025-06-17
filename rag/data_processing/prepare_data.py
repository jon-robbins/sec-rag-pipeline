"""
Data Preparation Script for SEC Filings
=========================================

This script processes the raw sentence-level data from the Hugging Face
Parquet file into an analysis-ready format.

The main function, `prepare_and_save_data`, performs the following steps:
1.  **Queries the raw data** using DuckDB to filter for specific tickers,
    unnest the ticker list, and extract `fiscal_year` and `section`.
2.  **Aggregates the data** using pandas to create two key columns for each filing:
    *   `report`: A structured dictionary mapping each section to a list of its sentences.
    *   `full_text`: A single string containing all sentences from the filing, concatenated.
3.  **Saves the processed data** to a new Parquet file, `data/filings_processed.parquet`,
    which becomes the single source of truth for the rest of the RAG pipeline.
"""
import duckdb
import pandas as pd
from collections import defaultdict
from pathlib import Path
import json

def prepare_and_save_data(
    source_path: str = "data/df_filings_full.parquet",
    output_path: str = "data/filings_processed.parquet",
    tickers_of_interest: list[str] = None
):
    """
    Processes raw SEC data into a structured format for the RAG pipeline.
    """
    if tickers_of_interest is None:
        tickers_of_interest = ['AAPL', 'META', 'TSLA', 'NVDA', 'AMZN']

    print("🔎 Starting data preparation...")
    
    # --- 1. Query and parse with DuckDB ---
    ticker_list_str = "','".join(tickers_of_interest)
    query = f"""
    WITH base AS (
        SELECT 
            reportDate, sentence, sentenceID, docID, UNNEST(tickers) AS ticker
        FROM read_parquet('{source_path}')
        WHERE tickers IS NOT NULL AND len(tickers) > 0
    ),
    filtered AS (
        SELECT * FROM base 
        WHERE ticker IN ('{ticker_list_str}')
    ),
    parsed AS (
        SELECT 
            ticker, sentence, sentenceID, docID,
            CAST(RIGHT(docID, 4) AS INTEGER) AS fiscal_year,
            regexp_extract(sentenceID, 'section_([^_]+)', 1) AS section
        FROM filtered
    )
    SELECT * FROM parsed WHERE section IS NOT NULL;
    """
    
    print("Executing DuckDB query to extract and parse filings...")
    df_sample = duckdb.query(query).to_df()
    print(f"✅ Found {len(df_sample)} sentences for specified tickers.")

    # --- 2. Aggregate with Pandas ---
    print("Aggregating sentences into documents...")
    grouped = defaultdict(lambda: {"report": defaultdict(list), "full_text": []})

    for _, row in df_sample.iterrows():
        key = (row["ticker"], row["fiscal_year"], row["docID"])
        grouped[key]["report"][row["section"]].append(row["sentence"])
        grouped[key]["full_text"].append(row["sentence"])

    records = []
    for (ticker, fiscal_year, docID), content in grouped.items():
        records.append({
            "ticker": ticker,
            "fiscal_year": fiscal_year,
            "docID": docID,
            "report": json.dumps(dict(content["report"])),
            "full_text": " ".join(content["full_text"])
        })

    if not records:
        print("⚠️ No records were generated from the source data.")
        # Create an empty DataFrame with the correct schema
        df_processed = pd.DataFrame(columns=['ticker', 'fiscal_year', 'docID', 'report', 'full_text'])
    else:
        df_processed = pd.DataFrame(records)

    # Save to Parquet using pyarrow engine for robust object handling
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_parquet(output_path, index=False, engine='pyarrow')
    
    print(f"✅ Successfully prepared and saved data to {output_path}")
    print(f"   - {len(df_processed)} documents processed.")
    print(f"   - Columns: {list(df_processed.columns)}")

if __name__ == '__main__':
    prepare_and_save_data() 