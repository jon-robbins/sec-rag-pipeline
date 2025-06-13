import json
import glob
import os
import pandas as pd
from pathlib import Path
import re
from datasets import load_dataset
from typing import Union
import ast

def load_filings(data_dir: str = "../data", 
                 tickers: list[str] = ["TSLA", "AAPL"],
                 refresh: bool = False) -> pd.DataFrame:
    
    data_dir = Path(data_dir)
    if not refresh and os.path.exists(data_dir / "df_filings.csv"):
        df = pd.read_csv(data_dir / "df_filings.csv")
        print(f"Loaded {len(df)} filings from {data_dir / 'df_filings.csv'}")
        return df
    
    print(f"Downloading filings for {tickers} to {data_dir / 'df_filings.csv'}")

    rows = []
    tickers_set = set(t.upper().strip() for t in tickers)

    total_companies = 0
    matched_ticker_count = 0
    ten_k_count = 0

    for split in ["train", "test", "val"]:
        files = glob.glob(str(Path(data_dir) / split / "*.jsonl"))
        for fpath in files:
            with open(fpath, "r") as f:
                for line in f:
                    try:
                        company = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    total_companies += 1
                    company_tickers = [t.upper().strip() for t in company.get("tickers", [])]

                    if not any(t in tickers_set for t in company_tickers):
                        continue

                    matched_ticker_count += 1

                    for filing in company.get("filings", []):
                        if filing.get("form") != "10-K":
                            continue

                        ten_k_count += 1
                        rows.append({
                            "split": split,
                            "cik": company.get("cik"),
                            "ticker": company_tickers[0] if company_tickers else None,
                            "fiscal_year": int(filing.get("filingDate").split("-")[0]) - 1,
                            "doc_id": filing.get("docID"),
                            "report": filing.get("report", None),
                            "returns": filing.get("returns", None),
                        })
    
    df = pd.DataFrame(rows)
    df['report'] = df['report'].apply(ast.literal_eval)
    return df


def load_news(tickers: Union[list[str], str]) -> pd.DataFrame:


    dataset_news = load_dataset("afeng/MTBench_finance_news")

    df_news = dataset_news['train'].to_pandas()

    mask = df_news["description"].str.contains(
        "|".join(map(re.escape, tickers)),   # →  "AAPL|TSLA"
        case=False,                          # ignore-case if you want
        na=False                             # treat NaNs as False
    )

    return df_news[mask]




if __name__ == "__main__":
    df_filings = load_filings(tickers=["TSLA", "AAPL"], refresh=True)
    print(df_filings.head())