from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Dict, Any


import tiktoken
ENC = tiktoken.encoding_for_model("gpt-3.5-turbo")
_tok = ENC.encode
_detok = ENC.decode


# src/exploder.py
import html, re, ast
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# reusable cleaners (same logic you had)
# ──────────────────────────────────────────────────────────────────────────────
_BULLET_RE   = re.compile(r"^[•\-\*\u2022]\s*")
_WS_RE       = re.compile(r"\s+")
_CONTROL_RE  = re.compile(r"[\u200B-\u200D\uFEFF]")

def clean_sentence(text: str) -> str:
    text = html.unescape(text)
    text = _CONTROL_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    return _WS_RE.sub(" ", text).strip()


SECTION_HEADING: Dict[str, str] = {
    "section_1":  "Item 1 – Business Overview",
    "section_1A": "Item 1A – Risk Factors",
    "section_7":  "Item 7 – MD&A",
    # extend as needed
}


# ──────────────────────────────────────────────────────────────────────────────
# exploder class
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class FilingExploder:
    """
    Turn a filings DataFrame with a nested `report` column
    into a flat, one-row-per-section DataFrame.
    """

    id_cols: List[str] = None          # override if your schema differs
    text_key: str = "report"

    def __post_init__(self):
        if self.id_cols is None:
            self.id_cols = ["cik", "ticker", "fiscal_year"]

    # public ------------------------------------------------------------
    def explode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flatten each 10-K into one row per Item / section.

        Returns
        -------
        DataFrame with
        ┌───────────────────────────────────────────────────────────┐
        │  cik | ticker | fiscal_year | section | item | text      │
        │                            |           | doc_id          │
        └───────────────────────────────────────────────────────────┘
        """
        df = df.copy()
        # parse JSON strings → dict
        df[self.text_key] = df[self.text_key].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

        records: List[Dict[str, Any]] = []

        for base, payload in zip(df[self.id_cols].to_dict("records"),
                                 df[self.text_key]):
            if not isinstance(payload, dict):
                continue

            for section_key, sent_list in payload.items():
                if not sent_list:
                    continue

                # ── 1) clean sentences ─────────────────────────────
                cleaned = [s for s in (clean_sentence(s) for s in sent_list) if s]
                if not cleaned:
                    continue

                # ── 2) human-readable heading (optional) ──────────
                prefix = SECTION_HEADING.get(section_key, "")
                text   = f"{prefix}: " * bool(prefix) + " ".join(cleaned)

                # ── 3) derive “item” label ────────────────────────
                #     section_1A  →  1A
                item = section_key.replace("section_", "")

                # prefer item in doc_id; change if you’d rather keep section_
                doc_id = f"{base['cik']}_{base['fiscal_year']}_{item}"

                records.append(
                    {
                        **base,                  # cik, ticker, fiscal_year
                        "section": section_key,  # keep original key if useful
                        "item":    item,         # NEW COLUMN
                        "text":    text,
                        "doc_id":  doc_id,
                    }
                )

        return pd.DataFrame(records)

if __name__ == "__main__":
    exploder = FilingExploder()
    df_filings = pd.read_csv("/Users/jon/GitHub/dowjones-takehome/data/df_filings.csv")
    df_exploded = exploder.explode(df=df_filings)
    print(df_exploded.head())