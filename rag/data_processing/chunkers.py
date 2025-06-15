# src/chunkers.py
from __future__ import annotations
import html, re
from dataclasses import dataclass
from typing import List, Dict, Any

import pandas as pd
import tiktoken

ENC = tiktoken.encoding_for_model("gpt-3.5-turbo")
_tok   = ENC.encode
_detok = ENC.decode

# ──────────────────────────────── SEC mapping ────────────────────────────────
SEC_10K_ITEMS = {
    "1":  "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2":  "Properties",
    "3":  "Legal Proceedings",
    "4":  "Mine Safety Disclosures",
    "5":  "Market for Registrant’s Common Equity, Related Stockholder Matters "
          "and Issuer Purchases of Equity Securities",
    "6":  "Selected Financial Data",
    "7":  "Management’s Discussion and Analysis of Financial Condition and "
          "Results of Operations (MD&A)",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8":  "Financial Statements and Supplementary Data",
    "9":  "Changes in and Disagreements with Accountants on Accounting and "
          "Financial Disclosure",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions that Prevent Inspections",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners and Management and "
          "Related Stockholder Matters",
    "13": "Certain Relationships and Related Transactions, and Director "
          "Independence",
    "14": "Principal Accounting Fees and Services",
    "15": "Exhibits and Financial Statement Schedules",
}

# ────────────────────────────── regex helpers ────────────────────────────────
_BULLETS  = re.compile(r"^[\s»\-–•\*]+\s*", re.MULTILINE)
_MULTI_WS = re.compile(r"\s+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")  # fallback splitter

# ────────────────────────────── data container ───────────────────────────────
@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any]   # ticker, fiscal_year, item, item_desc, section

# ───────────────────────────────── chunker ───────────────────────────────────
class SmartChunker:
    def __init__(self,
                 target_tokens: int = 350,
                 hard_ceiling: int = 800,
                 overlap_tokens: int = 50) -> None:

        if hard_ceiling <= target_tokens:
            raise ValueError("hard_ceiling must exceed target_tokens")
        self.target   = target_tokens
        self.ceiling  = hard_ceiling
        self.overlap  = overlap_tokens

    # ------------------------------ public API ------------------------------
    def chunk_text(self,
                   text: str,
                   *,
                   ticker: str,
                   fiscal_year: int,
                   section: str,
                   item: str) -> List[Chunk]:

        text = self._preprocess(text)
        if not text:
            return []

        sentences      = self._sentence_split(text)
        sent_token_len = [len(_tok(s)) for s in sentences]

        chunks, buf, buf_tokens = [], [], 0
        i = 0
        while i < len(sentences):
            n_tok = sent_token_len[i]

            # flush buffer if adding the sentence would cross soft limit
            if buf_tokens and buf_tokens + n_tok > self.target:
                chunks.extend(
                    self._emit_chunk(" ".join(buf), ticker, fiscal_year,
                                     section, item, len(chunks))
                )
                buf, buf_tokens = self._apply_overlap(buf, sent_token_len, i)

            # handle single über-long sentence
            if n_tok >= self.ceiling:
                for slice_txt in self._force_slice(sentences[i], n_tok):
                    chunks.extend(
                        self._emit_chunk(slice_txt, ticker, fiscal_year,
                                         section, item, len(chunks))
                    )
                i += 1
                continue

            buf.append(sentences[i])
            buf_tokens += n_tok
            i += 1

        if buf:
            chunks.extend(
                self._emit_chunk(" ".join(buf), ticker, fiscal_year,
                                 section, item, len(chunks))
            )
        return chunks

    def run(self, df: pd.DataFrame) -> List[Chunk]:
        df["text_clean"] = df["text"].map(self._preprocess)

        all_chunks: list[Chunk] = []
        for _, row in df.iterrows():
            all_chunks += self.chunk_text(
                row["text_clean"],
                ticker=row["ticker"],
                fiscal_year=row["fiscal_year"],
                section=row["section"],
                item=row["item"],
            )
        return all_chunks

    # ----------------------------- internals ------------------------------
    @staticmethod
    def _preprocess(raw: str) -> str:
        txt = html.unescape(raw or "")
        txt = _BULLETS.sub("", txt)
        txt = _MULTI_WS.sub(" ", txt)
        return txt.strip()

    @staticmethod
    def _sentence_split(text: str) -> List[str]:
        try:
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
        except Exception:
            return SENT_SPLIT_RE.split(text)

    def _apply_overlap(self, buf: List[str], tok_lens: List[int], idx: int):
        leftover, tok_cnt = [], 0
        j = idx - 1
        while j >= 0 and tok_cnt < self.overlap:
            leftover.insert(0, buf.pop())
            tok_cnt += tok_lens[j]
            j -= 1
        return leftover, tok_cnt

    def _force_slice(self, sentence: str, n_tok: int) -> List[str]:
        toks = _tok(sentence)
        return [_detok(toks[i:i + self.ceiling]) for i in range(0, n_tok, self.ceiling)]

    # main emitter
    def _emit_chunk(self,
                    chunk_text: str,
                    ticker: str,
                    fiscal_year: int,
                    section: str,
                    item: str,
                    seq: int) -> List[Chunk]:

        item_desc = SEC_10K_ITEMS.get(item, "")
        toks      = _tok(chunk_text)

        # split again if still above hard ceiling (rare)
        chunks: list[Chunk] = []
        for i in range(0, len(toks), self.ceiling):
            slice_txt = _detok(toks[i:i + self.ceiling]).strip()
            chunk_id  = f"{ticker}_{fiscal_year}_{item}_{seq}" + (f"_{i//self.ceiling}" if i else "")
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=slice_txt,
                    metadata={
                        "ticker": ticker,
                        "fiscal_year": fiscal_year,
                        "item": item,
                        "item_desc": item_desc,
                    },
                )
            )
        return chunks


if __name__ == "__main__":
    from filing_exploder import FilingExploder
    exploder   = FilingExploder()
    chunker    = SmartChunker(target_tokens=350, hard_ceiling=800, overlap_tokens=50)

    df_filings = pd.read_csv("/Users/jon/GitHub/dowjones-takehome/data/df_filings.csv")
    df_filings = df_filings[df_filings['fiscal_year'].between(2012, 2019)]
    df_filings_exploded = exploder.explode(df=df_filings)
    chunks = chunker.run(df_filings_exploded)
    for i in chunks[:5]:
        print(i)
        print("-"*50)
