# src/chunkers.py
from __future__ import annotations
import html, re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import uuid
import numpy as np

import pandas as pd
import tiktoken

# Create a consistent namespace for the project. This was generated once
# with `uuid.uuid4()` and is now a constant to ensure all UUIDs are from
# the same deterministic space.
CHUNK_UUID_NAMESPACE = uuid.UUID("f6db7265-c999-4618-8328-65dfdeb0600a")

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
    "5":  "Market for Registrant's Common Equity, Related Stockholder Matters "
          "and Issuer Purchases of Equity Securities",
    "6":  "Selected Financial Data",
    "7":  "Management's Discussion and Analysis of Financial Condition and "
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
    embedding: Optional[List[float]] = None

    def to_dict(self):
        """
        Convert the chunk to a dictionary with JSON-serializable types.
        """
        # Ensure all metadata values are standard Python types
        serializable_metadata = {}
        for key, value in self.metadata.items():
            if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                serializable_metadata[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                serializable_metadata[key] = float(value)
            else:
                serializable_metadata[key] = value

        return {
            "id": self.id,
            "text": self.text,
            "metadata": serializable_metadata,
            "embedding": self.embedding,
        }

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

    def run(self, df_sentences: pd.DataFrame) -> List[Chunk]:
        """
        Processes a DataFrame of sentences and returns a list of chunks.

        The input DataFrame must have columns: 'sentence', 'sentence_token_count',
        and metadata columns ('ticker', 'fiscal_year', 'section', 'item').
        """
        all_chunks: list[Chunk] = []

        # Group by document section to process each section independently
        for (ticker, fiscal_year, section, item), group in df_sentences.groupby([ 'ticker', 'fiscal_year', 'section', 'item']):
            sentences = group['sentence'].tolist()
            sent_token_lens = group['sentence_token_count'].tolist()
            
            section_chunks = self._chunk_sentence_group(
                sentences=sentences,
                sent_token_lens=sent_token_lens,
                ticker=ticker,
                fiscal_year=fiscal_year,
                section=section,
                item=item
            )
            all_chunks.extend(section_chunks)
            
        return all_chunks

    def _chunk_sentence_group(
        self,
        sentences: List[str],
        sent_token_lens: List[int],
        *,
        ticker: str,
        fiscal_year: int,
        section: str,
        item: str
    ) -> List[Chunk]:
        """Helper to process a group of sentences for a single document section."""
        chunks, buf, buf_tokens = [], [], 0
        i = 0
        while i < len(sentences):
            n_tok = sent_token_lens[i]

            # Flush buffer if adding the sentence would cross soft limit
            if buf_tokens and buf_tokens + n_tok > self.target:
                chunks.extend(
                    self._emit_chunk(" ".join(buf), ticker, fiscal_year,
                                     section, item, len(chunks))
                )
                buf, buf_tokens = self._apply_overlap(buf, sent_token_lens, i)

            # Handle single über-long sentence
            if n_tok >= self.ceiling:
                print(f"⚠️ Encountered oversized sentence ({n_tok} tokens). Forcibly slicing.")
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

    # ----------------------------- internals ------------------------------
    @staticmethod
    def _sentence_split(text: str) -> List[str]:
        # This is now only a fallback if NLTK isn't available
        try:
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
        except Exception:
            return SENT_SPLIT_RE.split(text)

    @staticmethod
    def _preprocess(raw: str) -> str:
        txt = html.unescape(raw or "")
        txt = _BULLETS.sub("", txt)
        txt = _MULTI_WS.sub(" ", txt)
        return txt.strip()

    def _apply_overlap(self, buf: List[str], tok_lens: List[int], idx: int):
        leftover, tok_cnt = [], 0
        j = idx - 1
        while j >= 0 and tok_cnt < self.overlap:
            leftover.insert(0, buf.pop())
            tok_cnt += tok_lens[j]
            j -= 1
        return leftover, tok_cnt

    def _force_slice(self, sentence: str, n_tok: int) -> List[str]:
        """Aggressively slice a very long sentence into target-sized pieces."""
        toks = _tok(sentence)
        return [_detok(toks[i:i + self.target]) for i in range(0, n_tok, self.target)]

    # main emitter
    def _emit_chunk(self,
                    chunk_text: str,
                    ticker: str,
                    fiscal_year: int,
                    section: str,
                    item: str,
                    seq: int) -> List[Chunk]:

        item_desc = SEC_10K_ITEMS.get(item, "")
        
        # Final safety check: split any chunk that is still too large
        # This can happen if a single "sentence" from the source is massive.
        final_chunks: list[Chunk] = []
        toks = _tok(chunk_text)
        
        for i, start_tok in enumerate(range(0, len(toks), self.target)):
            slice_toks = toks[start_tok : start_tok + self.target]
            slice_txt = _detok(slice_toks).strip()

            if not slice_txt:
                continue
                
            human_readable_id = f"{ticker}_{fiscal_year}_{item}_{seq}" + (f"_{i}" if i > 0 else "")
            
            # Best Practice: Combine metadata and content for a unique, deterministic ID
            unique_string = f"{ticker}|{fiscal_year}|{item}|{seq}|{i}|{slice_txt}"
            deterministic_uuid = str(uuid.uuid5(CHUNK_UUID_NAMESPACE, unique_string))
            
            final_chunks.append(
                Chunk(
                    id=deterministic_uuid,
                    text=slice_txt,
                    metadata={
                        "ticker": ticker,
                        "fiscal_year": fiscal_year,
                        "item": item,
                        "item_desc": item_desc,
                        "human_readable_id": human_readable_id
                    },
                )
            )
        return final_chunks


if __name__ == "__main__":
    from filing_exploder import FilingExploder
    exploder   = FilingExploder()
    chunker    = SmartChunker(target_tokens=350, hard_ceiling=800, overlap_tokens=50)

    df_filings = pd.read_csv("/Users/jon/GitHub/dowjones-takehome/data/df_filings.csv")
    df_filings = df_filings[df_filings['fiscal_year'].between(2012, 2019)]
    df_exploded = exploder.explode(df=df_filings)
    chunks = chunker.run(df_exploded)
    for i in chunks[:5]:
        print(i)
        print("-"*50)
