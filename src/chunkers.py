# src/chunkers.py
from __future__ import annotations
import logging, re, math
from dataclasses import dataclass
from typing import List, Dict, Any
import pandas as pd

import tiktoken
ENC = tiktoken.encoding_for_model("gpt-3.5-turbo")
_tok = ENC.encode
_detok = ENC.decode

# text preproc helpers
import html
_BULLETS   = re.compile(r"^[\s»\-–•\*]+\s*", re.MULTILINE)   # line-leading bullets/dashes
_MULTI_WS  = re.compile(r"\s+")  

# ──────────────────────────────────────────────────────────────────────────────
# Data container
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    id: str              # e.g. TSLA_2022_1A_4
    text: str
    metadata: Dict[str, Any]   # ticker, fiscal_year, section, item


# ──────────────────────────────────────────────────────────────────────────────
# The new chunker
# ──────────────────────────────────────────────────────────────────────────────
class SmartChunker:
    """
    Target-size / hard-ceiling sentence-aware chunker.

    Parameters
    ----------
    target_tokens : int
        Soft limit.  We *aim* to stop adding sentences once we cross this.
    hard_ceiling  : int
        Absolute max.  If a single sentence is longer, we split it
        token-wise so the resulting slices stay < hard_ceiling.
    overlap_tokens : int
        Back-overlap between consecutive chunks (token count, *after* the
        chunk has been built).
    """

    SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")  # quick sentence splitter

    def __init__(
        self,
        target_tokens: int = 350,
        hard_ceiling: int = 800,
        overlap_tokens: int = 50,
    ) -> None:
        if hard_ceiling <= target_tokens:
            raise ValueError("hard_ceiling must be larger than target_tokens")

        self.target = target_tokens
        self.ceiling = hard_ceiling
        self.overlap = overlap_tokens

    # ------------------------------------------------------------------ public
    def chunk_text(
        self,
        text: str,
        *,
        ticker: str,
        fiscal_year: int,
        section: str,
        item: str,
    ) -> List[Chunk]:
        """
        Returns a list of `Chunk` objects for *one* source document/section.
        """
        text = self._preprocess(text)
        if not text:
            return []

        sentences = self._sentence_split(text)
        sent_tok_lens = [len(_tok(s)) for s in sentences]

        chunks, buf, buf_tokens = [], [], 0
        i = 0
        while i < len(sentences):
            sent = sentences[i]
            n_tok = sent_tok_lens[i]

            # Would adding this sentence exceed the soft target?
            if buf_tokens and buf_tokens + n_tok > self.target:
                # emit current buffer
                chunks.extend(
                    self._finalise_chunk(" ".join(buf), ticker, fiscal_year, section, item, len(chunks))
                )
                # prepare overlap window
                buf, buf_tokens = self._apply_overlap(buf, sent_tok_lens, i)

            # Now handle *very* long single sentences
            if n_tok >= self.ceiling:
                slices = self._force_slice(sent, n_tok)
                for slice_txt in slices:
                    chunks.extend(
                        self._finalise_chunk(slice_txt, ticker, fiscal_year, section, item, len(chunks))
                    )
                i += 1
                continue

            # normal path: add sentence to buffer
            buf.append(sent)
            buf_tokens += n_tok
            i += 1

        # leftover
        if buf:
            chunks.extend(
                self._finalise_chunk(" ".join(buf), ticker, fiscal_year, section, item, len(chunks))
            )

        return chunks

    # ---------------------------------------------------------- helpers
    def _sentence_split(self, text: str) -> List[str]:
        # Use NLTK / spaCy if you already have them; fall back otherwise
        try:
            from nltk.tokenize import sent_tokenize
            return sent_tokenize(text)
        except Exception:
            return self.SENT_SPLIT_RE.split(text)

    def _apply_overlap(self, buf: List[str], sent_tok_lens: List[int], idx: int):
        """
        Return a new buffer & token count so that `overlap` tokens from the
        *end* of the previous chunk become the *start* of the next.
        """
        leftover, tok_cnt = [], 0
        j = idx - 1
        while j >= 0 and tok_cnt < self.overlap:
            leftover.insert(0, buf.pop())   # remove from end of buf
            tok_cnt += sent_tok_lens[j]
            j -= 1
        return leftover, tok_cnt

    def _force_slice(self, sentence: str, n_tok: int) -> List[str]:
        """
        Split an over-long single sentence into ≤ ceiling slices *on tokens*.
        """
        toks = _tok(sentence)
        out, i = [], 0
        while i < n_tok:
            out.append(_detok(toks[i : i + self.ceiling]))
            i += self.ceiling
        return out

    def _finalise_chunk(
        self,
        chunk_text: str,
        ticker: str,
        fiscal_year: int,
        section: str,
        item: str,
        n: int,
    ) -> List[Chunk]:
        """Return either one chunk or (rare) multiple if ceiling is violated."""
        toks = _tok(chunk_text)
        if len(toks) <= self.ceiling:
            return [
                Chunk(
                    id=f"{ticker}_{fiscal_year}_{item}_{n}",
                    text=chunk_text.strip(),
                    metadata={
                        "ticker": ticker,
                        "fiscal_year": fiscal_year,
                        "section": section,
                        "item": item,
                    },
                )
            ]

        # if buffer still too big (e.g. when target>ceiling), slice further
        chunks, i, seq = [], 0, 0
        while i < len(toks):
            slice_txt = _detok(toks[i : i + self.ceiling])
            chunks.append(
                Chunk(
                    id=f"{ticker}_{fiscal_year}_{item}_{n}_{seq}",
                    text=slice_txt.strip(),
                    metadata={
                        "ticker": ticker,
                        "fiscal_year": fiscal_year,
                        "section": section,
                        "item": item,
                    },
                )
            )
            seq += 1
            i += self.ceiling
        return chunks
    
    @staticmethod
    def _preprocess(raw: str) -> str:
        """
        Cheap, semantics-safe cleanup:
        • HTML entities → characters
        • strip leading list bullets / dashes
        • collapse multiple spaces / newlines
        """
        if not raw:
            return ""

        txt = html.unescape(raw)
        txt = _BULLETS.sub("", txt)      # per-line bullet removal
        txt = _MULTI_WS.sub(" ", txt)    # squeeze whitespace
        return txt.strip()
    def run(self, df: pd.DataFrame) -> List[Chunk]:
        """
        Chunk a DataFrame of text.
        """
        df['text_clean'] = df['text'].map(self._preprocess)

        all_chunks = []
        for _, r in df.iterrows():
            all_chunks += self.chunk_text(
                r["text_clean"],
                ticker=r["ticker"],
                fiscal_year=r["fiscal_year"],
                section=r["section"],
                item=r["item"],
            )
        return all_chunks

if __name__ == "__main__":
    from filing_exploder import FilingExploder
    exploder   = FilingExploder()
    chunker    = SmartChunker(target_tokens=350, hard_ceiling=800, overlap_tokens=50)

    df_filings = pd.read_csv("/Users/jon/GitHub/dowjones-takehome/data/df_filings.csv")
    df_filings = df_filings[df_filings['fiscal_year'].between(2012, 2019)]
    df_filings_exploded = exploder.explode(df=df_filings)
    chunks = chunker.run(df_filings_exploded)
    print(chunks[:5])