import html, re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import tiktoken

# ---------------------------------------------------------------------
# small helpers -------------------------------------------------------
# ---------------------------------------------------------------------
_BULLET_RE   = re.compile(r"^[•\-\*\u2022]\s*")   # leading bullet-like chars
_WS_RE       = re.compile(r"\s+")                 # collapse whitespace
_CONTROL_RE  = re.compile(r"[\u200B-\u200D\uFEFF]")  # zero-width stuff

def clean_sentence(text: str) -> str:
    """Remove bullets, html escapes, control chars, trim whitespace."""
    text = html.unescape(text)            # &#39; → '
    text = _CONTROL_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# optional map → more readable section headings inside the chunk
SECTION_HEADING = {
    "section_1":  "Item 1 – Business Overview",
    "section_1A": "Item 1A – Risk Factors",
    "section_7":  "Item 7 – MD&A",
    # …extend as you like
}


def explode_reports(
    df: pd.DataFrame,
    text_key: str = "report",
    id_cols: List[str] = ("cik", "ticker", "fiscal_year"),
) -> pd.DataFrame:
    """
    Flatten the nested `report` dict into (one row ↔ one section) and
    keep *only* clean human-readable text.
    """
    recs: list[dict[str, Any]] = []

    for base, payload in zip(df[id_cols].to_dict("records"), df[text_key]):
        for section_key, sent_list in payload.items():
            if not sent_list:
                continue

            cleaned = [s for s in (clean_sentence(s) for s in sent_list) if s]
            if not cleaned:
                continue

            prefix = SECTION_HEADING.get(section_key, "")
            text   = f"{prefix}: " * bool(prefix) + " ".join(cleaned)

            recs.append({
                **base,
                "section": section_key,
                "text":    text,
                "doc_id":  f"{base['cik']}_{base['fiscal_year']}_{section_key}",
            })

    return pd.DataFrame(recs)



enc = tiktoken.encoding_for_model("gpt-3.5-turbo")   # ≈ 4 chars / token on avg

def token_chunks(text, size=500, overlap_words=3):
    toks = enc.encode(text)
    i, chunks = 0, []
    while i < len(toks):
        window = enc.decode(toks[i:i+size])
        # enforce 3-word overlap by *backing up* that many words next round
        i += size - overlap_words*4              # 4 ≈ avg chars per word/token
        chunks.append(window)
    return chunks

def explode_news(df_news, size=500):
    rows = []
    for _, row in df_news.iterrows():
        for n, chunk in enumerate(token_chunks(row["content"])):     # or “description”
            rows.append({
                "id": f"news-{row['id']}-{n}",
                "source": "news",
                "ticker": row["tickers"][0],
                "fiscal_year": pd.to_datetime(row["published_utc"]).year - 1,
                "section": None,
                "chunk_id": n,
                "text": chunk,
            })
    return pd.DataFrame(rows)


