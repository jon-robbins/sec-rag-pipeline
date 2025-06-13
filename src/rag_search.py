import os
from typing import List, Dict, Any

import openai                    # pip install --upgrade openai
from qdrant_client import QdrantClient, models

# ---------------------------------------------------------------------
# config – keep identical to what you used when BUILDING the collection
# ---------------------------------------------------------------------
EMBED_MODEL        = "text-embedding-3-small"      # or "text-embedding-ada-002"
QDRANT_COLLECTION  = "sec_filings"
TOP_K              = 12                            # passages to fetch

# ---------------------------------------------------------------------
# clients
# ---------------------------------------------------------------------
oa      = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant  = QdrantClient(":memory:")                 # same as before

# ---------------------------------------------------------------------
# helper – embed ONE question (or a batch if you prefer)
# ---------------------------------------------------------------------
def embed(text: str, model: str = EMBED_MODEL) -> List[float]:
    """
    Returns a single vector for *text* using the same OpenAI model you
    used during indexing.
    """
    resp = oa.embeddings.create(input=[text], model=model)
    return resp.data[0].embedding


# ---------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------
def retrieve(question: str,
             *,
             top_k: int = TOP_K,
             collection: str = QDRANT_COLLECTION) -> List[Dict[str, Any]]:
    """
    Search Qdrant and return a list of dicts:
        {
            "chunk" : <str>,
            "score" : <float>,
            "ticker": ...,
            "fiscal_year": ...,
            "section": ...,
            "item": ...
        }
    """
    qvec = embed(question)                               # ① embedding

    hits = qdrant.search(                                # ② vector search
        collection_name=collection,
        query_vector=qvec,
        limit=top_k,
        with_payload=True,                               # bring back metadata
    )

    results = []
    for h in hits:
        pay = h.payload                  # whatever you stored in metas
        results.append({
            "chunk"      : pay.get("document", ""),       # change key if needed
            "score"      : h.score,
            "ticker"     : pay.get("ticker"),
            "fiscal_year": pay.get("fiscal_year"),
            "section"    : pay.get("section"),
            "item"       : pay.get("item"),
        })
    return results


# ---------------------------------------------------------------------
# quick sanity-check
# ---------------------------------------------------------------------
if __name__ == "__main__":
    question = "What risk factors did Meta mention in its 2019 10-K?"

    passages  = retrieve(question)

    print("Top 3 passages\n" + "="*60)
    for n, p in enumerate(passages[:3], 1):
        print(f"\n--- #{n}  ({p['score']:.4f})  "
              f"{p['ticker']} {p['fiscal_year']} Item {p['item']}")
        print(p["chunk"][:400], "…")
