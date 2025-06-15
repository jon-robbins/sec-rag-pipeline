# ── src/embedding.py ─────────────────────────────────────────────────────────
import os, time, math, pickle, random
from pathlib import Path
from typing import List, Tuple

import tiktoken
from openai import OpenAI, RateLimitError
from tqdm.auto import tqdm

client  = OpenAI()
ENC     = tiktoken.encoding_for_model("text-embedding-3-small")
CACHE   = Path("embeddings/chunks_embeddings.pkl")


def embed_texts(
    chunks,
    model: str = "text-embedding-3-small",
    batch_size: int = 128,
    refresh: bool = False,
) -> Tuple[List[str], List[List[float]], List[dict]]:
    """
    Embed `chunks` (list[Chunk]) with automatic OpenAI rate-limit handling.

    • keeps/extends a pickle cache
    • obeys 1 M TPM and RPM limits
    • never raises RateLimitError – it just waits and retries
    """
    # ── 1️⃣  load existing cache (if any) ──────────────────────────────────
    known_ids: set[str] = set()
    ids, vecs, metas = [], [], []
    if CACHE.exists() and not refresh:
        with CACHE.open("rb") as f:
            ids, vecs, metas = pickle.load(f)
        known_ids = set(ids)

    # ── 2️⃣  prepare the work list ─────────────────────────────────────────
    todo = [c for c in chunks if c.id not in known_ids]
    if not todo:
        # All chunks are cached - filter and return only requested chunks
        requested_ids = [c.id for c in chunks]
        filtered_ids, filtered_vecs, filtered_metas = [], [], []
        
        for i, chunk_id in enumerate(ids):
            if chunk_id in requested_ids:
                filtered_ids.append(chunk_id)
                filtered_vecs.append(vecs[i])
                filtered_metas.append(metas[i])
        
        return filtered_ids, filtered_vecs, filtered_metas

    pbar = tqdm(total=len(todo), desc="Embedding")
    idx  = 0
    consecutive_429 = 0

    while idx < len(todo):
        # Build a token-safe batch
        cur, tok_sum = [], 0
        while idx < len(todo) and len(cur) < batch_size:
            nxt = todo[idx].text
            if tok_sum + (nt := len(ENC.encode(nxt))) > 900_000 and cur:
                break                # push to next batch
            cur.append(nxt)
            tok_sum += nt
            idx += 1

        try:
            resp = client.embeddings.create(model=model, input=cur)
            vecs.extend([d.embedding for d in resp.data])
            ids.extend([c.id for c in todo[idx - len(cur): idx]])
            metas.extend([c.metadata for c in todo[idx - len(cur): idx]])
            pbar.update(len(cur))

            # successful batch -> reset back-off
            consecutive_429 = 0
            batch_size = min(batch_size * 2, 256)  # grow back slowly

            # persist incremental progress
            with CACHE.open("wb") as f:
                pickle.dump((ids, vecs, metas), f)

        except RateLimitError as e:
            consecutive_429 += 1
            wait = float(
                getattr(e, "response", None)
                and e.response.json()["error"].get("retry_after", 15)
            ) or 15
            # Longer wait after repeated hits
            wait *= 1.5 ** consecutive_429
            wait += random.uniform(0, 3)  # jitter
            print(f"429 → sleeping {wait:.1f}s  (batch={len(cur)})")
            time.sleep(wait)
            # back-off batch size after 2 consecutive 429s
            if consecutive_429 >= 2 and batch_size > 1:
                batch_size = max(1, batch_size // 2)

    pbar.close()
    
    # Filter to return only the requested chunks in the original order
    requested_ids = [c.id for c in chunks]
    filtered_ids, filtered_vecs, filtered_metas = [], [], []
    
    for chunk_id in requested_ids:
        try:
            idx = ids.index(chunk_id)
            filtered_ids.append(ids[idx])
            filtered_vecs.append(vecs[idx])
            filtered_metas.append(metas[idx])
        except ValueError:
            # This shouldn't happen if our logic is correct
            print(f"Warning: chunk {chunk_id} not found in results")
    
    return filtered_ids, filtered_vecs, filtered_metas
