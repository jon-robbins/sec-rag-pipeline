# vector_store.py
from __future__ import annotations

import os
import warnings
from typing import List, Dict, Any, Iterable

from qdrant_client import QdrantClient, models
import openai
import tiktoken


class VectorStore:
    """
    Thin convenience wrapper around Qdrant that

    • embeds text with OpenAI (or accepts pre-computed vectors)
    • enforces length / dimensionality consistency
    • stores the source text inside the payload for easy debugging
    """

    def __init__(
        self,
        collection_name: str = "sec_filings",
        dim: int = 1536,                      # text-embedding-3-small
        client: QdrantClient | None = None,
        model: str = "text-embedding-3-small",
        openai_key: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.dim             = dim
        self.model           = model
        self.client          = client or QdrantClient(":memory:")
        self.openai          = openai.OpenAI(api_key=openai_key or os.getenv("OPENAI_API_KEY"))
        self._enc            = tiktoken.encoding_for_model(model)

    # ── Embedding ──────────────────────────────────────────────────────────
    def _batch_iter(
        self, texts: Iterable[str], max_tokens: int = 280_000
    ) -> Iterable[list[str]]:
        """Yield batches that stay below `max_tokens`."""
        current, tok_sum = [], 0
        for txt in texts:
            n_tok = len(self._enc.encode(txt))
            if tok_sum + n_tok > max_tokens and current:
                yield current
                current, tok_sum = [txt], n_tok
            else:
                current.append(txt)
                tok_sum += n_tok
        if current:
            yield current

    def embed(self, texts: List[str]) -> List[List[float]]:
        """OpenAI embedding with token-safe batching."""
        out: list[list[float]] = []
        for batch in self._batch_iter(texts):
            resp = self.openai.embeddings.create(input=batch, model=self.model)
            out.extend(r.embedding for r in resp.data)
        return out

    # ── Collection init / reset ────────────────────────────────────────────
    def init_collection(self) -> None:
        """Create or reset the target collection."""
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=self.dim, distance=models.Distance.COSINE),
        )

    # ── Upsert ─────────────────────────────────────────────────────────────
    def upsert(
        self,
        *,
        metas: List[Dict[str, Any]],
        texts: List[str] | None = None,
        vectors: List[List[float]] | None = None,
        ids: List[Any] | None = None,
        batch_size: int = 1024,
        parallel: int = 4,
    ) -> None:
        """
        Add / update points.  One of `texts` **or** `vectors` must be provided.

        • If `vectors` is omitted -> they are computed from `texts`
        • `metas` is mandatory (at minimum ticker / fiscal_year …)
        """

        if vectors is None:
            if texts is None:
                raise ValueError("Provide either `texts` or `vectors`.")
            vectors = self.embed(texts)  # ⇦ may be large but batched internally
        elif texts is None:
            # You passed vectors directly – keep payload lean
            texts = [""] * len(vectors)

        # ── consistency checks ────────────────────────────────────────────
        assert len(vectors) == len(metas) == len(texts), (
            f"Length mismatch: vectors={len(vectors)}, "
            f"metas={len(metas)}, texts={len(texts)}"
        )

        if ids is None:
            ids = list(range(len(vectors)))
        else:
            assert len(ids) == len(vectors), "`ids` length must match vectors"

        vec_dim = len(vectors[0])
        if vec_dim != self.dim:
            warnings.warn(
                f"Vector dimension {vec_dim} ≠ configured dim {self.dim}. "
                "Re-creating the collection with the new size."
            )
            self.dim = vec_dim
            self.init_collection()

        # final payload assembly
        payloads = [{**meta, "text": txt} for meta, txt in zip(metas, texts)]

        self.client.upload_collection(
            collection_name=self.collection_name,
            ids=list(range(len(vectors))),
            vectors=vectors,
            payload=payloads,
            batch_size=batch_size,
            parallel=parallel,
        )

    # ── Search ────────────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        q_vec = self.embed([query])[0]
        hits  = self.client.search(
            collection_name=self.collection_name,
            query_vector=q_vec,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "score": hit.score,
                **hit.payload,           # ticker, year, section, …
                "text": hit.payload.get("text", ""),
            }
            for hit in hits
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Example usage
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pickle
    import pandas as pd
    from chunkers import SmartChunker
    from filing_exploder import FilingExploder
    from embedding import embed_texts

    # 1) explode + chunk
    df      = pd.read_csv("data/df_filings.csv")
    items   = FilingExploder().explode(df)
    chunks  = SmartChunker().run(items)         # .run returns list[Chunk]

    # 2) load pre-made embeddings (optional)
    # with open("embeddings/chunks_embeddings.pkl", "rb") as f:
        # ids, vecs, metas = pickle.load(f)
    ids, vecs, metas = embed_texts(chunks)
    # 3) initialise vector store
    vs = VectorStore()
    vs.init_collection()

    # 4) upsert without re-embedding
    vs.upsert(vectors=vecs, metas=metas, ids=ids, texts=[c.text for c in chunks])

    # 5) query
    print(vs.search("What risks did Meta mention in 2019?", top_k=5))
