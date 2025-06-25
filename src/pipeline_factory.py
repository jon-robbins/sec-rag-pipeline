from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from src.evaluation.runner import run_full_eval
from src.openai_functions.answer_question import AnswerGenerator
from src.preprocessing.chunkers import Chunk, Chunker, ChunkingConfig
from src.rag.pipeline import RAGPipeline
from src.vector_store.document_store import DocumentStore
from src.vector_store.embedding import EmbeddingManager
from src.vector_store.vector_store import VectorStore

if TYPE_CHECKING:
    import pandas as pd

"""pipeline_factory.py
Utility functions to create a ready-to-use `RAGPipeline` in one call.
The helper hides all boilerplate: loading raw sentences, chunking, embedding,
upserting into Qdrant, and instantiating the pipeline object.
"""

__all__ = ["build_pipeline", "build_and_evaluate"]


def build_pipeline(
    *,
    raw_df_path: Path,
    chunk_cfg: Optional[ChunkingConfig] = None,
    use_docker: bool = False,
) -> RAGPipeline:
    """End-to-end helper that returns a `RAGPipeline` ready for evaluation.

    Parameters
    ----------
    raw_df_path : Path
        Parquet file produced by the sentence extraction step.
    chunk_cfg : ChunkingConfig, optional
        Overrides the default chunking parameters.
    use_docker : bool
        If True, connect to an external Qdrant container instead of the in-memory
        instance.
    """

    # 1. Load raw sentences and aggregate to section-level documents
    doc_store = DocumentStore(raw_data_path=raw_df_path)
    section_df = doc_store.get_documents_for_chunking()

    # 2. Chunk
    chunker = Chunker(config=chunk_cfg or ChunkingConfig())
    chunk_df, stats = chunker.chunk_dataframe(section_df)
    print(
        f"🔹 Chunking complete — {stats['total_chunks']} chunks, "
        f"avg {stats['avg_tokens_per_chunk']} words"
    )

    chunks = [Chunk(**row) for row in chunk_df.to_dict("records")]

    # 3. Embed
    emb_mgr = EmbeddingManager()
    embeddings = emb_mgr.embed_texts_in_batches([c.text for c in chunks])
    for c, e in zip(chunks, embeddings):
        c.embedding = e

    # 4. Vector store
    vstore = VectorStore(use_docker=use_docker, embedding_manager=emb_mgr)
    # VectorStore.upsert_chunks expects each dict to have keys: id, text, metadata
    payloads = [
        {
            "id": c.id,
            "text": c.text,
            "metadata": c.metadata,
        }
        for c in chunks
    ]

    vstore.upsert_chunks(payloads, embeddings)
    print("🔹 Vector store populated")

    # 5. Assemble pipeline
    pipeline = RAGPipeline(vstore, AnswerGenerator())

    # Expose chunks and document store so evaluator can reuse them
    pipeline.chunks = chunks  # type: ignore[attr-defined]
    pipeline.get_chunks = lambda: chunks  # type: ignore[attr-defined]
    pipeline.document_store = doc_store  # type: ignore[attr-defined]

    return pipeline


def build_and_evaluate(
    *,
    raw_df_path: Path,
    chunk_cfg: Optional[ChunkingConfig] = None,
    num_questions: int = 300,
    methods: Optional[list[str]] = None,
    k_values: Optional[list[int]] = None,
    output_dir: Path = Path("data/results/final"),
    use_docker: bool = False,
) -> pd.DataFrame:
    """One-liner for notebooks: chunks ➜ pipeline ➜ evaluation ➜ DF.

    Returns the flattened evaluation `pandas.DataFrame`. The underlying helper
    (`run_full_eval`) also persists a *consolidated_results_fixed_tokens.csv*
    file under `output_dir` for convenience.
    """

    # ---------------- Parameter validation ----------------
    if not raw_df_path.exists():
        raise FileNotFoundError(raw_df_path)

    if chunk_cfg is not None and not isinstance(chunk_cfg, ChunkingConfig):
        raise TypeError("chunk_cfg must be a ChunkingConfig instance or None")

    if num_questions <= 0:
        raise ValueError("num_questions must be positive")

    allowed_methods = {
        "rag",
        "reranked_rag",
        "ensemble_rerank_rag",
        "unfiltered",
        "baseline",
        "web_search",
    }
    if methods is not None:
        illegal = [m for m in methods if m not in allowed_methods]
        if illegal:
            raise ValueError(
                f"Unknown methods: {illegal}. Allowed: {sorted(allowed_methods)}"
            )

    if k_values is not None and any(k <= 0 for k in k_values):
        raise ValueError("k_values must be positive integers")

    # -------------------------------------------------------

    pipeline = build_pipeline(
        raw_df_path=raw_df_path,
        chunk_cfg=chunk_cfg,
        use_docker=use_docker,
    )

    # Run the full evaluation; this function handles the flattening and CSV persistence.
    df_flat = run_full_eval(
        pipeline=pipeline,
        num_questions=num_questions,
        methods=methods,
        k_values=k_values,
        output_dir=output_dir,
        root_dir=Path.cwd(),
    )

    return df_flat
