# %% [markdown]
# # Evaluation
#
# Now that we have our pipelines developed, embeddings done, chunks optimized and created, we can evaluate its performance. You can do it with the code below, but I'll directly load the results.
#
# Please note that due to my own sloppiness, I ran several different evaluations and saved them to different csv's. I aggregated the results together into a csv.

# %%
from pathlib import Path

from src.pipeline_factory import build_and_evaluate
from src.preprocessing.chunkers import ChunkingConfig

# Add project root to path for imports
# sys.path.append(str(Path(__file__).parent.parent))


# -----------------------------------------------------
# End-to-end pipeline + evaluation in one call
# -----------------------------------------------------


if __name__ == "__main__":
    root_dir = Path.cwd()

    print(root_dir)
    df = build_and_evaluate(
        raw_df_path=root_dir / "data" / "raw" / "df_filings_full.parquet",
        chunk_cfg=ChunkingConfig(
            target_tokens=150, overlap_tokens=50, hard_ceiling=500
        ),
        num_questions=300,
        methods=[
            "rag",
            "reranked_rag",
            "ensemble_rerank_rag",
            "unfiltered",
            "baseline",
        ],
        k_values=[1, 3, 5, 7, 10],
        output_dir=root_dir / "data" / "results" / "final" / "final_results_0624.csv",
        use_docker=True,
    )

# %%
