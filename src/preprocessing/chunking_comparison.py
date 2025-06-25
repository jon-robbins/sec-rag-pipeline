from datetime import datetime
from typing import Dict, List

import pandas as pd

from src.evaluation.evaluator import ComprehensiveEvaluator
from src.rag.pipeline import RAGPipeline
from src.utils.config import RESULTS_DIR


def compare_chunking_configs(
    num_questions=20,
    configs: List[Dict] = [
        {"target_tokens": 150, "overlap_tokens": 25, "name": "Small_150_25"},
        {"target_tokens": 300, "overlap_tokens": 50, "name": "Medium_300_50"},
        {"target_tokens": 500, "overlap_tokens": 100, "name": "Large_500_100"},
        {"target_tokens": 750, "overlap_tokens": 150, "name": "XLarge_750_150"},
    ],
):
    """
    Compare chunking configurations using existing pipeline classes.
    """
    results = []

    for i, config in enumerate(configs):
        print(f"\n📊 Configuration {i+1}/{len(configs)}: {config['name']}")
        print(
            f"   Testing: {config['target_tokens']} tokens, {config['overlap_tokens']} overlap"
        )

        try:
            # Initialize pipeline with custom chunking
            from pathlib import Path

            from src.openai_functions.answer_question import AnswerGenerator
            from src.preprocessing.chunkers import Chunker, ChunkingConfig
            from src.utils.client_utils import get_openai_client
            from src.vector_store.document_store import DocumentStore
            from src.vector_store.embedding import EmbeddingManager
            from src.vector_store.vector_store import VectorStore

            # 1. Load data using proper interface
            raw_data_path = Path.cwd() / "data" / "raw" / "df_filings_full.parquet"
            doc_store = DocumentStore(raw_data_path=raw_data_path)
            chunking_documents = doc_store.get_documents_for_chunking()

            # 2. Create chunking configuration
            chunking_config = ChunkingConfig(
                target_tokens=config["target_tokens"],
                overlap_tokens=config["overlap_tokens"],
                hard_ceiling=config.get("hard_ceiling", 500),
            )

            # 3. Chunk the data
            chunker = Chunker(config=chunking_config)
            chunk_df, stats = chunker.chunk_dataframe(chunking_documents)

            # 4. Create chunk objects and generate embeddings
            from src.preprocessing.chunkers import Chunk

            chunks = [Chunk(**row) for row in chunk_df.to_dict("records")]

            embedding_manager = EmbeddingManager()
            texts = [chunk.text for chunk in chunks]
            embeddings = embedding_manager.embed_texts_in_batches(texts)

            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk.embedding = embeddings[i]

            # 5. Set up vector store with the chunks
            vector_store = VectorStore(
                use_docker=False, embedding_manager=embedding_manager
            )

            # Prepare chunks for vector store
            from src.openai_functions.qa_generation import (
                prepare_chunks_for_qa_generation,
            )

            chunk_dicts = prepare_chunks_for_qa_generation(chunks)
            # Filter out None embeddings to satisfy type checker
            embeddings_list = [
                chunk.embedding for chunk in chunks if chunk.embedding is not None
            ]

            # Upsert chunks into vector store
            vector_store.upsert_chunks(chunk_dicts, embeddings_list)

            # 6. Create answer generator and pipeline
            openai_client = get_openai_client()
            answer_generator = AnswerGenerator(openai_client)
            pipeline = RAGPipeline(vector_store, answer_generator)

            # Add chunks info to pipeline for evaluation
            pipeline.chunks = chunks  # type: ignore[attr-defined]
            pipeline.get_chunks = lambda: chunks  # type: ignore[attr-defined]

            # Run evaluation
            evaluator = ComprehensiveEvaluator(Path.cwd(), pipeline)
            eval_results, temp_dir = evaluator.evaluate_all_scenarios(
                num_questions=num_questions
            )

            # Clean up temp directory if it exists
            if temp_dir:
                import shutil

                shutil.rmtree(temp_dir)

            # Extract metrics
            if "rag" in eval_results.get("summary", {}):
                rag_metrics = eval_results["summary"]["rag"]
                result = {
                    "configuration": config["name"],
                    "target_tokens": config["target_tokens"],
                    "overlap_tokens": config["overlap_tokens"],
                    "total_chunks": len(chunks),
                    "recall_at_1": rag_metrics.get("retrieval", {}).get(
                        "recall_at_1", 0
                    ),
                    "recall_at_3": rag_metrics.get("retrieval", {}).get(
                        "recall_at_3", 0
                    ),
                    "recall_at_5": rag_metrics.get("retrieval", {}).get(
                        "recall_at_5", 0
                    ),
                    "recall_at_10": rag_metrics.get("retrieval", {}).get(
                        "recall_at_10", 0
                    ),
                    "mrr": rag_metrics.get("retrieval", {}).get("mrr", 0),
                    "rouge1_f": rag_metrics.get("rouge", {})
                    .get("rouge1", {})
                    .get("fmeasure", 0),
                    "rouge2_f": rag_metrics.get("rouge", {})
                    .get("rouge2", {})
                    .get("fmeasure", 0),
                    "rougeL_f": rag_metrics.get("rouge", {})
                    .get("rougeL", {})
                    .get("fmeasure", 0),
                }
                results.append(result)

                print(
                    f"   ✅ MRR: {result['mrr']:.3f}, Chunks: {result['total_chunks']}"
                )
            else:
                print("   ⚠️ RAG metrics not found in summary.")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback

            traceback.print_exc()

    if results:
        df_results = pd.DataFrame(results).sort_values("mrr", ascending=False)
        print("\n🎯 Chunking Comparison Results:")
        print("=" * 80)
        print(
            df_results[
                [
                    "configuration",
                    "target_tokens",
                    "overlap_tokens",
                    "total_chunks",
                    "recall_at_1",
                    "recall_at_5",
                    "mrr",
                    "rouge1_f",
                ]
            ].round(4)
        )

        best_config = df_results.iloc[0]
        print(f"\n🏆 Best configuration: {best_config['configuration']}")
        print(f"📈 Best MRR: {best_config['mrr']:.4f}")

        return df_results
    else:
        print("❌ No valid results generated")
        return pd.DataFrame()


if __name__ == "__main__":
    # Run with 50 questions as requested
    results = compare_chunking_configs(
        num_questions=100,
        configs=[
            {"target_tokens": 750, "overlap_tokens": 150, "name": "XLarge_750_150"}
        ],
    )
    if results is not None and not results.empty:
        print(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results.to_csv(
            RESULTS_DIR / f"chunking_comparison_results_{timestamp}.csv", index=False
        )
