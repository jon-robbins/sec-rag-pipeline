"""
Handles aggregation and reporting of evaluation results.
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List

import numpy as np
import pandas as pd

from ..rag.config import PRICING_PER_CALL, PRICING_PER_TOKEN, RESULTS_DIR

logger = logging.getLogger(__name__)


class EvaluationReporter:
    """
    Aggregates, prints, and saves evaluation results.
    """

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.results_dir = RESULTS_DIR

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregates results from all evaluation runs."""
        aggregated: DefaultDict[str, DefaultDict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )
        scenario_token_metrics: DefaultDict[str, DefaultDict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )

        for item_result in results:
            for scenario, metrics in item_result.items():
                if scenario in [
                    "question",
                    "ground_truth_answer",
                    "ground_truth_chunk_id",
                    "ticker",
                    "year",
                    "section",
                    "chunk_text",
                ]:
                    continue
                if "rouge" in metrics:
                    for rouge_type, scores in metrics["rouge"].items():
                        if isinstance(scores, dict) and "fmeasure" in scores:
                            aggregated[scenario][f"{rouge_type}_f"].append(
                                scores["fmeasure"]
                            )
                        elif isinstance(scores, dict) and "f" in scores:
                            aggregated[scenario][f"{rouge_type}_f"].append(scores["f"])
                        else:
                            aggregated[scenario][f"{rouge_type}_f"].append(scores)
                if "bleu" in metrics:
                    for bleu_type, score in metrics["bleu"].items():
                        aggregated[scenario][bleu_type].append(score)
                if "retrieval" in metrics and metrics["retrieval"]:
                    for metric, value in metrics["retrieval"].items():
                        aggregated[scenario][metric].append(value)
                if "tokens" in metrics and metrics["tokens"]:
                    for token_type, value in metrics["tokens"].items():
                        scenario_token_metrics[scenario][token_type].append(value)

        num_questions = len(results)
        final_summary = {}
        for scenario, metrics in aggregated.items():
            rouge_metrics = {m: np.mean(v) for m, v in metrics.items() if "_f" in m}
            bleu_metrics = {
                m: np.mean(v) for m, v in metrics.items() if m.startswith("bleu")
            }
            retrieval_metrics = {
                m: np.mean(v)
                for m, v in metrics.items()
                if m not in rouge_metrics and m not in bleu_metrics
            }
            scenario_tokens: Dict[str, Any] = scenario_token_metrics.get(scenario, {})
            token_summary = {t: np.mean(v) for t, v in scenario_tokens.items()}

            final_summary[scenario] = {
                "rouge": rouge_metrics,
                "bleu": bleu_metrics,
                "retrieval": retrieval_metrics,
                "tokens": token_summary,
                "total_cost": self._calculate_cost(
                    scenario_tokens, scenario, num_questions
                ),
            }

        return {
            "summary": final_summary,
            "individual": results,
            "per_question_metrics": self._extract_per_question_metrics(results),
        }

    def print_results(self, results: Dict[str, Any]):
        """Prints a summary of the evaluation results to the console."""
        summary = results.get("summary", {})
        logging.info("\n" + "=" * 80)
        logging.info("Evaluation Summary")
        logging.info("=" * 80)

        for scenario, metrics in summary.items():
            logging.info("\n--- Scenario: %s ---", scenario.upper())

            if "retrieval" in metrics and metrics["retrieval"]:
                logging.info("  Retrieval Metrics:")
                for key, val in metrics["retrieval"].items():
                    logging.info("    - %s: %.4f", key, val)

            if "rouge" in metrics and metrics["rouge"]:
                logging.info("  Generation Metrics (ROUGE-F):")
                for key, val in metrics["rouge"].items():
                    logging.info("    - %s: %.4f", key, val)

            if "bleu" in metrics and metrics["bleu"]:
                logging.info("  Generation Metrics (BLEU):")
                for key, val in metrics["bleu"].items():
                    logging.info("    - %s: %.4f", key, val)

            if "tokens" in metrics and metrics["tokens"]:
                logging.info("  Token Usage (avg per question):")
                for key, val in metrics["tokens"].items():
                    logging.info("    - %s: %.2f", key, val)

            if "total_cost" in metrics:
                logging.info("  Estimated Cost: $%.4f", metrics["total_cost"])

        logging.info("\n" + "=" * 80)

    def save_results(self, results: Dict[str, Any]):
        """Saves the aggregated results to JSON and CSV."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        final_results_filename = (
            self.results_dir / f"evaluation_results_{timestamp}.json"
        )
        final_csv_filename = self.results_dir / f"evaluation_results_{timestamp}.csv"

        logging.info("Saving final aggregated results to %s...", final_results_filename)
        with open(final_results_filename, "w") as f:
            json.dump(results, f, indent=4)

        logging.info("Exporting results to CSV: %s...", final_csv_filename)
        self.export_to_csv(results, final_csv_filename)

    def export_to_csv(self, results: Dict[str, Any], filename: Path):
        """Converts the results dictionary to a flat DataFrame and saves as CSV."""
        flat_results = []
        for res in results["individual"]:
            for method, metrics in res.items():
                if method in [
                    "question",
                    "ground_truth_answer",
                    "ground_truth_chunk_id",
                    "ticker",
                    "year",
                    "section",
                    "chunk_text",
                ]:
                    continue
                row = {
                    "method": method,
                    "question": res["question"],
                    "ground_truth_answer": res["ground_truth_answer"],
                    "ground_truth_chunk_id": res.get("ground_truth_chunk_id"),
                    "ticker": res.get("ticker"),
                    "year": res.get("year"),
                    "section": res.get("section"),
                    "generated_answer": metrics.get("answer"),
                }
                row.update(metrics.get("retrieval", {}))
                for rouge_type, scores in metrics.get("rouge", {}).items():
                    if isinstance(scores, dict) and "fmeasure" in scores:
                        row[f"{rouge_type}_f"] = scores["fmeasure"]
                for bleu_type, score in metrics.get("bleu", {}).items():
                    row[bleu_type] = score
                flat_results.append(row)

        df = pd.DataFrame(flat_results)
        df.to_csv(filename, index=False)

    def results_to_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """Converts the final results dictionary to a pandas DataFrame."""
        summary = results.get("summary", {})
        if not summary:
            return pd.DataFrame()

        records = []
        for scenario, metrics in summary.items():
            record = {"scenario": scenario}
            record.update(metrics.get("rouge", {}))
            record.update(metrics.get("bleu", {}))
            record.update(metrics.get("retrieval", {}))
            tokens = metrics.get("tokens", {})
            record["avg_prompt_tokens"] = tokens.get("prompt_tokens")
            record["avg_completion_tokens"] = tokens.get("completion_tokens")
            record["avg_total_tokens"] = tokens.get("total_tokens")
            record["total_cost"] = metrics.get("total_cost")
            records.append(record)

        return pd.DataFrame(records)

    def _extract_per_question_metrics(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, List[float]]]:
        per_question_metrics: DefaultDict[str, DefaultDict[str, List[float]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        for item_result in results:
            for scenario, metrics in item_result.items():
                if scenario in [
                    "question",
                    "ground_truth_answer",
                    "ground_truth_chunk_id",
                    "ticker",
                    "year",
                    "section",
                    "chunk_text",
                ]:
                    continue
                if "rouge" in metrics and metrics["rouge"]:
                    for rouge_type, scores in metrics["rouge"].items():
                        if isinstance(scores, dict) and "fmeasure" in scores:
                            per_question_metrics[scenario][f"{rouge_type}_f"].append(
                                scores["fmeasure"]
                            )
                if "bleu" in metrics and metrics["bleu"]:
                    for bleu_type, score in metrics["bleu"].items():
                        per_question_metrics[scenario][bleu_type].append(score)
                if "retrieval" in metrics and metrics["retrieval"]:
                    for metric, value in metrics["retrieval"].items():
                        per_question_metrics[scenario][metric].append(value)
        return {
            scenario: dict(metrics)
            for scenario, metrics in per_question_metrics.items()
        }

    def _calculate_cost(
        self, token_metrics: Dict[str, List[int]], scenario: str, num_questions: int
    ) -> float:
        total_cost = 0.0
        gpt4_mini_input_price = PRICING_PER_TOKEN["gpt-4o-mini"]["input"]
        gpt4_mini_output_price = PRICING_PER_TOKEN["gpt-4o-mini"]["output"]
        embedding_price = PRICING_PER_TOKEN["text-embedding-3-small"]["input"]
        web_search_price = PRICING_PER_CALL["gpt-4o-mini-search-preview"]

        if "prompt_tokens" in token_metrics and token_metrics["prompt_tokens"]:
            total_prompt_tokens: int = int(np.sum(token_metrics["prompt_tokens"]))
            total_cost += total_prompt_tokens * gpt4_mini_input_price
        if "completion_tokens" in token_metrics and token_metrics["completion_tokens"]:
            total_completion_tokens: int = int(
                np.sum(token_metrics["completion_tokens"])
            )
            total_cost += total_completion_tokens * gpt4_mini_output_price
        if (
            "retrieval_query_tokens" in token_metrics
            and token_metrics["retrieval_query_tokens"]
        ):
            total_retrieval_tokens: int = int(
                np.sum(token_metrics["retrieval_query_tokens"])
            )
            total_cost += total_retrieval_tokens * embedding_price
        if scenario == "web_search":
            total_cost += num_questions * web_search_price
        return total_cost
