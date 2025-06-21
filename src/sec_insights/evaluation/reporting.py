"""
Handles aggregation and reporting of evaluation results.
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List

import numpy as np
import pandas as pd

from ..rag.config import RESULTS_DIR


class EvaluationReporter:
    """
    Aggregates, prints, and saves evaluation results.
    """

    def __init__(self, run_id: str, quiet: bool = False):
        self.run_id = run_id
        self.quiet = quiet
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
            retrieval_metrics = {
                m: np.mean(v) for m, v in metrics.items() if m not in rouge_metrics
            }
            scenario_tokens: Dict[str, Any] = scenario_token_metrics.get(scenario, {})
            token_summary = {t: np.mean(v) for t, v in scenario_tokens.items()}

            final_summary[scenario] = {
                "rouge": rouge_metrics,
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
        if self.quiet:
            return

        summary = results.get("summary", {})
        print("\n" + "=" * 80)
        print("📊 Evaluation Summary")
        print("=" * 80)

        for scenario, metrics in summary.items():
            print(f"\n--- Scenario: {scenario.upper()} ---")

            if "retrieval" in metrics and metrics["retrieval"]:
                print("  Retrieval Metrics:")
                for key, val in metrics["retrieval"].items():
                    print(f"    - {key}: {val:.4f}")

            if "rouge" in metrics and metrics["rouge"]:
                print("  Generation Metrics (ROUGE-F):")
                for key, val in metrics["rouge"].items():
                    print(f"    - {key}: {val:.4f}")

            if "tokens" in metrics and metrics["tokens"]:
                print("  Token Usage (avg per question):")
                for key, val in metrics["tokens"].items():
                    print(f"    - {key}: {val:.2f}")

            if "total_cost" in metrics:
                print(f"  Estimated Cost: ${metrics['total_cost']:.4f}")

        print("\n" + "=" * 80)

    def save_results(self, results: Dict[str, Any]):
        """Saves the aggregated results to JSON and CSV."""
        if self.quiet:
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        final_results_filename = (
            self.results_dir / f"evaluation_results_{timestamp}.json"
        )
        final_csv_filename = self.results_dir / f"evaluation_results_{timestamp}.csv"

        print(f"💾 Saving final aggregated results to {final_results_filename}...")
        with open(final_results_filename, "w") as f:
            json.dump(results, f, indent=4)

        print(f"📊 Exporting results to CSV: {final_csv_filename}...")
        self.export_to_csv(results, final_csv_filename)

    def export_to_csv(self, results: Dict[str, Any], filename: Path):
        """Converts the results dictionary to a flat DataFrame and saves as CSV."""
        flat_results = []
        for res in results["individual"]:
            for method, metrics in res.items():
                if method in [
                    "question",
                    "ground_truth_answer",
                    "section",
                    "chunk_text",
                ]:
                    continue
                row = {
                    "method": method,
                    "question": res["question"],
                    "ground_truth_answer": res["ground_truth_answer"],
                }
                row.update(metrics.get("retrieval", {}))
                for rouge_type, scores in metrics.get("rouge", {}).items():
                    if isinstance(scores, dict) and "fmeasure" in scores:
                        row[f"{rouge_type}_f"] = scores["fmeasure"]
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
        gpt4_mini_input_price = 0.15 / 1_000_000
        gpt4_mini_output_price = 0.60 / 1_000_000
        embedding_price = 0.02 / 1_000_000
        web_search_price = 27.5 / 1_000

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
