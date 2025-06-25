"""
Handles aggregation and reporting of evaluation results.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List

import numpy as np
import pandas as pd

from src.utils.config import RESULTS_DIR

logger = logging.getLogger(__name__)


class EvaluationReporter:
    """Aggregates, prints, and saves evaluation results."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.results_dir = RESULTS_DIR / run_id
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_results(self, results: Dict[str, Any]):
        """
        Aggregates results, prints a summary, and saves raw and summary files.
        """
        individual_results = results.get("individual", [])
        if not individual_results:
            logger.warning("No individual results to process.")
            return

        # 1. Save raw individual results
        raw_results_path = self.results_dir / "individual_results.json"
        logger.info("Saving raw individual results to %s", raw_results_path)
        with open(raw_results_path, "w") as f:
            json.dump(individual_results, f, indent=4)

        # 2. Aggregate results
        summary = self._aggregate_results(individual_results)

        # 3. Save summary
        summary_path = self.results_dir / "summary.json"
        logger.info("Saving summary results to %s", summary_path)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        # 4. Print summary to console
        self.print_summary(summary)

        # 5. Export summary to CSV
        csv_path = self.results_dir / "summary.csv"
        logger.info("Exporting summary to CSV: %s", csv_path)
        self.export_summary_to_csv(summary, csv_path)

    def _aggregate_results(
        self, individual_results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregates results from a list of individual question outcomes."""
        # method -> metric -> [values]
        agg: DefaultDict[str, DefaultDict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )

        for res in individual_results:
            for method, data in res.items():
                if not isinstance(data, dict):
                    continue

                for metric, value in data.get("retrieval_metrics", {}).items():
                    agg[method][metric].append(value)

                for token_type, count in data.get("token_usage", {}).items():
                    agg[method][token_type].append(count)

        # Calculate means
        summary: Dict[str, Dict[str, Any]] = defaultdict(dict)
        for method, metrics in agg.items():
            for metric, values in metrics.items():
                summary[method][f"mean_{metric}"] = np.mean(values)

        return dict(summary)

    def print_summary(self, summary: Dict[str, Any]):
        """Prints a summary of the evaluation results to the console."""
        logging.info("\n" + "=" * 80)
        logging.info("Evaluation Summary (Run ID: %s)", self.run_id)
        logging.info("=" * 80)

        for method, metrics in summary.items():
            logging.info("\n--- Method: %s ---", method.upper())
            for key, val in metrics.items():
                logging.info("    - %s: %.4f", key, val)

        logging.info("\n" + "=" * 80)

    def export_summary_to_csv(self, summary: Dict[str, Any], filename: Path):
        """Converts the summary dictionary to a DataFrame and saves as CSV."""
        df = pd.DataFrame.from_dict(summary, orient="index")
        df.index.name = "method"
        df.to_csv(filename)
