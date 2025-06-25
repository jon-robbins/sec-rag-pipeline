from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from src.evaluation.evaluator import ComprehensiveEvaluator
from src.evaluation.flatten import flatten_evaluator_output

"""evaluation.runner
Convenience wrapper that executes `ComprehensiveEvaluator`, flattens the
results to the legacy CSV format, and returns the path to the CSV.
"""


def run_full_eval(
    *,
    pipeline,
    num_questions: int = 300,
    methods: Optional[List[str]] = None,
    k_values: Optional[List[int]] = None,
    output_dir: Path | None = None,
    root_dir: Path | None = None,
):
    """Run evaluator, flatten output, save consolidated CSV.

    Returns the path to the consolidated CSV.
    """
    root_dir = root_dir or Path.cwd().parent

    # Ensure output_dir is a directory even if user passes a file path
    if output_dir is None:
        output_dir = Path("data/results/final")
    if output_dir.suffix:
        # Has a file extension -> use its parent as directory
        output_dir = output_dir.parent

    output_dir.mkdir(parents=True, exist_ok=True)

    evaluator = ComprehensiveEvaluator(root_dir, pipeline)
    _, tmp_dir = evaluator.evaluate_all_scenarios(
        num_questions=num_questions,
        methods=methods,
        k_values=k_values,
    )

    df_flat = flatten_evaluator_output(tmp_dir)
    csv_path = output_dir / "consolidated_results_fixed_tokens.csv"
    df_flat.to_csv(csv_path, index=False)
    return df_flat
