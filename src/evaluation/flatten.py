from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# The aggregator lives under notebooks/, add to path at import time
_NOTEBOOKS_DIR = Path(__file__).parent.parent.parent / "notebooks"
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.append(str(_NOTEBOOKS_DIR))

# Import from notebooks after adding to path
from notebooks.eval_utils import get_cost  # cost calculation helper  # noqa: E402
from notebooks.results_aggregator import ResultsAggregator  # type: ignore  # noqa: E402

"""Utilities to convert the nested `individual_results.json` produced by
ComprehensiveEvaluator into the same flattened, per-question CSV structure
used by the legacy *consolidated_results_fixed_tokens.csv*.

This simply re-uses the existing `ResultsAggregator` logic so we keep one
source of truth for column names and cost calculation.
"""


def flatten_evaluator_output(
    run_dir: Path, template_csv: Path | None = None
) -> pd.DataFrame:  # noqa: D401
    """Return a DataFrame matching the legacy consolidated CSV format.

    Parameters
    ----------
    run_dir : Path
        Directory that contains `individual_results.json` (typically the
        `temp_dir` returned by `ComprehensiveEvaluator`).
    template_csv : Path, optional
        Path to an existing consolidated CSV whose header should be treated
        as the canonical column order. If provided, the returned DataFrame is
        re-indexed to match this order (missing columns will be filled with
        NA).
    """
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)

    aggregator = ResultsAggregator(str(run_dir))
    df = aggregator.get_individual_results_df(include_rrf=False)

    if df.empty:
        raise ValueError("No data found to flatten in " + str(run_dir))

    # Ensure legacy naming conventions
    if "method" in df.columns and "system" not in df.columns:
        df.rename(columns={"method": "system"}, inplace=True)

    # Add cost (old pipelines relied on this specific column name)
    if "cost" not in df.columns:
        df["cost"] = df.apply(get_cost, axis=1)

    # Accuracy flag used downstream
    if "rougeL" in df.columns and "is_accurate" not in df.columns:
        df["is_accurate"] = df["rougeL"] >= 0.35

    # Re-index to the template if supplied
    if template_csv and template_csv.exists():
        header = template_csv.read_text().splitlines()[0].split(",")
        for col in header:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[header]

    return df
