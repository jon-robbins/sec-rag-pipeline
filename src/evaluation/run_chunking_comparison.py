import os
from pathlib import Path

from chunking_comparison import compare_chunking_configs

configs = [
    {"target_tokens": 150, "overlap_tokens": 25, "name": "Small_150_25"},
    {"target_tokens": 300, "overlap_tokens": 50, "name": "Medium_300_50"},
    {"target_tokens": 500, "overlap_tokens": 100, "name": "Large_500_100"},
    {"target_tokens": 750, "overlap_tokens": 150, "name": "XLarge_750_150"},
]

data_path = Path(os.getcwd()) / "data"

# df_results = pd.read_csv(data_path / 'chunking_comparison_results.csv')
# df_results

if __name__ == "__main__":
    df_results = compare_chunking_configs(num_questions=50, configs=configs)
    df_results.to_csv(data_path / "results" / "chunking_comparison_results.csv")
