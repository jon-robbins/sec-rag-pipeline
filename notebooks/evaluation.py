# %% [markdown]
# # Evaluation
#  
# Now that we have our pipelines developed, embeddings done, chunks optimized and created, we can evaluate its performance. You can do it with the code below, but I'll directly load the results. 
# 
# Please note that due to my own sloppiness, I ran several different evaluations and saved them to different csv's. I aggregated the results together into a csv. 

# %%
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline_factory import build_and_evaluate
from src.preprocessing.chunkers import ChunkingConfig

# -----------------------------------------------------
# End-to-end pipeline + evaluation in one call
# -----------------------------------------------------

root_dir = Path.cwd().parent

df = build_and_evaluate(
    raw_df_path   = root_dir / "data" / "raw" / "df_filings_full.parquet",
    chunk_cfg     = ChunkingConfig(target_tokens=150, overlap_tokens=50, hard_ceiling=500),
    num_questions = 300,
    methods       = ["rag", "reranked_rag", "ensemble_rerank_rag", "unfiltered", "baseline"],
    k_values      = [1, 3, 5, 7, 10],
    output_dir    = root_dir / "data" / "results" / "final" / "final_results_0624.csv",
    use_docker    = True,
)

# Quick sanity preview
df.head()

# %% [markdown]
# Let's look at the distributions of Bleu-4 and Rouge-L

# %%
from eval_utils import plot_distributions
plot_distributions(df)

# %% [markdown]
# Our ROUGE-L scores are quite good, especially in comparison to other finance QA systems. [Chen et al 2024](https://aclanthology.org/2024.acl-long.328.pdf) reported RougeL scores in the 20s, but their corpus was much bigger than ours. 
# 
# BLEU-4 has a consistently low score, especially for the web and baseline systems.  
# However, this isn't necessarily a red flag in the finance domain. In financial QA,  
# we're more interested in whether the answer contains the correct facts (e.g., entities, figures, dates)  
# than whether it uses the exact phrasing of the ground truth.
# 
# For example, consider this ground truth:
# 
# > Apple has a market cap of $2.2 trillion.
# 
# 
# **Answer 1:** Tesla has a market cap of $2.2 billion.  
# **Answer 2:** As of 2024, Apple's $2.2 trillion market cap overshadows all other tech companies.
# 
# Answer 1 might score deceptively high on BLEU-4 because it overlaps lexically with the reference,  
# even though it's factually incorrect (wrong company and amount).
# 
# Answer 2 might score relatively low on BLEU-4 due to phrasing differences,  
# but high on ROUGE-L because it preserves key factual content in a paraphrased form.
# 
# In domains like legal or customer service QA, where exact wording can be critical,  
# a low BLEU-4 would be more concerning.
# 
# In our case, we treat ROUGE-L as a more reliable indicator of factual correctness,  
# and interpret BLEU-4 as a secondary signal for surface-level fluency.
# 
# Based on a cursory overview of the responses, the generated answers were typically much more verbose than the ground truth, which could explain the lower BLEU scores but the decent ROUGE-L scores. 
# 
# The distribution of ROUGE-L of the more accurate systems seems to center at 0.35. Let's check ROUGE-L scores about 0.35 to make sure that they're generally correct. 

# %%
from eval_utils import show_sample_questions
show_sample_questions(df, n=5)

# %% [markdown]
# I checked around 100 questions with a ROUGE-L score between 0.35-0.4, and I'd say about 95% of the answers are factually accurate. 
# 
# There are other methods that we could use to gauge whether or not it's factually accurate. LLM's as a judge is one option, or using BERTScore. However most papers evaluating Q&A tasks tend to focus on recall heavy metrics. Recall is more important for our use case because a false negative cost is much higher than a false positive cost. Regulators care much more about ommitted facts than extra information. 
# 
# For the purposes of this exercise, we'll count any question with a ROUGE-L score of >=0.35 as correct. 

# %%
df['is_accurate'] = df['rougeL'] >= 0.35

# %% [markdown]
# Now, which systems are most accurate?

# %%
from eval_utils import plot_accuracy_bars
plot_accuracy_bars(df)

# %% [markdown]
# The reranked ensemble method performs almost as well as uploading the corresponding SEC file. 
# 
# Now let's evaluate the cost per accurate question

# %%
from eval_utils import calculate_cost_analysis
# Calculate and display cost analysis tables
cost_analysis = calculate_cost_analysis(df)

print("Cost per 1,000 Accurate Answers by System:")
print("=" * 50)
for system in cost_analysis.index:
    cost_per_1k_accurate = cost_analysis.loc[system, 'cost_per_1k_accurate']
    accuracy_rate = cost_analysis.loc[system, 'accuracy_rate']
    print(f"{system:20} ${cost_per_1k_accurate:8.2f} (accuracy: {accuracy_rate:.1%})")

print("\n" + "=" * 50)
print("Cost per 1,000 Queries by System:")
print("=" * 50)
for system in cost_analysis.index:
    avg_cost_per_question = cost_analysis.loc[system, 'avg_cost_per_question']
    cost_per_1k_queries = avg_cost_per_question * 1000
    print(f"{system:20} ${cost_per_1k_queries:8.2f}")

print("\n" + "=" * 50)


# %%
from eval_utils import calculate_cost_analysis, plot_cost_scatter
cost_analysis = calculate_cost_analysis(df)
plot_cost_scatter(cost_analysis)

# %% [markdown]
# Our primary metric is Cost Per Accurate Answer, because we want to capture the tradeoff between accuracy and cost. In a real world scenario we'd also want to consider latency, but for the purposes of this exercise we'll assume that's not a factor. 
# 
# We can see that the web search is an extreme outlier, costing $300 per correct answer. This is a function of its abysmal performance in getting accurate answers, as well as its per-call cost of $27.5 per 1000 calls, in addition to any token costs. 
# 
# The unfiltered model (uploading all context for a specific SEC filing to the LLM) is the most accurate model by a small margin, but costs about 40x more than the RAG models. 
# 
# One key component of RAG models that make it an attractive architecture from the business standpoint is its potential for optimization. Currently we are sending the top 10 relevant chunks to the user's query to the LLM. But we set the number of chunks relatively arbitrarily. Let's see if there is room for optimization there to save more costs. 

# %%
from eval_utils import plot_recall_performance
plot_recall_performance(df)


# %% [markdown]
# Looks to be flattening out, let's verify by making sure it's an exponential model. 

# %%
from eval_utils import fit_exponential_model, pareto_analysis
model, params = fit_exponential_model(df)

# %%
(df[df['system']
    .isin(['ensemble_rerank_rag', 'rag', 'reranked_rag'])]
    .groupby('system')['ndcg@10']
    .mean()
)

# %% [markdown]
# Recall@k means of the top k chunks returned, how many of them were relevant. The more chunks we send, the higher the probability that the chunks we provide will be relevant. But as with any exponential function, it comes with diminishing returns. As we can see here, the delta between 5 and 10 is just 0.022, or about 3.5%. 
# 
# Recall our hypothetical user base:
# - 1000 users
# - Each online for 10 hours per day
# - Each asks one question every 3 minutes
# 
# That's 200,000 questions per day. 
# 
# Every 1 million tokens we send to GPT are $0.15. Each chunk we send has 1500 context tokens. 
# 
# If we could cut the number of context tokens in half our savings would be:

# %%
users = 1000
active_hours = 10
questions_per_hour = 20

questions_per_year = users * active_hours * questions_per_hour * 365

tokens_per_chunk = 150  
cost_per_million_tokens = 0.15 

current_chunk_number = 10
reduced_chunk_number = 5

current_annual_context_tokens = current_chunk_number * tokens_per_chunk * questions_per_year
reduced_annual_context_tokens = reduced_chunk_number * tokens_per_chunk * questions_per_year

current_annual_context_token_cost = (current_annual_context_tokens / 1_000_000) * cost_per_million_tokens
reduced_annual_context_token_cost = (reduced_annual_context_tokens / 1_000_000) * cost_per_million_tokens

print(f"Questions per year: {questions_per_year:,}")
print(f"Current annual context tokens: {current_annual_context_tokens:,}")
print(f"Current annual operating cost: {df[df['system'] == 'ensemble_rerank_rag']['cost'].mean()*questions_per_year:,.2f}")
print(f"Theoretical annual savings: ${current_annual_context_token_cost - reduced_annual_context_token_cost:,.2f}")


# %% [markdown]
# We could save up to 8k by simply tuning a model parameter with minimal loss to accuracy. 

# %%
pareto_analysis(df, model, params)

# %% [markdown]
# We optimize for both cost and accuracy at k=7. However, this model is just our MVP. We can optimize other parts of the architecture for more savings to build a more scaleable system. For more information see Next Steps in the README.md file. 

# %%
import seaborn as sns, pandas as pd

sns.violinplot(
    data=df[df['system'].isin(['ensemble_rerank_rag', 'rag', 'reranked_rag'])], x="system", y="ndcg@10", inner=None, cut=0, palette="Set3"
)
sns.swarmplot(
    data=df[df['system'].isin(['ensemble_rerank_rag', 'rag', 'reranked_rag'])], x="system", y="ndcg@10", color="k", size=2, alpha=.5
)
sns.despine()

# %% [markdown]
# This is slightly on the lower side, the 2nd place winners at the [ICAIF 2024 FinanceRAG hackathon](https://arxiv.org/pdf/2411.16732) had a ndgc@10 of about 0.63. This could be due to the embeddings we used (the small model rather than large model) or using the small reranker model.

# %%
from src.methods.baseline import run_baseline_scenario
from src.methods.web_search import run_web_search_scenario
from src.methods.unfiltered_text import run_unfiltered_context_scenario
from src.vector_store.vector_store import VectorStore
from src.vector_store.embedding import EmbeddingManager
from src.rag.reranker import BGEReranker
from src.methods.ensemble_rerank_rag import CrossEncoder

# -----------------------
# Consolidate results into default-format CSV under data/results/final/
# -----------------------
from notebooks.results_aggregator import ResultsAggregator

final_results_dir = root_dir / 'data' / 'results' / 'final'
final_results_dir.mkdir(parents=True, exist_ok=True)

# Copy summary.csv into final dir with unique name for traceability
import shutil, datetime

summary_src = temp_dir / 'summary.csv'
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
summary_dest = final_results_dir / f'summary_{timestamp}.csv'

# Copy files
if summary_src.exists():
    shutil.copy(summary_src, summary_dest)

raw_src = temp_dir / 'individual_results.json'
raw_dest = final_results_dir / f'individual_results_{timestamp}.json'
if raw_src.exists():
    shutil.copy(raw_src, raw_dest)

aggregator = ResultsAggregator(str(final_results_dir))
df_individual = aggregator.get_individual_results_df(include_rrf=False)

if not df_individual.empty:
    # Ensure cost column matches old pipeline naming
    from notebooks.eval_utils import get_cost
    df_individual['cost'] = df_individual.apply(get_cost, axis=1)
    # is_accurate flag per old logic
    if 'rougeL' in df_individual.columns:
        df_individual['is_accurate'] = df_individual['rougeL'] >= 0.35
    # Rename method -> system for backward compatibility if needed
    if 'method' in df_individual.columns and 'system' not in df_individual.columns:
        df_individual.rename(columns={'method':'system'}, inplace=True)

    df_individual.to_csv(final_results_dir / 'consolidated_results_fixed_tokens.csv', index=False)


