import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tiktoken
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any
import warnings
from sklearn.metrics.pairwise import cosine_similarity

# Import bootstrap analyzer for type hints and functionality
import sys
sys.path.append(str(Path(__file__).parent.parent))
from evaluation.bootstrap_analysis import BootstrapAnalyzer

def plot_filing_coverage(df):
    """
    Plot a heatmap showing filing coverage by ticker and fiscal year.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'ticker' and 'fiscal_year' columns
    """

    
    # Create a binary indicator for each (ticker, year)
    pivot = (
        df[['ticker', 'fiscal_year']]
        .drop_duplicates()
        .assign(has_filing=1)
        .pivot(index='ticker', columns='fiscal_year', values='has_filing')
        .fillna(0)
        .astype(int)
        .sort_index(axis=1)  # sort years left to right
    )

    plt.figure(figsize=(12, 6))
    sns.heatmap(pivot, cmap="Blues", cbar=False, linewidths=0.5, linecolor="gray")
    plt.title("Filing Coverage by Ticker and Fiscal Year")
    plt.xlabel("Fiscal Year")
    plt.ylabel("Ticker")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)    
    plt.tight_layout()
    plt.show()


# ============================================================================
# Bootstrap Confidence Interval Visualization Functions
# ============================================================================

def plot_confidence_intervals(
    bootstrap_results: pd.DataFrame, 
    metric: str, 
    save_path: Optional[str] = None,
    figsize: tuple = (12, 8),
    alpha: float = 0.05
) -> None:
    """
    Plot confidence intervals for method comparisons on a specific metric.
    
    Args:
        bootstrap_results: DataFrame from BootstrapAnalyzer.bootstrap_all_pairs()
        metric: Name of the metric to plot
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height)
        alpha: Significance level for highlighting
    """
    # Filter for the specific metric
    metric_data = bootstrap_results[bootstrap_results['metric'] == metric].copy()
    
    if metric_data.empty:
        raise ValueError(f"No data found for metric '{metric}'")
    
    # Sort by observed difference for better visualization
    metric_data = metric_data.sort_values('observed_difference')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot confidence intervals
    for i, (_, row) in enumerate(metric_data.iterrows()):
        color = 'red' if row['is_significant'] else 'blue'
        alpha_val = 0.8 if row['is_significant'] else 0.6
        
        # Plot confidence interval
        ax.errorbar(
            x=row['observed_difference'], 
            y=i,
            xerr=[[row['observed_difference'] - row['ci_lower']], 
                  [row['ci_upper'] - row['observed_difference']]],
            fmt='o',
            color=color,
            alpha=alpha_val,
            capsize=5,
            label='Significant' if i == 0 and row['is_significant'] else ('Non-significant' if i == 0 else None)
        )
    
    # Add vertical line at zero
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='No difference')
    
    # Customize plot
    ax.set_xlabel(f'Difference in {metric.replace("_", " ").title()}')
    ax.set_ylabel('Method Comparisons')
    ax.set_yticks(range(len(metric_data)))
    ax.set_yticklabels(metric_data['comparison'])
    ax.set_title(f'{(1-alpha)*100:.0f}% Confidence Intervals for {metric.replace("_", " ").title()} Differences')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_difference_distributions(
    analyzer: BootstrapAnalyzer,
    method1: str,
    method2: str, 
    metrics: List[str],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 10)
) -> None:
    """
    Plot bootstrap difference distributions for multiple metrics.
    
    Args:
        analyzer: Initialized BootstrapAnalyzer instance
        method1: Name of the first method
        method2: Name of the second method
        metrics: List of metrics to plot
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height)
    """
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        try:
            # Get bootstrap results
            result = analyzer.bootstrap_difference(method1, method2, metric)
            
            # Plot histogram of bootstrap differences
            axes[i].hist(
                result['bootstrap_differences'], 
                bins=50, 
                alpha=0.7, 
                color='skyblue',
                edgecolor='black',
                density=True
            )
            
            # Add vertical lines for observed difference and CI
            axes[i].axvline(
                result['observed_difference'], 
                color='red', 
                linestyle='-', 
                linewidth=2,
                label='Observed Difference'
            )
            axes[i].axvline(
                result['ci_lower'], 
                color='orange', 
                linestyle='--',
                label=f'{result["confidence_level"]*100:.0f}% CI'
            )
            axes[i].axvline(
                result['ci_upper'], 
                color='orange', 
                linestyle='--'
            )
            axes[i].axvline(0, color='black', linestyle=':', alpha=0.5, label='No difference')
            
            # Customize subplot
            axes[i].set_xlabel(f'Difference in {metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Density')
            axes[i].set_title(f'{metric.replace("_", " ").title()}\n({method1} - {method2})')
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
            
            # Add significance indicator
            if result['is_significant']:
                axes[i].text(
                    0.02, 0.98, 
                    'Significant', 
                    transform=axes[i].transAxes,
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
                    fontsize=10,
                    verticalalignment='top'
                )
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Error: {str(e)}', 
                        transform=axes[i].transAxes, ha='center', va='center')
            axes[i].set_title(f'{metric} (Error)')
    
    # Remove empty subplots
    for i in range(len(metrics), len(axes)):
        if i < len(axes):
            fig.delaxes(axes[i])
    
    plt.suptitle(f'Bootstrap Difference Distributions: {method1} vs {method2}', 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_pairwise_heatmap(
    bootstrap_results: pd.DataFrame,
    metric: str, 
    confidence_level: float = 0.95,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8)
) -> None:
    """
    Plot a heatmap showing pairwise performance differences.
    
    Args:
        bootstrap_results: DataFrame from BootstrapAnalyzer.bootstrap_all_pairs()
        metric: Name of the metric to plot
        confidence_level: Confidence level for significance
        save_path: Optional path to save the figure
        figsize: Figure size as (width, height)
    """
    # Filter for the specific metric
    metric_data = bootstrap_results[bootstrap_results['metric'] == metric].copy()
    
    if metric_data.empty:
        raise ValueError(f"No data found for metric '{metric}'")
    
    # Get all unique methods
    methods = list(set(metric_data['method1'].tolist() + metric_data['method2'].tolist()))
    methods.sort()
    
    # Create matrices for differences and significance
    diff_matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)
    sig_matrix = pd.DataFrame(index=methods, columns=methods, dtype=bool)
    
    # Fill matrices
    for _, row in metric_data.iterrows():
        m1, m2 = row['method1'], row['method2']
        diff = row['observed_difference']
        is_sig = row['is_significant']
        
        # Fill both directions
        diff_matrix.loc[m1, m2] = diff
        diff_matrix.loc[m2, m1] = -diff
        
        sig_matrix.loc[m1, m2] = is_sig
        sig_matrix.loc[m2, m1] = is_sig
    
    # Fill diagonal with zeros
    for method in methods:
        diff_matrix.loc[method, method] = 0
        sig_matrix.loc[method, method] = False
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        diff_matrix.astype(float),
        annot=True,
        fmt='.3f',
        cmap='RdBu_r',
        center=0,
        square=True,
        cbar_kws={'label': f'Difference in {metric.replace("_", " ").title()}'},
        ax=ax
    )
    
    # Add significance indicators
    for i, method1 in enumerate(methods):
        for j, method2 in enumerate(methods):
            if sig_matrix.loc[method1, method2]:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, 
                                         edgecolor='black', linewidth=3))
    
    # Customize plot
    ax.set_title(f'Pairwise Differences in {metric.replace("_", " ").title()}\n'
                f'(Black borders indicate significance at {confidence_level*100:.0f}% level)')
    ax.set_xlabel('Method 2')
    ax.set_ylabel('Method 1')
    
    # Add interpretation note
    fig.text(0.02, 0.02, 
             'Interpretation: Positive values indicate Method 1 > Method 2', 
             fontsize=10, style='italic')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_comprehensive_bootstrap_report(
    results_path: str,
    output_dir: Optional[str] = None,
    methods: Optional[List[str]] = None,
    metrics: Optional[List[str]] = None,
    n_bootstrap: int = 1000
) -> None:
    """
    Create a comprehensive bootstrap analysis report with multiple visualizations.
    
    Args:
        results_path: Path to the evaluation results JSON file
        output_dir: Directory to save plots (if None, plots are shown but not saved)
        methods: List of methods to analyze (default: all available)
        metrics: List of metrics to analyze (default: all available)
        n_bootstrap: Number of bootstrap samples
    """
    print("🔄 Loading results and initializing bootstrap analyzer...")
    
    # Initialize analyzer
    analyzer = BootstrapAnalyzer(results_path, n_bootstrap=n_bootstrap)
    
    # Get available methods and metrics
    available_methods = analyzer.get_available_methods()
    available_metrics = analyzer.get_available_metrics()
    
    if methods is None:
        methods = available_methods
    if metrics is None:
        metrics = available_metrics
        
    print(f"📊 Analyzing {len(methods)} methods across {len(metrics)} metrics...")
    print(f"Methods: {', '.join(methods)}")
    print(f"Metrics: {', '.join(metrics)}")
    
    # Create output directory if specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate bootstrap results for all pairs
    print("🎲 Computing bootstrap confidence intervals...")
    bootstrap_df = analyzer.bootstrap_all_pairs(methods=methods, metrics=metrics)
    
    # Create visualizations for each metric
    for metric in metrics:
        print(f"📈 Generating plots for {metric}...")
        
        # 1. Confidence intervals plot
        save_path_ci = str(output_path / f"{metric}_confidence_intervals.png") if output_dir else None
        try:
            plot_confidence_intervals(bootstrap_df, metric, save_path=save_path_ci)
        except Exception as e:
            warnings.warn(f"Failed to create confidence intervals plot for {metric}: {e}")
        
        # 2. Pairwise heatmap
        save_path_hm = str(output_path / f"{metric}_pairwise_heatmap.png") if output_dir else None
        try:
            plot_pairwise_heatmap(bootstrap_df, metric, save_path=save_path_hm)
        except Exception as e:
            warnings.warn(f"Failed to create heatmap for {metric}: {e}")
    
    # 3. Distribution plots for selected method pairs
    if len(methods) >= 2:
        print("📊 Generating difference distribution plots...")
        for i in range(len(methods)):
            for j in range(i+1, len(methods)):
                method1, method2 = methods[i], methods[j]
                save_path_dist = str(output_path / f"{method1}_vs_{method2}_distributions.png") if output_dir else None
                try:
                    plot_difference_distributions(
                        analyzer, method1, method2, metrics, save_path=save_path_dist
                    )
                except Exception as e:
                    warnings.warn(f"Failed to create distribution plot for {method1} vs {method2}: {e}")
    
    # 4. Generate summary statistics
    print("📋 Generating summary statistics...")
    summary_stats = analyzer.summary_statistics()
    print("\nSummary Statistics:")
    print(summary_stats.to_string(index=False))
    
    # 5. Check for problematic metrics
    issues = analyzer.detect_problematic_metrics()
    if issues:
        print("\n⚠️ Detected potential issues:")
        for issue_type, problem_metrics in issues.items():
            print(f"  {issue_type}: {', '.join(problem_metrics)}")
    
    # 6. Save bootstrap results to CSV
    if output_dir:
        csv_path = output_path / "bootstrap_results.csv"
        bootstrap_df.to_csv(csv_path, index=False)
        print(f"💾 Bootstrap results saved to {csv_path}")
        
        # Save summary statistics
        summary_path = output_path / "summary_statistics.csv"
        summary_stats.to_csv(summary_path, index=False)
        print(f"💾 Summary statistics saved to {summary_path}")
    
    print("✅ Bootstrap analysis report complete!")
    
    return bootstrap_df, summary_stats


def count_tokens(row):
    """
    Count the number of tokens in each text.
    """
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(enc.encode(str(row)))

#plot tight graph of distribution of num_tokens by ticker and year
def plot_token_distribution(df, token_col='num_tokens', log_scale=False, title=None, ylim=None):
    """
    Plot the distribution of token counts by ticker and fiscal year.
    Each company gets its own subplot showing token distribution across years.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'ticker', 'fiscal_year', and 'num_tokens' columns
    """
    # Get unique tickers
    tickers = df['ticker'].unique()
    n_tickers = len(tickers)
    
    # Calculate grid dimensions
    n_cols = min(3, n_tickers)
    n_rows = (n_tickers + n_cols - 1) // n_cols
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    # Plot distribution for each ticker
    for idx, ticker in enumerate(tickers):
        ticker_data = df[df['ticker'] == ticker]
        
        # Create violin plot
        sns.violinplot(data=ticker_data, x='fiscal_year', y=token_col, ax=axes[idx])
        if ylim:
            axes[idx].set_ylim(ylim)
        # Customize subplot
        axes[idx].set_yscale('log' if log_scale else 'linear')
        axes[idx].set_ylabel(f'Number of Tokens{" (log scale)" if log_scale else ""}')
        
        # Set title
        default_title = f'Token Distribution by Company and Year{" (log scale)" if log_scale else ""}'
        plt.suptitle(title if title else default_title, y=1.02)
        
        axes[idx].set_title(f'{ticker}')
        axes[idx].set_xlabel('Fiscal Year')
        axes[idx].tick_params(axis='x', rotation=45)
    
    # Remove empty subplots
    for idx in range(len(tickers), len(axes)):
        fig.delaxes(axes[idx])
    
    
    plt.tight_layout()
    plt.show()

def plot_top_sections_by_tokens(df, top_n=10, figsize=(16, 10)):
    """
    Plot the top N sections by total token count.
    
    Args:
        df: DataFrame with 'section_num', 'section_letter', and 'sentence_token_count' columns
        top_n: Number of top sections to display (default: 10)
        figsize: Figure size tuple (default: (16, 10))
    """
    from rag.config import SEC_10K_SECTIONS
    
    # Sum tokens by section
    section_tokens = df.groupby(['section_num', 'section_letter']).agg({
        'sentence_token_count': 'sum'
    }).reset_index()
    
    # Create section identifier
    section_tokens['section_id'] = section_tokens['section_num'].astype(str) + section_tokens['section_letter']
    
    # Map to section descriptions
    section_tokens['section_description'] = section_tokens['section_id'].map(SEC_10K_SECTIONS)
    section_tokens['section_and_description'] = section_tokens['section_id'] + ' - ' + section_tokens['section_description']
    
    # Get top N by token count
    section_tokens = section_tokens.sort_values('sentence_token_count', ascending=True).head(top_n)
    
    # Create bar chart
    plt.figure(figsize=figsize)
    bars = plt.barh(section_tokens['section_and_description'], section_tokens['sentence_token_count'])
    
    plt.title(f'Top {top_n} Sections by Total Tokens')
    plt.xlabel('Total Tokens')
    plt.ylabel('Section')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                 f'{int(width):,}', ha='left', va='center')
    
    # Format y-axis labels
    plt.gca().set_yticklabels(section_tokens['section_and_description'], wrap=True)
    plt.gca().yaxis.set_tick_params(pad=20)
    
    plt.tight_layout()
    plt.show()


def plot_embedding_similarity(df_embeddings, dimension='company', figsize=(10, 8)):
    """
    Calculate and plot cosine similarity between embeddings aggregated by a given dimension.
    
    Args:
        df_embeddings: DataFrame with 'embedding' column and the specified dimension column
        dimension: Column name to aggregate by (e.g., 'company', 'year', 'section_desc')
        figsize: Tuple for figure size
    """
    
    # Calculate average embeddings for each group
    group_embeddings = {}
    for group in df_embeddings[dimension].unique():
        group_data = df_embeddings[df_embeddings[dimension] == group]
        avg_embedding = np.mean(group_data['embedding'].tolist(), axis=0)
        group_embeddings[group] = avg_embedding

    # Convert to matrix format
    groups = sorted(group_embeddings.keys())
    embedding_matrix = np.array([group_embeddings[group] for group in groups])

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(embedding_matrix)

    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        similarity_matrix, 
        xticklabels=groups, 
        yticklabels=groups,
        annot=True, 
        cmap='viridis', 
        vmin=0, 
        vmax=1,
        square=True
    )
    plt.title(f'Cosine Similarity Between {dimension.title()} Embeddings')
    plt.xlabel(dimension.title())
    plt.ylabel(dimension.title())
    plt.tight_layout()
    plt.show()