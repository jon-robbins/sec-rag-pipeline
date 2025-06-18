import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tiktoken

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


def count_tokens(row):
    """
    Count the number of tokens in each text.
    """
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    return len(enc.encode(str(row)))

#plot tight graph of distribution of num_tokens by ticker and year
def plot_token_distribution(df, log_scale=False, title=None, ylim=None):
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
        sns.violinplot(data=ticker_data, x='fiscal_year', y='num_tokens', ax=axes[idx])
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
