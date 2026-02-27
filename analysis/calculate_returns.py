import logging
import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sqlalchemy import text

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in os.sys.path:
    os.sys.path.insert(0, PROJECT_ROOT)

from db.connection import get_engine

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.dirname(__file__)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)

RISK_FREE_RATE = 0.05  # annual, e.g. US T-bill rate


def get_prices_data(ticker: str = None) -> pd.DataFrame:
    """
    Fetch adjusted close prices for one ticker (or all if ticker=None).
    Returns DataFrame with columns: date (index), ticker, company_name, adj_close
    """
    engine = get_engine()

    query = """
    SELECT
        p.date,
        a.ticker,
        a.company_name,
        p.adj_close
    FROM prices p
    JOIN assets a ON p.asset_id = a.id
    WHERE p.adj_close IS NOT NULL
    """

    params = {}

    if ticker:
        query += " AND a.ticker=:ticker"
        params["ticker"] = ticker.upper()

    query += " ORDER BY a.ticker, p.date"

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)

    if df.empty:
        logger.warning(f"No price data found{f' for {ticker}' if ticker else ''}")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds simple returns, log returns, cumulative returns, and Sharpe ratio.
    Expects df with 'adj_close' column (and optionally 'ticker').
    """
    if df.empty:
        return df

    df = df.sort_index()

    if 'ticker' in df.columns:
        grouped = df.groupby('ticker')['adj_close']
        df["simple_return"] = grouped.pct_change()
        df["log_return"] = np.log(df["adj_close"] / df.groupby('ticker')['adj_close'].shift(1))
        df["cum_simple_return"] = (1 + df["simple_return"].fillna(0)).groupby(df['ticker']).cumprod() - 1
        df["cum_log_return"] = df.groupby('ticker')['log_return'].cumsum()
    else:
        df["simple_return"] = df["adj_close"].pct_change()
        df["log_return"] = np.log(df["adj_close"] / df["adj_close"].shift(1))
        df["cum_simple_return"] = (1 + df["simple_return"].fillna(0)).cumprod() - 1
        df["cum_log_return"] = df["log_return"].cumsum()

    return df


def sharpe_ratio(df: pd.DataFrame) -> float:
    """Annualised Sharpe ratio using daily simple returns."""
    daily_rf = RISK_FREE_RATE / 252
    excess = df["simple_return"].dropna() - daily_rf
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def plot_returns(df: pd.DataFrame, ticker: str, company_name: str):
    """Plot adj_close + cumulative returns for a single ticker."""
    if df.empty:
        return

    label = f"{ticker} ({company_name})"

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Price on left axis
    ax1.plot(df.index, df["adj_close"], color="steelblue", label="Adj Close")
    ax1.set_ylabel("Adjusted Close Price", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    # Cumulative returns on right axis
    ax2 = ax1.twinx()
    ax2.plot(df.index, df["cum_simple_return"] * 100, color="seagreen", label="Cum Simple Return (%)")
    ax2.plot(df.index, df["cum_log_return"] * 100, color="darkorange", linestyle="--", label="Cum Log Return (%)")
    ax2.set_ylabel("Cumulative Return (%)")
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax2.tick_params(axis="y", labelcolor="black")

    plt.title(f"{label} — Price & Cumulative Returns", fontsize=13, fontweight="bold")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "plots", f"{ticker}_returns_{datetime.now().strftime('%Y%m%d')}.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot: {os.path.basename(save_path)}")


def plot_correlation_matrix(df_raw: pd.DataFrame):
    """
    Build and save a correlation heatmap of daily returns across all tickers.
    df_raw must contain 'ticker' and 'adj_close' columns.
    """
    # Pivot to wide format: columns = tickers, rows = dates
    pivot = df_raw.pivot_table(index=df_raw.index, columns="ticker", values="adj_close")
    daily_returns = pivot.pct_change().dropna(how="all")
    corr = daily_returns.corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1, vmax=1,
        center=0,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Daily Returns Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, "plots", f"correlation_matrix_{datetime.now().strftime('%Y%m%d')}.png")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved correlation matrix: {os.path.basename(save_path)}")


def print_summary(df: pd.DataFrame, ticker: str, company_name: str):
    """Print stats including Sharpe ratio for a single ticker."""
    if df.empty:
        return

    sr = sharpe_ratio(df)

    stats = {
        "Period start":              df.index.min().date(),
        "Period end":                df.index.max().date(),
        "Total days":                len(df),
        "Avg daily simple return":   f"{df['simple_return'].mean():.4%}",
        "Avg daily log return":      f"{df['log_return'].mean():.4%}",
        "Total cum simple return":   f"{df['cum_simple_return'].iloc[-1]:.2%}",
        "Total cum log return":      f"{df['cum_log_return'].iloc[-1]:.2%}",
        "Daily return volatility":   f"{df['simple_return'].std():.4%}",
        "Sharpe ratio (annual)":     f"{sr:.2f}",
    }

    print(f"\n=== {ticker} ({company_name}) ===")
    for k, v in stats.items():
        print(f"  {k:<28}: {v}")


def main():
    df_raw = get_prices_data()  # all tickers

    if df_raw.empty:
        logger.error("No data available. Run ingestion first.")
        return

    # Build a lookup: ticker -> company_name
    name_map = (
        df_raw.reset_index()[["ticker", "company_name"]]
        .drop_duplicates("ticker")
        .set_index("ticker")["company_name"]
        .to_dict()
    )

    # Per-ticker analysis
    for ticker, group in df_raw.groupby("ticker"):
        company_name = name_map.get(ticker, ticker)
        logger.info(f"Processing {ticker} ({company_name}) — {len(group)} rows")

        df = calculate_returns(group[["adj_close"]].copy())
        print_summary(df, ticker, company_name)
        plot_returns(df, ticker, company_name)

    # Correlation matrix across all tickers
    logger.info("Generating correlation matrix...")
    plot_correlation_matrix(df_raw)

    logger.info("Returns calculation & plotting complete.")


if __name__ == "__main__":
    main()