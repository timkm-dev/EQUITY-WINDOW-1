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

RISK_FREE_RATE = 0.05  # annual


def get_sector_prices() -> pd.DataFrame:
    """
    Fetch adj_close, ticker, sector for all assets.
    Returns DataFrame indexed by date with columns: ticker, sector, adj_close.
    """
    engine = get_engine()
    query = """
    SELECT
        p.date,
        a.ticker,
        a.sector,
        p.adj_close
    FROM prices p
    JOIN assets a ON p.asset_id = a.id
    WHERE p.adj_close IS NOT NULL
    ORDER BY a.ticker, p.date
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    if df.empty:
        logger.error("No data found. Run ingestion first.")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def compute_sector_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each sector, compute:
      - avg total cumulative return (equal-weighted across tickers in sector)
      - avg annualised volatility
      - avg Sharpe ratio
    Returns a DataFrame indexed by sector.
    """
    daily_rf = RISK_FREE_RATE / 252
    records = []

    for sector, sector_df in df.groupby("sector"):
        tickers = sector_df["ticker"].unique()
        ticker_stats = []

        for ticker in tickers:
            t_df = sector_df[sector_df["ticker"] == ticker]["adj_close"].sort_index()
            if len(t_df) < 2:
                continue
            daily_ret = t_df.pct_change().dropna()
            cum_ret = (1 + daily_ret).prod() - 1
            vol = daily_ret.std() * np.sqrt(252)
            excess = daily_ret - daily_rf
            sharpe = (excess.mean() / excess.std()) * np.sqrt(252) if excess.std() > 0 else 0.0
            ticker_stats.append({
                "ticker": ticker,
                "total_return": cum_ret,
                "annual_vol": vol,
                "sharpe": sharpe,
            })

        if not ticker_stats:
            continue

        stats_df = pd.DataFrame(ticker_stats)
        records.append({
            "sector": sector,
            "tickers": ", ".join(tickers),
            "n_assets": len(tickers),
            "avg_total_return": stats_df["total_return"].mean(),
            "avg_annual_vol": stats_df["annual_vol"].mean(),
            "avg_sharpe": stats_df["sharpe"].mean(),
        })

    return pd.DataFrame(records).set_index("sector").sort_values("avg_total_return", ascending=False)


def compute_sector_cum_returns_over_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute equal-weighted average cumulative return per sector over time.
    Returns a wide DataFrame: index=date, columns=sector.
    """
    pivot = df.pivot_table(index=df.index, columns="ticker", values="adj_close")
    daily_returns = pivot.pct_change()

    # Map ticker -> sector
    ticker_sector = df.reset_index()[["ticker", "sector"]].drop_duplicates().set_index("ticker")["sector"]

    sector_avg_rets = {}
    for sector in ticker_sector.unique():
        tickers_in_sector = ticker_sector[ticker_sector == sector].index.tolist()
        cols = [t for t in tickers_in_sector if t in daily_returns.columns]
        if cols:
            sector_avg_rets[sector] = daily_returns[cols].mean(axis=1)

    sector_daily = pd.DataFrame(sector_avg_rets).dropna(how="all")
    sector_cum = (1 + sector_daily.fillna(0)).cumprod() - 1
    return sector_cum


def plot_sector_cumulative_returns(sector_cum: pd.DataFrame):
    """Line chart: cumulative return over time per sector."""
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab10.colors

    for i, sector in enumerate(sector_cum.columns):
        ax.plot(
            sector_cum.index,
            sector_cum[sector] * 100,
            label=sector,
            color=colors[i % len(colors)],
            linewidth=1.8,
        )

    ax.set_title("Sector Cumulative Returns (Equal-Weighted)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Cumulative Return (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "plots", f"sector_cumulative_returns_{datetime.now().strftime('%Y%m%d')}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {os.path.basename(path)}")


def plot_sector_bar_charts(stats: pd.DataFrame):
    """Three side-by-side bar charts: total return, volatility, Sharpe."""
    sectors = stats.index.tolist()
    x = range(len(sectors))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Sector Performance Comparison (Equal-Weighted Avg)", fontsize=14, fontweight="bold")

    # --- Total Return ---
    colors_ret = ["seagreen" if v >= 0 else "tomato" for v in stats["avg_total_return"]]
    axes[0].bar(x, stats["avg_total_return"] * 100, color=colors_ret, edgecolor="white")
    axes[0].set_title("Avg Total Return (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sectors, rotation=35, ha="right", fontsize=8)
    axes[0].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].grid(axis="y", alpha=0.3)

    # --- Annual Volatility ---
    axes[1].bar(x, stats["avg_annual_vol"] * 100, color="steelblue", edgecolor="white")
    axes[1].set_title("Avg Annual Volatility (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sectors, rotation=35, ha="right", fontsize=8)
    axes[1].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    axes[1].grid(axis="y", alpha=0.3)

    # --- Sharpe Ratio ---
    colors_sh = ["seagreen" if v >= 0 else "tomato" for v in stats["avg_sharpe"]]
    axes[2].bar(x, stats["avg_sharpe"], color=colors_sh, edgecolor="white")
    axes[2].set_title("Avg Sharpe Ratio (Annual)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(sectors, rotation=35, ha="right", fontsize=8)
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "plots", f"sector_bar_charts_{datetime.now().strftime('%Y%m%d')}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {os.path.basename(path)}")


def print_sector_summary(stats: pd.DataFrame):
    """Print a ranked sector summary table."""
    print("\n" + "=" * 70)
    print(f"{'SECTOR PERFORMANCE SUMMARY (ranked by total return)':^70}")
    print("=" * 70)
    header = f"{'Sector':<22} {'Assets':>6} {'Total Ret':>10} {'Ann. Vol':>10} {'Sharpe':>8}"
    print(header)
    print("-" * 70)
    for sector, row in stats.iterrows():
        print(
            f"{sector:<22} {int(row['n_assets']):>6} "
            f"{row['avg_total_return']:>9.1%} "
            f"{row['avg_annual_vol']:>9.1%} "
            f"{row['avg_sharpe']:>8.2f}"
        )
    print("=" * 70)


def main():
    logger.info("Loading price data with sector info...")
    df = get_sector_prices()
    if df.empty:
        return

    logger.info("Computing sector statistics...")
    stats = compute_sector_stats(df)
    print_sector_summary(stats)

    logger.info("Computing sector cumulative returns over time...")
    sector_cum = compute_sector_cum_returns_over_time(df)

    logger.info("Plotting sector cumulative returns...")
    plot_sector_cumulative_returns(sector_cum)

    logger.info("Plotting sector bar charts...")
    plot_sector_bar_charts(stats)

    logger.info("Sector performance analysis complete.")


if __name__ == "__main__":
    main()
