import logging
import sys
import os
from datetime import datetime

# Ensure the project root is on sys.path so sibling packages (db, config) are importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import yfinance as yf
from sqlalchemy import text
from db.connection import get_engine, get_connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TICKERS = {
    # Equities
    "AAPL": ("Apple Inc.", "Technology"),
    "MSFT": ("Microsoft Corp.", "Technology"),
    "JPM": ("JPMorgan Chase", "Financials"),
    "GS": ("Goldman Sachs", "Financials"),
    "XOM": ("Exxon Mobil", "Energy"),
    "JNJ": ("Johnson & Johnson", "Healthcare"),
    "AMZN": ("Amazon", "Consumer Discretionary"),
    "PG": ("Procter & Gamble", "Consumer Staples"),
    "NEE": ("NextEra Energy", "Utilities"),
    "CAT": ("Caterpillar", "Industrials"),
    "BHP": ("BHP Group", "Materials"),
    "V": ("Visa Inc.", "Financials"),
    # Index ETFs
    "SPY": ("S&P 500 Index ETF", "Index"),
    "QQQ": ("Nasdaq 100 Index ETF", "Index"),
    "DIA": ("Dow Jones Industrial Avg ETF", "Index"),
    "IWM": ("Russell 2000 Index ETF", "Index"),
    # Commodities
    "GLD": ("Gold ETF", "Commodities"),
    "SLV": ("Silver ETF", "Commodities"),
    # Fixed Income
    "TLT": ("Long-Term Treasury ETF", "Fixed Income"),
    "BND": ("Total Bond Market ETF", "Fixed Income"),
    # Emerging Markets
    "EEM": ("Emerging Markets ETF", "Emerging Markets"),
    # Volatility
    "VXX": ("Volatility ETF", "Volatility"),
}

START_DATE = "2020-01-01"
END_DATE   = None 

def insert_assets():
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for ticker, (name, sector) in TICKERS.items():
                cur.execute("""
                    INSERT INTO assets (ticker, company_name, sector)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (ticker) DO NOTHING
                """, (ticker, name, sector))
        conn.commit()
        logger.info("Assets inserted or already present.")
    except Exception as e:
        logger.error(f"Error inserting assets: {e}")
        conn.rollback()
    finally:
        conn.close()

def fetch_and_insert_prices():
    engine = get_engine()

    with engine.connect() as conn:
        result = conn.execute(text("SELECT ticker,id FROM assets"))
        asset_map = {row.ticker: row.id for row in result}
    if not asset_map:
        logger.error("No assets found in DB. Run insert_assets first.")
        return

    for ticker in TICKERS:
        if ticker not in asset_map:
            logger.warning(f"Ticker {ticker} not in DB - skipping.")
            continue
        asset_id = asset_map[ticker]
        logger.info(f"Fetching {ticker} (asset_id={asset_id})....")

        try:
            df = yf.download(
                ticker,
                start=START_DATE,
                end=END_DATE,
                progress=False,
                auto_adjust=False,
                repair=True
            )

            if df.empty:
                logger.warning(f"No data for {ticker}")
                continue

            # Handle possible MultiIndex columns (common in recent yfinance)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)  # take first level (Open, High, etc.)

            # Standardize column names (case insensitive-ish)
            df = df.rename(columns={
                col: col.lower().replace(" ", "_") 
                for col in df.columns
            }).rename(columns={
                "date": "date",  # after reset_index it might be 'Date'
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "adj_close": "adj_close",
                "volume": "volume",
            })

            df = df.reset_index()  # Date becomes column
            if "Date" in df.columns:
                df = df.rename(columns={"Date": "date"})

            # Select & prepare
            df = df[["date", "open", "high", "low", "close", "volume", "adj_close"]]
            df["asset_id"] = asset_id
            
            # Optional: round to match your NUMERIC(12,4)
            for col in ["open", "high", "low", "close", "adj_close"]:
                df[col] = df[col].round(4)

            # Upsert: insert new rows, skip duplicates based on (asset_id, date)
            insert_query = text("""
                INSERT INTO prices (date, open, high, low, close, volume, adj_close, asset_id)
                VALUES (:date, :open, :high, :low, :close, :volume, :adj_close, :asset_id)
                ON CONFLICT (asset_id, date) DO NOTHING
            """)

            rows = df.to_dict(orient="records")
            inserted = 0
            with engine.begin() as upsert_conn:
                for row in rows:
                    result = upsert_conn.execute(insert_query, row)
                    inserted += result.rowcount

            logger.info(f"→ {ticker} done — {inserted}/{len(df)} new rows inserted (duplicates skipped)")

        except Exception as e:
            logger.error(f"Failed on {ticker}: {str(e)}", exc_info=True)
            continue


if __name__ == "__main__":
    logger.info("Starting ingestion...")
    insert_assets()
    fetch_and_insert_prices()
    logger.info("Ingestion complete.")
