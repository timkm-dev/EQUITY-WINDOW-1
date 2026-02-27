import pandas as pd
import streamlit as st
import os
import sys
from sqlalchemy import text

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in os.sys.path:
    os.sys.path.insert(0, PROJECT_ROOT)

from db.connection import get_engine

@st.cache_data(ttl=3600)
def load_all_prices() -> pd.DataFrame:
    """
    Fetch all price data with ticker and sector info.
    Returns DataFrame indexed by date with columns: ticker, company_name, sector, adj_close
    """
    engine = get_engine()
    
    query = """
    SELECT
        p.date,
        a.ticker,
        a.company_name,
        a.sector,
        p.adj_close
    FROM prices p
    JOIN assets a ON p.asset_id = a.id
    WHERE p.adj_close IS NOT NULL
    ORDER BY a.ticker, p.date
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.set_index("date")
        
    return df

@st.cache_data(ttl=3600)
def get_ticker_list() -> list:
    """Returns a list of formatted strings: 'TICKER - Company Name'"""
    try:
        engine = get_engine()
        query = "SELECT ticker, company_name FROM assets ORDER BY ticker"
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        if df.empty:
            return []
        
        return [f"{row['ticker']} - {row['company_name']}" for _, row in df.iterrows()]
    except Exception:
        return []
