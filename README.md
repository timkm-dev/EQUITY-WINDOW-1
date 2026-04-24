# Equity Window

Pulls daily price data from Yahoo Finance, stores it in PostgreSQL, and shows everything in an interactive Streamlit dashboard. Tracks equities, index ETFs, commodities, bonds, and volatility -- computes returns, Sharpe ratios, sector breakdowns, and correlations.

Link to project: [Equity Window](https://equity-window-1-kkun2b79kwzakcboh3nr9p.streamlit.app/)

---

## Setup

### Environment variables

Create a `.env` file in the project root:

```
DB_HOST=localhost
DB_PORT=5434
DB_NAME=equity_db
DB_USER=postgres
DB_PASSWORD=your_password_here
```

Or set `DATABASE_URL` directly if using a hosted database.

### With Docker (recommended)

```bash
docker-compose up --build
```

Starts Postgres and the app together. Schema gets created automatically.

### Without Docker

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Make sure Postgres is running, then:

```bash
python main.py              # ingest data
streamlit run dashboard.py  # start dashboard
```

---

## Project structure

```
main.py                     # Entry point for ingestion
dashboard.py                # Streamlit dashboard
config.py                   # DB config from .env
data/ingest.py              # Fetches Yahoo Finance data, loads into DB
analysis/
  data.py                   # Shared data loaders for the dashboard
  calculate_returns.py      # Returns, Sharpe, correlation plots
  sector_performance.py     # Sector-level stats and charts
db/
  connection.py             # SQLAlchemy + psycopg2 helpers
  init.sql/                 # DB schema
migrate_to_supabase.py      # Push local data to Supabase
```

---

## Usage

**Ingest data:** `python main.py` -- pulls prices from Yahoo Finance, skips dates already in the DB.

**Dashboard:** `streamlit run dashboard.py` -- opens at `localhost:8501`. Four tabs: asset overview, individual ticker deep dive, sector comparisons, and a correlation heatmap.

**Static analysis:** Run `python analysis/calculate_returns.py` or `python analysis/sector_performance.py` to generate PNG charts in `analysis/plots/`.

---

## Things worth knowing

- Sharpe ratio uses a 5% annual risk-free rate. Change it in `calculate_returns.py` and `sector_performance.py`.
- Sector returns are equal-weighted across tickers in each sector.
- Dashboard caches data for one hour. Clear the cache from Streamlit's menu if you need fresh data immediately after ingestion.
- Default ticker list and start date (Jan 2020) are configured in `data/ingest.py`.

---

## Tech stack

Python 3.11, PostgreSQL 15, SQLAlchemy, psycopg2, yfinance, pandas, numpy, Streamlit, Plotly, matplotlib, seaborn, Docker.
