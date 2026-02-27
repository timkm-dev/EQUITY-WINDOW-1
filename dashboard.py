import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from analysis.data import load_all_prices, get_ticker_list
from analysis.calculate_returns import sharpe_ratio, calculate_returns

# App configuration
st.set_page_config(
    page_title="Equity Window Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better padding
st.markdown("""
<style>
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 5px;
        border-left: 4px solid #00d4b2;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data(ttl=3600)
def get_data():
    return load_all_prices()

try:
    df_all = get_data()
    tickers_list = get_ticker_list()
except Exception as e:
    st.error(f"Failed to connect to the database: {e}")
    st.stop()

if df_all.empty:
    st.warning("No data found in the database. Run ingestion first.")
    st.stop()

# Sidebar
st.sidebar.title("📈 Equity Window")
st.sidebar.markdown("---")

selected_ticker_formatted = st.sidebar.selectbox("Select Asset for Deep Dive", tickers_list)
selected_ticker = selected_ticker_formatted.split(" - ")[0] if selected_ticker_formatted else None

min_date = df_all.index.min()
max_date = df_all.index.max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) != 2:
    st.stop()

start_date, end_date = date_range

# Filter dataset based on dates
df_filtered = df_all[(df_all.index >= start_date) & (df_all.index <= end_date)]

if df_filtered.empty:
    st.warning("No data in selected date range.")
    st.stop()

# Main Area Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Ticker Deep Dive", "🏭 Sector Analysis", "🔗 Correlation"])

# --- TAB 1: OVERVIEW ---
with tab1:
    st.header("All Assets Overview")
    
    # Calculate stats for all tickers
    overview_stats = []
    
    for ticker, group in df_filtered.groupby("ticker"):
        # Create a copy and ensure index is sorted
        t_df = group[["adj_close", "company_name", "sector"]].copy().sort_index()
        if len(t_df) < 2:
            continue
            
        t_returns = calculate_returns(t_df[["adj_close"]])
        if not t_returns.empty:
            sr = sharpe_ratio(t_returns)
            company_name = t_df["company_name"].iloc[0]
            sector = t_df["sector"].iloc[0]
            
            overview_stats.append({
                "Ticker": ticker,
                "Company": company_name,
                "Sector": sector,
                "Total Return (%)": t_returns["cum_simple_return"].iloc[-1] * 100,
                "Ann. Volatility (%)": t_returns["simple_return"].std() * np.sqrt(252) * 100,
                "Sharpe Ratio": round(sr, 2)
            })
            
    if overview_stats:
        overview_df = pd.DataFrame(overview_stats)
        
        # Format the dataframe for display
        st.dataframe(
            overview_df.style.background_gradient(subset=["Total Return (%)", "Sharpe Ratio"], cmap="RdYlGn")
            .format({
                "Total Return (%)": "{:.1f}%",
                "Ann. Volatility (%)": "{:.1f}%",
                "Sharpe Ratio": "{:.2f}"
            }),
            use_container_width=True,
            height=600
        )

# --- TAB 2: TICKER DEEP DIVE ---
with tab2:
    if selected_ticker:
        st.header(selected_ticker_formatted)
        
        ticker_data = df_filtered[df_filtered["ticker"] == selected_ticker].copy()
        ticker_returns = calculate_returns(ticker_data[["adj_close"]])
        
        if not ticker_returns.empty:
            # Metrics Row
            col1, col2, col3, col4 = st.columns(4)
            
            total_ret = ticker_returns["cum_simple_return"].iloc[-1] * 100
            ann_vol = ticker_returns["simple_return"].std() * np.sqrt(252) * 100
            sharpe = sharpe_ratio(ticker_returns)
            last_price = ticker_returns["adj_close"].iloc[-1]
            
            col1.metric("Last Price", f"${last_price:.2f}")
            col2.metric("Total Return", f"{total_ret:.1f}%", delta=f"{total_ret:.1f}%")
            col3.metric("Ann. Volatility", f"{ann_vol:.1f}%")
            col4.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            # Interactive Charts
            st.subheader("Price History")
            fig_price = px.line(ticker_returns.reset_index(), x='date', y='adj_close', color_discrete_sequence=['#00d4b2'])
            fig_price.update_layout(xaxis_title="", yaxis_title="Adjusted Close ($)", height=400, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_price, use_container_width=True)
            
            st.subheader("Cumulative Returns")
            
            # Melt the dataframe for plotly express
            cr_df = ticker_returns.reset_index()[['date', 'cum_simple_return', 'cum_log_return']]
            cr_df['cum_simple_return'] *= 100
            cr_df['cum_log_return'] *= 100
            
            fig_returns = go.Figure()
            fig_returns.add_trace(go.Scatter(x=cr_df['date'], y=cr_df['cum_simple_return'], 
                                            mode='lines', name='Simple Return (%)', line=dict(color='seagreen')))
            fig_returns.add_trace(go.Scatter(x=cr_df['date'], y=cr_df['cum_log_return'], 
                                            mode='lines', name='Log Return (%)', line=dict(color='darkorange', dash='dash')))
            
            fig_returns.update_layout(
                xaxis_title="", 
                yaxis_title="Return (%)",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_returns, use_container_width=True)

# --- TAB 3: SECTOR ANALYSIS ---
with tab3:
    st.header("Sector Performance")
    
    # Calculate sector cumulative returns (equal-weighted)
    pivot = df_filtered.pivot_table(index=df_filtered.index, columns="ticker", values="adj_close")
    daily_returns = pivot.pct_change()

    ticker_sector = df_filtered.reset_index()[["ticker", "sector"]].drop_duplicates().set_index("ticker")["sector"]

    sector_avg_rets = {}
    for sector in ticker_sector.unique():
        tickers_in_sector = ticker_sector[ticker_sector == sector].index.tolist()
        cols = [t for t in tickers_in_sector if t in daily_returns.columns]
        if cols:
            sector_avg_rets[sector] = daily_returns[cols].mean(axis=1)

    if sector_avg_rets:
        sector_daily = pd.DataFrame(sector_avg_rets).dropna(how="all")
        sector_cum = (1 + sector_daily.fillna(0)).cumprod() - 1
        
        # Plotly line chart for sector cumulative returns
        st.subheader("Cumulative Returns over Time")
        
        sector_cum_pct = sector_cum * 100
        fig_sec_lines = px.line(sector_cum_pct.reset_index(), x='date', y=sector_cum.columns)
        fig_sec_lines.update_layout(
            xaxis_title="", 
            yaxis_title="Cumulative Return (%)",
            legend_title="Sector",
            height=500,
            hovermode="x unified"
        )
        # Add horizontal line at 0
        fig_sec_lines.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        st.plotly_chart(fig_sec_lines, use_container_width=True)
        
        # Sector Bar Charts (similar to sector_performance.py but in Plotly)
        st.subheader("Risk vs Return by Sector")
        
        # Need to compute stats for the bar charts
        from analysis.sector_performance import compute_sector_stats
        
        stats_df = compute_sector_stats(df_filtered)
        
        if not stats_df.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig_ret = px.bar(
                    stats_df.reset_index(), 
                    x='sector', 
                    y='avg_total_return',
                    color='avg_total_return',
                    color_continuous_scale=px.colors.diverging.RdYlGn,
                    color_continuous_midpoint=0
                )
                fig_ret.update_layout(title="Avg Total Return", yaxis_title="Return", coloraxis_showscale=False, yaxis_tickformat='.0%')
                st.plotly_chart(fig_ret, use_container_width=True)
                
            with col2:
                fig_vol = px.bar(
                    stats_df.reset_index(), 
                    x='sector', 
                    y='avg_annual_vol',
                    color_discrete_sequence=['steelblue']
                )
                fig_vol.update_layout(title="Avg Annual Volatility", yaxis_title="Volatility", yaxis_tickformat='.0%')
                st.plotly_chart(fig_vol, use_container_width=True)
                
            with col3:
                fig_sharpe = px.bar(
                    stats_df.reset_index(), 
                    x='sector', 
                    y='avg_sharpe',
                    color='avg_sharpe',
                    color_continuous_scale=px.colors.diverging.RdYlGn,
                    color_continuous_midpoint=0
                )
                fig_sharpe.update_layout(title="Avg Sharpe Ratio", yaxis_title="Sharpe Ratio", coloraxis_showscale=False)
                st.plotly_chart(fig_sharpe, use_container_width=True)

# --- TAB 4: CORRELATION ---
with tab4:
    st.header("Correlation Matrix")
    st.markdown("Shows how daily returns of assets move relative to each other.")
    
    pivot = df_filtered.pivot_table(index=df_filtered.index, columns="ticker", values="adj_close")
    daily_returns = pivot.pct_change().dropna(how="all")
    corr = daily_returns.corr()
    
    fig_corr = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        zmin=-1, zmax=1
    )
    fig_corr.update_layout(height=800)
    st.plotly_chart(fig_corr, use_container_width=True)
