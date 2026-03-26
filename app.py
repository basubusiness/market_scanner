import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(page_title="Global Knife Scanner", layout="wide")

st.title("🏹 Global Market Scanner v2.0")
st.caption("2026 Strategy: Focused on ETFs, ISIN Tracking, and Research Links")

# 1. DATA MAPPING (ISINs & Links)
# Since yfinance doesn't consistently provide ISINs, we map the majors here.
ISIN_DATABASE = {
    "QQQ": "US46090E1038", "NVDA": "US67066G1040", "TSLA": "US88160R1014", "AMD": "US0079031078",
    "SCHD": "US8085247976", "VTV": "US9229087440", "VYM": "US9219464065", "KO": "US1912161007",
    "VEA": "US9219378182", "EWJ": "US4642867710", "VWO": "US9220428588", "INDA": "US46429B5984",
    "GLD": "US78463V1035", "TLT": "US4642874329", "VDC": "US9220427424", "XLP": "US81369Y3080"
}

# 2. TICKER GROUPS
tickers = {
    "Equity ETFs": ["QQQ", "SCHD", "VTV", "VYM"],
    "International ETFs": ["VEA", "EWJ", "VWO", "INDA"],
    "Bond/Commodity ETFs": ["GLD", "TLT", "VDC", "XLP"],
    "Individual Stocks": ["NVDA", "TSLA", "AMD", "KO"]
}

@st.cache_data(ttl=3600)
def get_market_data(ticker):
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False)
        if data.empty: return None
        # Clean MultiIndex columns if they exist
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except:
        return None

# 3. SCANNER EXECUTION
if st.button("Execute Global Scan", type="primary"):
    results = []
    
    with st.spinner("Analyzing Global ETFs and Stocks..."):
        for cat, list_t in tickers.items():
            for t in list_t:
                df = get_market_data(t)
                if df is None: continue

                # Technical Math
                close = float(df['Close'].iloc[-1])
                ma200 = float(df['Close'].rolling(200).mean().iloc[-1])
                dist_ma = ((close - ma200) / ma200) * 100
                vol = float(df['Close'].pct_change().std() * np.sqrt(252) * 100)

                # Logic for Status
                if dist_ma < -15:
                    status = "🔪 Falling Knife"
                elif -5 < dist_ma < 5:
                    status = "🟢 Buy Zone"
                else:
                    status = "🟡 Neutral"

                # Generate Dynamic Link (ETF.com for ETFs, Yahoo for Stocks)
                link = f"https://www.etf.com/{t}" if cat != "Individual Stocks" else f"https://finance.yahoo.com/quote/{t}"

                results.append({
                    "Ticker": t,
                    "ISIN": ISIN_DATABASE.get(t, "Contact Admin"),
                    "Category": cat,
                    "Price": round(close, 2),
                    "Dist. 200MA": f"{dist_ma:.1f}%",
                    "Volatility": f"{vol:.1f}%",
                    "Status": status,
                    "Research Link": link
                })

    if results:
        df_display = pd.DataFrame(results)

        # 4. DISPLAY ENHANCEMENTS (Link Columns & Formatting)
        st.data_editor(
            df_display,
            column_config={
                "Research Link": st.column_config.LinkColumn(
                    "Deep Research",
                    help="Click to open external analysis",
                    validate="^https://.*",
                    display_text="Open Chart ↗️"
                ),
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Ticker": st.column_config.TextColumn(help="Security Ticker Symbol"),
                "ISIN": st.column_config.TextColumn(help="International Securities Identification Number"),
            },
            hide_index=True,
            use_container_width=True,
            disabled=df_display.columns # Makes it read-only
        )
    else:
        st.error("Connection error or no data returned.")

st.divider()
st.sidebar.header("Scanner Settings")
st.sidebar.info("Criteria: Falling Knife is defined as Price < 15% below the 200-day Moving Average.")
