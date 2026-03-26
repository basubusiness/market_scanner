import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(page_title="Market Decision Engine 2026", layout="wide")

# --- DATA FETCHING (LIVE VIX) ---
def get_live_vix():
    try:
        # Fetching ^VIX ticker from Yahoo Finance
        vix_df = yf.download("^VIX", period="1d", progress=False)
        if not vix_df.empty:
            # Handle potential MultiIndex columns
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = vix_df.columns.get_level_values(0)
            return float(vix_df['Close'].iloc[-1])
    except:
        return 20.0 # Standard fallback
    return 20.0

# --- HEADER & SENTIMENT INPUTS ---
st.title("🏹 Market Decision Engine v4.5")
col_vix, col_fg = st.columns(2)

with col_vix:
    live_vix = get_live_vix()
    st.metric("📊 Live VIX (^VIX)", f"{live_vix:.2f}")
    st.markdown("[View VIX on Yahoo Finance](https://finance.yahoo.com/quote/%5EVIX/)")

with col_fg:
    # Manual Input for CNN Fear & Greed as requested
    fg_index = st.slider("🧠 CNN Fear & Greed Index", 0, 100, 50, help="0=Extreme Fear, 100=Extreme Greed")
    st.markdown("[Check CNN Fear & Greed Index](https://edition.cnn.com/markets/fear-and-greed)")

# --- ROBUSTNESS LOGIC ---
# Concept: High VIX + Low F&G = Higher conviction for 'Knife' catching
# We use this to adjust the "Knife" threshold dynamically
risk_multiplier = 1.0 + ((live_vix / 20) * ((100 - fg_index) / 50))
st.sidebar.subheader("🛡️ Risk Parameters")
st.sidebar.write(f"Risk Multiplier: **{risk_multiplier:.2f}x**")

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=3600)
def analyze_ticker(ticker, isin):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = float(df['Close'].iloc[-1])
        ma200 = float(df['Close'].rolling(200).mean().iloc[-1])
        rsi = float(calculate_rsi(df['Close']).iloc[-1])
        dist_ma = ((close - ma200) / ma200) * 100

        # DECISION CRITERIA (Robust Core)
        # Entry: Price significantly below 200MA AND RSI oversold (<30)
        # Exit: Price significantly above 200MA AND RSI overbought (>70)
        action = "WAIT / HOLD"
        if dist_ma < -10 and rsi < 30: action = "🎯 BUY / ENTRY"
        elif dist_ma > 10 and rsi > 75: action = "⚠️ SELL / EXIT"

        # Status Logic (Adjusted by Sentiment Multiplier)
        # The threshold for a 'Knife' is usually -15%, but we loosen it if fear is high
        knife_threshold = -15 * (1 / risk_multiplier) 
        status = "Neutral"
        if dist_ma < knife_threshold: status = "🔪 Falling Knife"
        elif -5 < dist_ma < 5: status = "🟢 Value/Support"

        return {
            "Ticker": ticker,
            "ISIN": isin,
            "Price": round(close, 2),
            "Dist. 200MA": f"{dist_ma:.1f}%",
            "RSI": round(rsi, 1),
            "Action": action,
            "Status": status,
            "YF": f"https://finance.yahoo.com/quote/{ticker}",
            "ETF": f"https://www.etf.com/{ticker}",
            "JustETF": f"https://www.justetf.com/en/search.html?query={isin}"
        }
    except: return None

# --- TICKERS ---
watchlist = {
    "Equity": ["QQQ", "SCHD", "VTV", "VYM"],
    "Growth": ["NVDA", "TSLA", "AMD", "MSFT"],
    "International": ["VEA", "EWJ", "VWO", "INDA"],
    "Safe Havens": ["GLD", "TLT", "VDC", "XLP"]
}

ISIN_MAP = {
    "QQQ": "US46090E1038", "NVDA": "US67066G1040", "TSLA": "US88160R1014", "AMD": "US0079031078",
    "SCHD": "US8085247976", "VTV": "US9229087440", "VYM": "US9219464065", "MSFT": "US5949181045",
    "VEA": "US9219378182", "EWJ": "US4642867710", "VWO": "US9220428588", "INDA": "US46429B5984",
    "GLD": "US78463V1035", "TLT": "US4642874329", "VDC": "US9220427424", "XLP": "US81369Y3080"
}

# --- EXECUTION ---
if st.button("🔄 Refresh Analysis", type="primary"):
    results = []
    with st.spinner("Analyzing Global Confluence..."):
        for cat, list_t in watchlist.items():
            for t in list_t:
                res = analyze_ticker(t, ISIN_MAP.get(t, "N/A"))
                if res: results.append(res)
    
    if results:
        df_display = pd.DataFrame(results)
        st.data_editor(
            df_display,
            column_config={
                "RSI": st.column_config.ProgressColumn(min_value=0, max_value=100),
                "YF": st.column_config.LinkColumn("Yahoo", display_text="Chart"),
                "ETF": st.column_config.LinkColumn("ETF.com", display_text="Stats"),
                "JustETF": st.column_config.LinkColumn("JustETF", display_text="EU Filter"),
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Action": st.column_config.TextColumn("🎯 Signal")
            },
            hide_index=True,
            use_container_width=True
        )

st.divider()
st.info("💡 **Strategy Guidance:** Buy signals trigger when an asset is both technically oversold (RSI < 30) and trading significantly below its 200-day trend line.")
