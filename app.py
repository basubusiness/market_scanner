import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(page_title="Market Decision Engine 2026", layout="wide")

# --- DATA FETCHING (VIX) ---
def get_live_vix():
    try:
        vix_df = yf.download("^VIX", period="1d", progress=False)
        if not vix_df.empty:
            # Flatten MultiIndex if necessary and get the last close
            val = vix_df['Close'].iloc[-1]
            return float(val)
    except:
        return 20.0 # Standard baseline if fetch fails

# --- HEADER & SENTIMENT INPUTS ---
st.title("🏹 Market Decision Engine v4.0")
col_vix, col_fg = st.columns(2)

with col_vix:
    live_vix = get_live_vix()
    st.metric("📊 Live VIX (^VIX)", f"{live_vix:.2f}")
    st.caption("Pulled directly from Yahoo Finance")

with col_fg:
    # Manual Input for CNN Fear & Greed
    fg_index = st.slider("🧠 CNN Fear & Greed Index", 0, 100, 50, help="0=Extreme Fear, 100=Extreme Greed")
    st.markdown("[Check CNN Fear & Greed Index](https://edition.cnn.com/markets/fear-and-greed)")

# Calculated Risk Multiplier (Concept Robustness)
# High VIX + Low F&G = Higher conviction for 'Knife' catching
risk_multiplier = 1.0 + ((live_vix / 20) * ( (100 - fg_index) / 50))
st.sidebar.write(f"🛡️ Current Risk Multiplier: **{risk_multiplier:.2f}x**")

# --- INDICATOR ENGINE ---
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

        # Technical Metrics
        close = float(df['Close'].iloc[-1])
        ma200 = float(df['Close'].rolling(200).mean().iloc[-1])
        rsi = float(calculate_rsi(df['Close']).iloc[-1])
        dist_ma = ((close - ma200) / ma200) * 100

        # ROBUSTNESS LOGIC (The "Original" metrics)
        # Entry: Price < MA200 AND RSI < 30
        # Exit: Price > MA200 AND RSI > 70
        action = "WAIT"
        if dist_ma < -10 and rsi < 30: action = "🎯 BUY / ENTRY"
        elif dist_ma > 10 and rsi > 70: action = "⚠️ SELL / EXIT"

        # Status Logic (Adjusted by VIX/F&G Multiplier)
        knife_threshold = -15 * (1/risk_multiplier) # Tighter threshold when market is greedy
        status = "Neutral"
        if dist_ma < knife_threshold: status = "🔪 Falling Knife"
        elif -5 < dist_ma < 5: status = "🟢 Value Zone"

        return {
            "Ticker": ticker,
            "ISIN": isin,
            "Price": round(close, 2),
            "Dist. 200MA": f"{dist_ma:.1f}%",
            "RSI": round(rsi, 1),
            "Status": status,
            "Action": action,
            "Yahoo": f"https://finance.yahoo.com/quote/{ticker}",
            "ETF.com": f"https://www.etf.com/{ticker}",
            "JustETF": f"https://www.justetf.com/en/search.html?query={isin}"
        }
    except: return None

# --- TICKER DATABASE ---
watchlist = {
    "QQQ": "US46090E1038", "NVDA": "US67066G1040", "TSLA": "US88160R1014",
    "SCHD": "US8085247976", "VEA": "US9219378182", "VWO": "US9220428588",
    "GLD": "US78463V1035", "TLT": "US4642874329"
}

# --- EXECUTION ---
if st.button("🔄 Refresh Analysis", type="primary"):
    results = []
    for t, isin in watchlist.items():
        data = analyze_ticker(t, isin)
        if data: results.append(data)
    
    if results:
        df_display = pd.DataFrame(results)
        st.data_editor(
            df_display,
            column_config={
                "RSI": st.column_config.ProgressColumn(min_value=0, max_value=100),
                "Yahoo": st.column_config.LinkColumn(display_text="Chart"),
                "ETF.com": st.column_config.LinkColumn(display_text="Holdings"),
                "JustETF": st.column_config.LinkColumn(display_text="EU Version"),
            },
            hide_index=True,
            use_container_width=True
        )

st.divider()
st.info("💡 **Logic:** When VIX is high and F&G is low, the 'Falling Knife' threshold becomes more lenient (easier to trigger).")
