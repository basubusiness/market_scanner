import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Global Knife Scanner", layout="wide")

st.title("🏹 Global Market Scanner v2026")
st.caption("Scanning for High-Risk 'Knives' and Buyable Value")

# This keeps the app fast and prevents "Ages to load"
@st.cache_data(ttl=3600)
def get_data(ticker):
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

tickers = {
    "Tech/Growth": ["QQQ", "NVDA", "TSLA", "AMD"],
    "Value/Div": ["SCHD", "VTV", "VYM", "KO"],
    "International": ["VEA", "EWJ", "VWO", "INDA"],
    "Safe Havens": ["GLD", "TLT", "VDC", "XLP"]
}

if st.button("Execute Global Scan", type="primary"):
    results = []
    
    with st.spinner("Fetching market data..."):
        for cat, list_t in tickers.items():
            for t in list_t:
                try:
                    df = get_data(t)
                    if df.empty: continue

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

                    results.append({
                        "Ticker": t,
                        "Category": cat,
                        "Price": f"${close:.2f}",
                        "Dist. 200MA": f"{dist_ma:.1f}%",
                        "Risk (Vol)": f"{vol:.1f}%",
                        "Status": status
                    })
                except Exception:
                    continue

    if results:
        df_display = pd.DataFrame(results)
        
        # Color styling for the Status column
        def color_status(val):
            if "Knife" in val: return 'color: red; font-weight: bold'
            if "Buy" in val: return 'color: green; font-weight: bold'
            return 'color: gray'

        st.table(df_display.style.applymap(color_status, subset=['Status']))
    else:
        st.error("No data found. Check your internet connection.")

st.divider()
st.info("Note: 'Falling Knife' is defined as price >15% below the 200-day Moving Average.")
