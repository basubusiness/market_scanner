import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(page_title="Global Knife Scanner 2026", layout="wide")

# --- UI HEADER & MASTER INPUTS ---
st.title("🏹 Global Market Scanner v3.0")
col_a, col_b = st.columns([2, 1])

with col_a:
    st.caption("Deep Technical Analysis: Falling Knives, RSI, and Entry/Exit Signals")

with col_b:
    # Master Fear Index Input (Simulating Sentiment Filter)
    fear_index = st.slider("📊 Market Fear Index (VIX/Sentiment)", 0, 100, 50, help="0=Greed, 100=Extreme Fear. Use this to gauge your entry aggression.")
    fear_multiplier = 1.0 + (fear_index / 200) # Higher fear allows for deeper 'Knife' threshold

# --- DATA & MAPPING ---
ISIN_DATABASE = {
    "QQQ": "US46090E1038", "NVDA": "US67066G1040", "TSLA": "US88160R1014", "AMD": "US0079031078",
    "SCHD": "US8085247976", "VTV": "US9229087440", "VYM": "US9219464065", "KO": "US1912161007",
    "VEA": "US9219378182", "EWJ": "US4642867710", "VWO": "US9220428588", "INDA": "US46429B5984",
    "GLD": "US78463V1035", "TLT": "US4642874329", "VDC": "US9220427424", "XLP": "US81369Y3080"
}

tickers = {
    "Equity ETFs": ["QQQ", "SCHD", "VTV", "VYM"],
    "International ETFs": ["VEA", "EWJ", "VWO", "INDA"],
    "Commodity/Bond": ["GLD", "TLT", "VDC", "XLP"],
    "Growth Stocks": ["NVDA", "TSLA", "AMD", "KO"]
}

# --- TECHNICAL ENGINE ---
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=3600)
def get_full_analysis(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Calc Indicators
        df['MA200'] = df['Close'].rolling(200).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        
        last = df.iloc[-1]
        dist_ma = ((last['Close'] - last['MA200']) / last['MA200']) * 100
        rsi_val = last['RSI']
        
        # Entry/Exit Logic
        # BUY: Price < MA200 AND RSI < 35 (Oversold)
        # SELL: Price > MA200 AND RSI > 70 (Overbought)
        action = "Hold"
        if dist_ma < -10 and rsi_val < 35: action = "🎯 BUY / ENTRY"
        elif dist_ma > 10 and rsi_val > 65: action = "⚠️ SELL / EXIT"
        
        # Falling Knife Logic (Modified by Master Fear Index)
        knife_threshold = -15 * fear_multiplier
        status = "Neutral"
        if dist_ma < knife_threshold: status = "🔪 Falling Knife"
        elif abs(dist_ma) < 5: status = "🟢 Stabilized"
        
        return {
            "Price": round(float(last['Close']), 2),
            "Dist_MA": round(dist_ma, 1),
            "RSI": round(rsi_val, 1),
            "Action": action,
            "Status": status
        }
    except:
        return None

# --- MAIN EXECUTION ---
if st.button("🚀 Execute Global Scan", type="primary"):
    results = []
    with st.spinner("Crunching RSI and Moving Averages..."):
        for cat, list_t in tickers.items():
            for t in list_t:
                analysis = get_full_analysis(t)
                if not analysis: continue
                
                isin = ISIN_DATABASE.get(t, "N/A")
                results.append({
                    "Ticker": t,
                    "ISIN": isin,
                    "Status": analysis["Status"],
                    "Action": analysis["Action"],
                    "Price": analysis["Price"],
                    "Dist. 200MA": f"{analysis['Dist_MA']}%",
                    "RSI": analysis["RSI"],
                    "YF": f"https://finance.yahoo.com/quote/{t}",
                    "ETF.com": f"https://www.etf.com/{t}",
                    "JustETF": f"https://www.justetf.com/en/search.html?search=ETFS&query={isin}"
                })

    if results:
        df = pd.DataFrame(results)
        st.data_editor(
            df,
            column_config={
                "YF": st.column_config.LinkColumn("Yahoo", display_text="View"),
                "ETF.com": st.column_config.LinkColumn("ETF.com", display_text="Analysis"),
                "JustETF": st.column_config.LinkColumn("JustETF", display_text="Europe"),
                "RSI": st.column_config.ProgressColumn("RSI", min_value=0, max_value=100, format="%.0f"),
                "Price": st.column_config.NumberColumn(format="$%.2f"),
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.error("No data found.")

st.divider()
st.info("💡 Strategy: A 'Buy' signal is triggered when the asset is significantly below its 200MA and RSI is oversold (<35).")
