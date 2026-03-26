import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Page Configuration
st.set_page_config(page_title="Market Decision Engine 2026", layout="wide")

# --- CUSTOM CSS FOR ACTION STATUSES ---
st.markdown("""
    <style>
    .buy-signal { color: #2ecc71; font-weight: bold; }
    .sell-signal { color: #e74c3c; font-weight: bold; }
    .watch-signal { color: #f1c40f; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR & MASTER INPUTS ---
with st.sidebar:
    st.header("⚙️ Master Parameters")
    vix_value = st.number_input("📊 Current VIX (Fear Index)", min_value=1.0, max_value=100.0, value=20.0, step=0.1)
    
    st.divider()
    st.subheader("Threshold Sensitivity")
    # Higher VIX allows for more aggressive "Knife" catching
    risk_adj = 1.0 + (vix_value / 50) 
    st.write(f"VIX Risk Multiplier: **{risk_adj:.2f}x**")
    
    st.divider()
    st.info("🎯 **Entry Logic:** RSI < 30 & Price < 200MA\n\n⚠️ **Exit Logic:** RSI > 70 & Price > 200MA")

# --- DATA MAPPING ---
ISIN_DATABASE = {
    "QQQ": "US46090E1038", "NVDA": "US67066G1040", "TSLA": "US88160R1014", "AMD": "US0079031078",
    "SCHD": "US8085247976", "VTV": "US9229087440", "VYM": "US9219464065", "KO": "US1912161007",
    "VEA": "US9219378182", "EWJ": "US4642867710", "VWO": "US9220428588", "INDA": "US46429B5984",
    "GLD": "US78463V1035", "TLT": "US4642874329", "VDC": "US9220427424", "XLP": "US81369Y3080"
}

# --- FUNCTIONS ---
@st.cache_data(ttl=3600)
def get_analysis(ticker, risk_multiplier):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Indicators
        df['200MA'] = df['Close'].rolling(200).mean()
        
        # RSI Calculation
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        curr = df.iloc[-1]
        close = float(curr['Close'])
        ma200 = float(curr['200MA'])
        rsi = float(curr['RSI'])
        dist_ma = ((close - ma200) / ma200) * 100
        
        # --- DECISION LOGIC ---
        # Baseline threshold -15%, adjusted by VIX
        knife_limit = -15 * risk_multiplier 
        
        status = "🟡 Neutral"
        if dist_ma < knife_limit: status = "🔪 Falling Knife"
        elif abs(dist_ma) < 5: status = "🟢 Stabilizing"
        
        decision = "⚖️ HOLD / WAIT"
        if dist_ma < -5 and rsi < 30: decision = "🎯 PREPARE BUY"
        elif dist_ma > 10 and rsi > 70: decision = "⚠️ PREPARE EXIT"

        return {
            "Price": round(close, 2),
            "Dist_MA": round(dist_ma, 1),
            "RSI": round(rsi, 1),
            "Status": status,
            "Decision": decision
        }
    except: return None

# --- MAIN DASHBOARD ---
st.title("🏹 Global Market Scanner v3.5")

if st.button("Execute Global Scan", type="primary"):
    results = []
    with st.spinner("Analyzing Global Markets..."):
        all_tickers = [t for sub in [["QQQ", "NVDA", "TSLA", "AMD"], ["SCHD", "VTV", "VYM", "KO"], ["VEA", "EWJ", "VWO", "INDA"], ["GLD", "TLT", "VDC", "XLP"]] for t in sub]
        
        for t in all_tickers:
            data = get_analysis(t, risk_adj)
            if not data: continue
            
            isin = ISIN_DATABASE.get(t, "N/A")
            results.append({
                "Ticker": t,
                "ISIN": isin,
                "Decision": data["Decision"],
                "Status": data["Status"],
                "RSI": data["RSI"],
                "Dist. 200MA": f"{data['Dist_MA']}%",
                "Price": data["Price"],
                "YF": f"https://finance.yahoo.com/quote/{t}",
                "ETF": f"https://www.etf.com/{t}",
                "JustETF": f"https://www.justetf.com/en/search.html?query={isin}"
            })

    if results:
        df = pd.DataFrame(results)
        st.data_editor(
            df,
            column_config={
                "RSI": st.column_config.ProgressColumn("RSI momentum", min_value=0, max_value=100, format="%.0f"),
                "YF": st.column_config.LinkColumn("Yahoo", display_text="Chart"),
                "ETF": st.column_config.LinkColumn("ETF.com", display_text="Stats"),
                "JustETF": st.column_config.LinkColumn("JustETF", display_text="EU Filter"),
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "Decision": st.column_config.TextColumn("🎯 Trading Signal")
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.error("Data fetch failed. Check connection.")

st.divider()
st.caption("Strategy Note: Decision signals combine RSI momentum and 200-day Moving Average distance, weighted by current VIX volatility levels.")
