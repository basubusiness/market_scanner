import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

st.set_page_config(page_title="Market Decision Engine 2026", layout="wide")

# ---------------- VIX ----------------
def get_live_vix():
    try:
        vix_df = yf.download("^VIX", period="1d", progress=False)
        if not vix_df.empty:
            if isinstance(vix_df.columns, pd.MultiIndex):
                vix_df.columns = vix_df.columns.get_level_values(0)
            return float(vix_df['Close'].iloc[-1])
    except:
        return 20.0
    return 20.0

# ---------------- RSI ----------------
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss.replace(0, 0.001)
    return 100 - (100 / (1 + rs))

# ---------------- UI ----------------
st.title("🏹 Market Decision Engine v5.0")

col_vix, col_fg = st.columns(2)

with col_vix:
    live_vix = get_live_vix()
    st.metric("📊 Live VIX", f"{live_vix:.2f}")

with col_fg:
    fg_index = st.slider("🧠 Fear & Greed", 0, 100, 50)

# ---------------- Risk Multiplier ----------------
risk_multiplier = 1.0 + ((live_vix / 20) * ((100 - fg_index) / 50))
st.sidebar.write(f"Risk Multiplier: {risk_multiplier:.2f}x")

# ---------------- Core Engine ----------------
@st.cache_data(ttl=3600)
def analyze_ticker(ticker):
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df['Close']
        price = float(close.iloc[-1])

        ma200 = float(close.rolling(200).mean().iloc[-1])
        rsi = float(calculate_rsi(close).iloc[-1])

        dist_ma = ((price - ma200) / ma200) * 100

        # ---------------- Volatility ----------------
        vol = close.pct_change().rolling(20).std().iloc[-1] * 100

        # ---------------- Trend ----------------
        trend_down = close.iloc[-1] < close.iloc[-5]

        # ---------------- Falling Knife ----------------
        knife_threshold = -15 * (1 / risk_multiplier)
        knife = (dist_ma < knife_threshold) and (rsi < 35) and trend_down

        # ---------------- Confidence ----------------
        confidence = (
            min(abs(dist_ma) / 20, 1) * 0.5 +
            min(abs(50 - rsi) / 50, 1) * 0.5
        )

        # Adjust confidence by volatility
        confidence = confidence * (1 - min(vol / 5, 0.5))

        # ---------------- Action ----------------
        if dist_ma < -10 and rsi < 35:
            action = "🟢 BUY"
        elif dist_ma > 10 and rsi > 70:
            action = "🔴 SELL"
        else:
            action = "🔵 WATCH"

        # Strength labeling
        if confidence > 0.7:
            strength = "🔥 Strong"
        elif confidence > 0.4:
            strength = "⚖️ Medium"
        else:
            strength = "🔍 Weak"

        return {
            "Ticker": ticker,
            "Price": round(price, 2),
            "Dist%": round(dist_ma, 1),
            "RSI": round(rsi, 1),
            "Vol": round(vol, 2),
            "Confidence": round(confidence, 2),
            "Signal": f"{strength} {action}",
            "Knife": "🔪 Yes" if knife else "",
            "Link": f"https://finance.yahoo.com/quote/{ticker}"
        }

    except:
        return None

# ---------------- Universe ----------------
watchlist = [
    "QQQ","SPY","VTI","GLD","TLT",
    "NVDA","MSFT","AAPL","AMZN","TSLA",
    "AMD","META","GOOGL","XLF","XLE","XLV"
]

# ---------------- Execution ----------------
if st.button("🔄 Run Full Market Scan"):

    results = []

    with st.spinner("Scanning market..."):
        for t in watchlist:
            res = analyze_ticker(t)
            if res:
                results.append(res)

    if results:

        df = pd.DataFrame(results)

        # ---------------- Ranking ----------------
        df["Rank"] = df["Dist%"].rank()
        df = df.sort_values("Dist%", ascending=True)

        # ---------------- Display ----------------
        st.subheader("🏆 Top Buy Opportunities")
        st.dataframe(df.head(5), use_container_width=True)

        st.subheader("⚠️ Risk / Sell Zone")
        st.dataframe(df.tail(5), use_container_width=True)

        st.subheader("📊 Full Market View")
        st.dataframe(
            df,
            column_config={
                "RSI": st.column_config.ProgressColumn(min_value=0, max_value=100),
                "Link": st.column_config.LinkColumn("Chart")
            },
            use_container_width=True
        )

# ---------------- Explanation ----------------
with st.expander("🔍 How this works"):
    st.write("""
This system combines:

• Distance from long-term trend (200MA)  
• Momentum (RSI)  
• Volatility (risk adjustment)  
• Sentiment (VIX + Fear & Greed)  

Confidence Score:
- Higher when strong deviation + momentum  
- Lower when volatility is high  

Falling Knife:
- Deep drop + weak momentum + still falling  

Goal:
Find best opportunities RELATIVE to entire market
""")
