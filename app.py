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

# ---------------- HEADER ----------------
st.title("🏹 Market Decision Engine v5.2")

col_vix, col_fg = st.columns(2)

with col_vix:
    live_vix = get_live_vix()
    st.metric("📊 Live VIX", f"{live_vix:.2f}")
    st.markdown("🔗 https://finance.yahoo.com/quote/%5EVIX/")

with col_fg:
    fg_index = st.slider("🧠 Fear & Greed", 0, 100, 50)
    st.markdown("🔗 https://edition.cnn.com/markets/fear-and-greed")

# ---------------- RISK CONTEXT ----------------
risk_multiplier = 1.0 + ((live_vix / 20) * ((100 - fg_index) / 50))

st.sidebar.subheader("🛡️ Market Context")

if risk_multiplier > 2:
    st.sidebar.error(f"🔥 High Fear ({risk_multiplier:.2f}x) → Volatile opportunities")
elif risk_multiplier < 1.2:
    st.sidebar.info(f"😌 Calm Market ({risk_multiplier:.2f}x)")
else:
    st.sidebar.warning(f"⚖️ Normal Risk ({risk_multiplier:.2f}x)")

# ---------------- ENGINE ----------------
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

        # Volatility
        vol = close.pct_change().rolling(20).std().iloc[-1] * 100

        # Trend
        trend_down = close.iloc[-1] < close.iloc[-5]

        # Falling Knife
        knife_threshold = -15 * (1 / risk_multiplier)
        knife = (dist_ma < knife_threshold) and (rsi < 35) and trend_down

        # Confidence
        confidence = (
            min(abs(dist_ma) / 20, 1) * 0.5 +
            min(abs(50 - rsi) / 50, 1) * 0.5
        )
        confidence *= (1 - min(vol / 5, 0.5))

        # Action
        if dist_ma < -10 and rsi < 35:
            action = "BUY"
        elif dist_ma > 10 and rsi > 70:
            action = "SELL"
        else:
            action = "WAIT"

        # Strength
        if confidence > 0.7:
            strength = "🔥 Strong"
        elif confidence > 0.4:
            strength = "⚖️ Medium"
        else:
            strength = "🔍 Weak"

        signal = f"{action} ({strength})"

        return {
            "Ticker": ticker,
            "Price": round(price, 2),
            "Dist%": round(dist_ma, 1),
            "RSI": round(rsi, 1),
            "Vol": round(vol, 2),
            "Confidence": round(confidence, 2),
            "Signal": signal,
            "Knife": "🔴 Falling Knife" if knife else "",
            "Yahoo": f"https://finance.yahoo.com/quote/{ticker}",
            "ETF": f"https://www.etf.com/{ticker}",
            "JustETF": f"https://www.justetf.com/en/search.html?query={ticker}"
        }

    except:
        return None

# ---------------- UNIVERSE ----------------
watchlist = [
    "QQQ","SPY","VTI","GLD","TLT",
    "NVDA","MSFT","AAPL","AMZN","TSLA",
    "AMD","META","GOOGL","XLF","XLE","XLV"
]

# ---------------- RUN ----------------
if st.button("🔄 Run Market Scan", type="primary"):

    results = []

    with st.spinner("Scanning market..."):
        for t in watchlist:
            res = analyze_ticker(t)
            if res:
                results.append(res)

    if results:

        df = pd.DataFrame(results)

        # ---------------- SCORING ----------------
        df["Score"] = (
            (-df["Dist%"] / 20) * 0.5 +
            ((50 - df["RSI"]) / 50) * 0.3 +
            (df["Confidence"]) * 0.2
        )

        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        df["Rank"] = df.index + 1

        # ---------------- ALLOCATION LOGIC (FROM OLD APP) ----------------
        def allocation_decision(row):
            score = 0
        
            # Sentiment
            if fg_index < 35:
                score += 40
        
            # RSI
            if row["RSI"] < 40:
                score += 30
        
            # Trend (below MA = good entry)
            if row["Dist%"] < 0:
                score += 30
        
            if score >= 70:
                return f"🔥 {baseline * 2:,.0f}"
            elif score >= 35:
                return f"⚖️ {baseline:,.0f}"
            else:
                return f"⚠️ {baseline * 0.5:,.0f}"
        
        df["Suggested €"] = df.apply(allocation_decision, axis=1)

        # ---------------- TODAY SIGNALS ----------------
        top = df.iloc[0]
        worst = df.iloc[-1]

        st.subheader("🎯 Today’s Signals")

        c1, c2 = st.columns(2)

        with c1:
            st.success(f"""
🔥 BEST OPPORTUNITY  
**{top['Ticker']}**

Signal: {top['Signal']}  
Confidence: {top['Confidence']}  
Distance: {top['Dist%']}%
""")

        with c2:
            st.error(f"""
⚠️ RISK / AVOID  
**{worst['Ticker']}**

Signal: {worst['Signal']}  
Confidence: {worst['Confidence']}  
Distance: {worst['Dist%']}%
""")

        # ---------------- TOP / BOTTOM ----------------
        st.subheader("🏆 Top Opportunities")
        st.dataframe(df.head(5), use_container_width=True, hide_index=True)

        st.subheader("⚠️ Risk / Avoid")
        st.dataframe(df.tail(5), use_container_width=True, hide_index=True)

        # ---------------- FULL TABLE ----------------
        st.subheader("📊 Full Market Scan")

        st.dataframe(
            df,
            column_config={
                "Rank": st.column_config.NumberColumn("🏆 Rank"),
                "RSI": st.column_config.ProgressColumn(min_value=0, max_value=100),
                "Suggested €": st.column_config.TextColumn("💰 Action"),
                "Yahoo": st.column_config.LinkColumn("Yahoo", display_text="Chart"),
                "ETF": st.column_config.LinkColumn("ETF", display_text="Stats"),
                "JustETF": st.column_config.LinkColumn("EU ETF", display_text="Check")
            },
            use_container_width=True,
            hide_index=True
        )

# ---------------- EXPLANATION ----------------
with st.expander("🔍 How this works"):
    st.write("""
This system combines:

• Distance from long-term trend (200MA)  
• Momentum (RSI)  
• Volatility (risk adjustment)  
• Sentiment (VIX + Fear & Greed)  

Score:
Ranks opportunities across entire market  

Confidence:
Higher when signal is strong and stable  

Falling Knife:
Deep drop + still falling → high risk  

Goal:
Find best opportunities RELATIVE to market
""")
