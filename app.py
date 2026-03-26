import streamlit as st
st.set_page_config(page_title="Market Decision Engine 2026", layout="wide")

import yfinance as yf
import pandas as pd
import numpy as np
import financedatabase as fd


@st.cache_data(show_spinner="Loading market universe...")
def load_universe():
    etfs = fd.ETFs().select()
    equities = fd.Equities().select()
    
    etfs["type"] = "ETF"
    equities["type"] = "Stock"

    df = pd.concat([etfs, equities], axis=0)
    df = df.reset_index()
    df.rename(columns={df.columns[0]: "ticker"}, inplace=True)

    return df


universe = load_universe()
universe = universe.dropna(subset=["ticker"])
universe = universe[universe["ticker"].str.len() <= 5]

# ✅ FIX: handle missing metadata safely
universe["country"] = universe.get("country", "Unknown")
universe["sector"] = universe.get("sector", "Unknown")

# ---------------- FILTER UI ----------------
st.sidebar.subheader("🌍 Market Filters")

asset_type = st.sidebar.multiselect(
    "Asset Type",
    ["ETF", "Stock"],
    default=["ETF"]
)

country = st.sidebar.multiselect(
    "Country",
    sorted(universe["country"].dropna().unique()),
    default=["United States"]
)

sector = st.sidebar.multiselect(
    "Sector",
    sorted(universe["sector"].dropna().unique())
)

# ✅ FIX: apply filters safely
filtered = universe[universe["type"].isin(asset_type)]

if country:
    filtered = filtered[filtered["country"].isin(country)]

if sector:
    filtered = filtered[filtered["sector"].isin(sector)]

# ✅ Optional UX hint (non-breaking)
if "ETF" in asset_type and sector:
    st.sidebar.info("ℹ️ Sector filter may not work well for ETFs")

# ✅ DEBUG (safe, lightweight)
st.sidebar.caption(f"Universe: {len(universe)} | Filtered: {len(filtered)}")

tickers = filtered["ticker"].tolist()

# ✅ FIX: remove duplicates BEFORE use
tickers = list(dict.fromkeys(tickers))

# ✅ FIX: safety checks BEFORE loop
if len(tickers) == 0:
    st.warning("No assets found")
    st.stop()

if len(tickers) > 30:
    st.warning("⚠️ Limiting scan to 30 assets for performance")

tickers = tickers[:30]


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


# ---------------- ALLOCATION ----------------
def allocation_decision(row, baseline, fg_index):
    score = 0

    if fg_index < 35:
        score += 40
    if row["RSI"] < 40:
        score += 30
    if row["Dist%"] < 0:
        score += 30

    if score >= 70:
        return f"🔥 {baseline * 2:,.0f}"
    elif score >= 35:
        return f"⚖️ {baseline:,.0f}"
    else:
        return f"⚠️ {baseline * 0.5:,.0f}"


# ---------------- HEADER ----------------
st.title("🏹 Market Decision Engine v5.3")

col_vix, col_fg = st.columns(2)

with col_vix:
    live_vix = get_live_vix()
    st.metric("📊 Live VIX", f"{live_vix:.2f}")
    st.markdown("🔗 https://finance.yahoo.com/quote/%5EVIX/")

with col_fg:
    fg_index = st.slider("🧠 Fear & Greed", 0, 100, 50)
    st.markdown("🔗 https://edition.cnn.com/markets/fear-and-greed")

baseline = st.sidebar.number_input("💰 Monthly Investment", value=1000)

# ---------------- RISK ----------------
risk_multiplier = 1.0 + ((live_vix / 20) * ((100 - fg_index) / 50))

st.sidebar.subheader("🛡️ Market Context")

if risk_multiplier > 2:
    st.sidebar.error(f"🔥 High Fear ({risk_multiplier:.2f}x)")
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

        vol = close.pct_change().rolling(20).std().iloc[-1] * 100
        trend_down = close.iloc[-1] < close.iloc[-5]

        knife_threshold = -15 * (1 / risk_multiplier)
        knife = (dist_ma < knife_threshold) and (rsi < 35) and trend_down

        confidence = (
            min(abs(dist_ma) / 20, 1) * 0.5 +
            min(abs(50 - rsi) / 50, 1) * 0.5
        )
        confidence *= (1 - min(vol / 5, 0.5))

        if dist_ma < -10 and rsi < 35:
            action = "BUY"
        elif dist_ma > 10 and rsi > 70:
            action = "SELL"
        else:
            action = "WAIT"

        strength = "🔥 Strong" if confidence > 0.7 else "⚖️ Medium" if confidence > 0.4 else "🔍 Weak"
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


# ---------------- RUN ----------------
if st.button("🔄 Run Market Scan", type="primary"):

    results = []

    with st.spinner("Scanning market..."):
        for t in tickers:
            res = analyze_ticker(t)
            if res:
                results.append(res)

    if results:
        df = pd.DataFrame(results)

        df["Score"] = (
            (-df["Dist%"] / 20) * 0.5 +
            ((50 - df["RSI"]) / 50) * 0.3 +
            (df["Confidence"]) * 0.2
        )

        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        df["Rank"] = df.index + 1

        df["Suggested €"] = df.apply(
            lambda row: allocation_decision(row, baseline, fg_index),
            axis=1
        )

        # UI unchanged...
