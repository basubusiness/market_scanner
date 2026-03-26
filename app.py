import streamlit as st

st.set_page_config(page_title="Market Decision Engine 2026", layout="wide")

import yfinance as yf
import pandas as pd
import numpy as np
import financedatabase as fd

# ─────────────────────────────────────────────
# UNIVERSE LOADER
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Loading market universe...")
def load_universe():
    etfs = fd.ETFs().select()
    equities = fd.Equities().select()

    etfs = etfs.copy()
    equities = equities.copy()

    etfs["type"] = "ETF"
    equities["type"] = "Stock"

    df = pd.concat([etfs, equities], axis=0)
    df = df.reset_index()

    # Rename first column to 'ticker' regardless of its name
    df.rename(columns={df.columns[0]: "ticker"}, inplace=True)

    # Normalise column presence
    for col in ["country", "sector", "name"]:
        if col not in df.columns:
            df[col] = "Unknown"

    df["country"] = df["country"].fillna("Unknown")
    df["sector"] = df["sector"].fillna("Unknown")

    # Drop tickers that are empty / too long
    df = df.dropna(subset=["ticker"])
    df = df[df["ticker"].str.strip().str.len() <= 5]
    df = df[df["ticker"].str.strip() != ""]

    return df


universe = load_universe()

# ─────────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────────

st.sidebar.title("⚙️ Configuration")
st.sidebar.subheader("🌍 Market Filters")

asset_type = st.sidebar.multiselect(
    "Asset Type",
    ["ETF", "Stock"],
    default=["ETF"],
)

all_countries = sorted(universe[universe["type"].isin(asset_type)]["country"].dropna().unique())
country = st.sidebar.multiselect("Country", all_countries, default=["United States"] if "United States" in all_countries else [])

all_sectors = sorted(universe[universe["type"].isin(asset_type)]["sector"].dropna().unique())
sector = st.sidebar.multiselect("Sector", all_sectors)

if "ETF" in asset_type and sector:
    st.sidebar.info("ℹ️ Sector filter may not apply to ETFs")

# Apply filters
filtered = universe[universe["type"].isin(asset_type)].copy()

if country:
    filtered = filtered[filtered["country"].isin(country)]

if sector:
    filtered = filtered[filtered["sector"].isin(sector)]

tickers = list(dict.fromkeys(filtered["ticker"].str.strip().tolist()))

MAX_TICKERS = 50
st.sidebar.caption(f"Universe: {len(universe):,} | Filtered: {len(filtered):,} | Scanning: {min(len(tickers), MAX_TICKERS)}")

if len(tickers) > MAX_TICKERS:
    st.sidebar.warning(f"⚠️ Capped at {MAX_TICKERS} assets for performance")
    tickers = tickers[:MAX_TICKERS]

baseline = st.sidebar.number_input("💰 Monthly Investment (€)", value=1000, min_value=100, step=100)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def flatten_columns(df):
    """Flatten MultiIndex columns if present."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def get_live_vix():
    try:
        vix_df = yf.download("^VIX", period="2d", progress=False, auto_adjust=True)
        vix_df = flatten_columns(vix_df)
        if not vix_df.empty and "Close" in vix_df.columns:
            return float(vix_df["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return 20.0


def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs = gain / loss.replace(0, 0.001)
    return 100 - (100 / (1 + rs))


def allocation_label(row, baseline, fg_index):
    score = 0
    if fg_index < 35:
        score += 40
    if row["RSI"] < 40:
        score += 30
    if row["Dist%"] < 0:
        score += 30

    if score >= 70:
        return f"🔥 €{baseline * 2:,.0f}"
    elif score >= 35:
        return f"⚖️ €{baseline:,.0f}"
    else:
        return f"⚠️ €{baseline * 0.5:,.0f}"


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.title("🏹 Market Decision Engine 2026")

col_vix, col_fg = st.columns(2)

with col_vix:
    live_vix = get_live_vix()
    st.metric("📊 Live VIX", f"{live_vix:.2f}")
    st.caption("Source: [Yahoo Finance](https://finance.yahoo.com/quote/%5EVIX/)")

with col_fg:
    fg_index = st.slider("🧠 Fear & Greed Index (manual input)", 0, 100, 50)
    st.caption("Source: [CNN Fear & Greed](https://edition.cnn.com/markets/fear-and-greed)")

# Risk multiplier
risk_multiplier = 1.0 + ((live_vix / 20) * ((100 - fg_index) / 50))

st.sidebar.subheader("🛡️ Market Context")
if risk_multiplier > 2:
    st.sidebar.error(f"🔥 High Fear — Risk Multiplier: {risk_multiplier:.2f}x")
elif risk_multiplier < 1.2:
    st.sidebar.info(f"😌 Calm Market — Risk Multiplier: {risk_multiplier:.2f}x")
else:
    st.sidebar.warning(f"⚖️ Normal Risk — Risk Multiplier: {risk_multiplier:.2f}x")

# ─────────────────────────────────────────────
# TICKER ANALYSER
# ─────────────────────────────────────────────

# NOTE: risk_multiplier is intentionally NOT in cache key —
# we cache per ticker and apply risk_multiplier logic outside.
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_data(ticker: str):
    """Download and compute raw metrics for a ticker. Cached 1 h."""
    try:
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None

        df = flatten_columns(df)

        if "Close" not in df.columns:
            return None

        close = df["Close"].dropna()
        if len(close) < 30:
            return None

        price = float(close.iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
        rsi = float(calculate_rsi(close).iloc[-1])
        dist_ma = ((price - ma200) / ma200) * 100
        vol = float(close.pct_change().rolling(20).std().iloc[-1] * 100)
        trend_down = bool(close.iloc[-1] < close.iloc[-5])

        confidence = (
            min(abs(dist_ma) / 20, 1) * 0.5 +
            min(abs(50 - rsi) / 50, 1) * 0.5
        )
        confidence *= 1 - min(vol / 5, 0.5)

        return {
            "price": price,
            "ma200": ma200,
            "rsi": rsi,
            "dist_ma": dist_ma,
            "vol": vol,
            "trend_down": trend_down,
            "confidence": confidence,
        }
    except Exception:
        return None


def analyse_ticker(ticker: str, risk_multiplier: float):
    raw = fetch_ticker_data(ticker)
    if raw is None:
        return None

    dist_ma = raw["dist_ma"]
    rsi = raw["rsi"]
    vol = raw["vol"]
    confidence = raw["confidence"]
    trend_down = raw["trend_down"]

    knife_threshold = -15 * (1 / risk_multiplier)
    knife = (dist_ma < knife_threshold) and (rsi < 35) and trend_down

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
        "Price": round(raw["price"], 2),
        "MA200": round(raw["ma200"], 2),
        "Dist%": round(dist_ma, 1),
        "RSI": round(rsi, 1),
        "Vol%": round(vol, 2),
        "Confidence": round(confidence, 2),
        "Signal": signal,
        "Action": action,
        "Knife": "🔴 Falling Knife" if knife else "✅",
        "Yahoo": f"https://finance.yahoo.com/quote/{ticker}",
        "etf.com": f"https://www.etf.com/{ticker}",
        "justETF": f"https://www.justetf.com/en/search.html?query={ticker}",
    }


# ─────────────────────────────────────────────
# SCAN BUTTON & RESULTS
# ─────────────────────────────────────────────

if len(tickers) == 0:
    st.warning("⚠️ No assets match the current filters. Adjust the sidebar and try again.")
    st.stop()

run = st.button("🔄 Run Market Scan", type="primary", use_container_width=True)

if run:
    results = []
    progress = st.progress(0, text="Scanning…")

    for i, t in enumerate(tickers):
        res = analyse_ticker(t, risk_multiplier)
        if res:
            results.append(res)
        progress.progress((i + 1) / len(tickers), text=f"Scanning {t}…")

    progress.empty()

    if not results:
        st.error("❌ No data returned for any ticker. Try different filters or check your internet connection.")
        st.stop()

    df = pd.DataFrame(results)

    # Composite score (lower dist% + lower RSI = better buying opportunity)
    df["Score"] = (
        (-df["Dist%"] / 20) * 0.5 +
        ((50 - df["RSI"]) / 50) * 0.3 +
        df["Confidence"] * 0.2
    )
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1

    df["Suggested €"] = df.apply(lambda row: allocation_label(row, baseline, fg_index), axis=1)

    # ── Summary KPIs
    st.markdown("---")
    st.subheader("📊 Scan Results")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Scanned", len(df))
    k2.metric("BUY Signals", int((df["Action"] == "BUY").sum()))
    k3.metric("SELL Signals", int((df["Action"] == "SELL").sum()))
    k4.metric("WAIT Signals", int((df["Action"] == "WAIT").sum()))

    # ── Action filter tabs
    tab_all, tab_buy, tab_sell, tab_wait = st.tabs(["All", "🟢 BUY", "🔴 SELL", "🟡 WAIT"])

    display_cols = ["Rank", "Ticker", "Price", "MA200", "Dist%", "RSI", "Vol%",
                    "Confidence", "Signal", "Knife", "Suggested €"]

    def colour_action(val):
        if "BUY" in str(val):
            return "color: #00c853; font-weight:bold"
        if "SELL" in str(val):
            return "color: #ff1744; font-weight:bold"
        return "color: #ffd600"

    def render_table(data):
        if data.empty:
            st.info("No results in this category.")
            return

        styled = (
            data[display_cols]
            .style
            .applymap(colour_action, subset=["Signal"])
            .format({
                "Price": "{:.2f}",
                "MA200": "{:.2f}",
                "Dist%": "{:+.1f}%",
                "RSI": "{:.1f}",
                "Vol%": "{:.2f}%",
                "Confidence": "{:.2f}",
            })
            .background_gradient(subset=["Dist%"], cmap="RdYlGn")
            .background_gradient(subset=["RSI"], cmap="RdYlGn_r")
        )
        st.dataframe(styled, use_container_width=True, height=500)

        # Links for top 10
        st.markdown("#### 🔗 Quick Links (Top 10)")
        top = data.head(10)
        for _, row in top.iterrows():
            st.markdown(
                f"**{row['Ticker']}** — "
                f"[Yahoo]({row['Yahoo']}) | "
                f"[ETF.com]({row['etf.com']}) | "
                f"[justETF]({row['justETF']})"
            )

    with tab_all:
        render_table(df)

    with tab_buy:
        render_table(df[df["Action"] == "BUY"].reset_index(drop=True))

    with tab_sell:
        render_table(df[df["Action"] == "SELL"].reset_index(drop=True))

    with tab_wait:
        render_table(df[df["Action"] == "WAIT"].reset_index(drop=True))

    # ── CSV download
    st.markdown("---")
    csv = df[display_cols + ["Yahoo", "etf.com", "justETF"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Full Results as CSV",
        data=csv,
        file_name="market_scan_results.csv",
        mime="text/csv",
        use_container_width=True,
    )
