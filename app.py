import streamlit as st
st.set_page_config(page_title="Market Scanner v6", layout="wide")

APP_VERSION = "v6.0"

import yfinance as yf
import pandas as pd
import numpy as np
import financedatabase as fd

@st.cache_data(show_spinner="Loading market universe...")
def load_universe():
    etfs = fd.ETFs().select().copy()
    equities = fd.Equities().select().copy()
    etfs["type"] = "ETF"
    equities["type"] = "Stock"
    df = pd.concat([etfs, equities], axis=0).reset_index()
    df.rename(columns={df.columns[0]: "ticker"}, inplace=True)
    for col in ["country", "sector", "name"]:
        if col not in df.columns:
            df[col] = ""
    df["country"] = df["country"].fillna("").astype(str).str.strip()
    df["sector"]  = df["sector"].fillna("").astype(str).str.strip()
    df["ticker"]  = df["ticker"].fillna("").astype(str).str.strip()
    df = df[df["ticker"].str.len().between(1, 5)]
    return df

universe = load_universe()

st.sidebar.title(f"Config {APP_VERSION}")
st.sidebar.subheader("Filters")

asset_type = st.sidebar.multiselect("Asset Type", ["ETF", "Stock"], default=["ETF"])
stocks_selected = "Stock" in asset_type
etfs_selected   = "ETF"   in asset_type

country = []
sector  = []

if stocks_selected:
    su = universe[universe["type"] == "Stock"]
    ctries = sorted([c for c in su["country"].unique() if c])
    country = st.sidebar.multiselect(
        "Country (Stocks only)",
        ctries,
        default=["United States"] if "United States" in ctries else [],
    )
    sects = sorted([s for s in su["sector"].unique() if s])
    sector = st.sidebar.multiselect("Sector (Stocks only)", sects)
else:
    st.sidebar.info("ETFs: no country/sector metadata - all ETFs included")

parts = []
if etfs_selected:
    parts.append(universe[universe["type"] == "ETF"])
if stocks_selected:
    s = universe[universe["type"] == "Stock"]
    if country:
        s = s[s["country"].isin(country)]
    if sector:
        s = s[s["sector"].isin(sector)]
    parts.append(s)

filtered = pd.concat(parts) if parts else pd.DataFrame()
tickers  = list(dict.fromkeys(filtered["ticker"].tolist()))

MAX_TICKERS = 50
st.sidebar.caption(f"Universe: {len(universe):,} | Filtered: {len(filtered):,} | Scan: {min(len(tickers), MAX_TICKERS)}")
if len(tickers) > MAX_TICKERS:
    tickers = tickers[:MAX_TICKERS]

baseline = st.sidebar.number_input("Monthly Investment (EUR)", value=1000, min_value=100, step=100)

with st.sidebar.expander("Debug - raw data"):
    st.write("ETF country values:")
    st.write(universe[universe["type"]=="ETF"]["country"].value_counts().head(10))
    st.write("Stock country values:")
    st.write(universe[universe["type"]=="Stock"]["country"].value_counts().head(10))
    st.write("Queued tickers:", tickers[:10])

def flatten_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data(ttl=300)
def get_live_vix():
    try:
        d = flatten_df(yf.download("^VIX", period="2d", progress=False, auto_adjust=True))
        if not d.empty and "Close" in d.columns:
            return float(d["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return 20.0

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs    = gain / loss.replace(0, 0.001)
    return 100 - (100 / (1 + rs))

def allocation_label(row, base, fg):
    score = (40 if fg < 35 else 0) + (30 if row["RSI"] < 40 else 0) + (30 if row["Dist%"] < 0 else 0)
    if score >= 70: return f"EUR {base*2:,.0f} (Strong Buy)"
    if score >= 35: return f"EUR {base:,.0f} (Normal)"
    return f"EUR {base*0.5:,.0f} (Cautious)"

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_data(ticker):
    try:
        df = flatten_df(yf.download(ticker, period="1y", progress=False, auto_adjust=True))
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < 30:
            return None
        price   = float(close.iloc[-1])
        ma200   = float(close.rolling(200).mean().iloc[-1])
        rsi     = float(calculate_rsi(close).iloc[-1])
        dist_ma = ((price - ma200) / ma200) * 100
        vol     = float(close.pct_change().rolling(20).std().iloc[-1] * 100)
        conf    = min(abs(dist_ma)/20,1)*0.5 + min(abs(50-rsi)/50,1)*0.5
        conf   *= 1 - min(vol/5, 0.5)
        return dict(price=price, ma200=ma200, rsi=rsi, dist_ma=dist_ma,
                    vol=vol, trend_down=bool(close.iloc[-1] < close.iloc[-5]), confidence=conf)
    except Exception:
        return None

def analyse_ticker(ticker, risk_mult):
    raw = fetch_ticker_data(ticker)
    if raw is None:
        return None
    dm, rsi, conf = raw["dist_ma"], raw["rsi"], raw["confidence"]
    knife  = (dm < -15*(1/risk_mult)) and (rsi < 35) and raw["trend_down"]
    action = "BUY" if (dm < -10 and rsi < 35) else "SELL" if (dm > 10 and rsi > 70) else "WAIT"
    strength = "Strong" if conf > 0.7 else "Medium" if conf > 0.4 else "Weak"
    return {
        "Ticker": ticker, "Price": round(raw["price"],2), "MA200": round(raw["ma200"],2),
        "Dist%": round(dm,1), "RSI": round(rsi,1), "Vol%": round(raw["vol"],2),
        "Confidence": round(conf,2), "Signal": f"{action} ({strength})", "Action": action,
        "Knife": "Falling Knife" if knife else "OK",
        "Yahoo":   f"https://finance.yahoo.com/quote/{ticker}",
        "etf.com": f"https://www.etf.com/{ticker}",
        "justETF": f"https://www.justetf.com/en/search.html?query={ticker}",
    }

st.title(f"Market Decision Engine {APP_VERSION}")

col_vix, col_fg = st.columns(2)
with col_vix:
    live_vix = get_live_vix()
    st.metric("Live VIX", f"{live_vix:.2f}")
with col_fg:
    fg_index = st.slider("Fear & Greed (manual)", 0, 100, 50)

risk_mult = 1.0 + ((live_vix / 20) * ((100 - fg_index) / 50))
st.sidebar.subheader("Market Context")
if   risk_mult > 2:   st.sidebar.error(f"High Fear ({risk_mult:.2f}x)")
elif risk_mult < 1.2: st.sidebar.info(f"Calm ({risk_mult:.2f}x)")
else:                 st.sidebar.warning(f"Normal ({risk_mult:.2f}x)")

if len(tickers) == 0:
    st.error("No tickers found. Open the Debug panel in the sidebar.")
    st.stop()

c1, c2 = st.columns([4, 1])
with c1:
    run = st.button("Run Market Scan", type="primary", use_container_width=True)
with c2:
    if st.button("Clear", use_container_width=True):
        st.session_state.pop("scan_results", None)
        st.rerun()

if run:
    results = []
    prog = st.progress(0, text="Starting...")
    for i, t in enumerate(tickers):
        r = analyse_ticker(t, risk_mult)
        if r:
            results.append(r)
        prog.progress((i+1)/len(tickers), text=f"Scanning {t}...")
    prog.empty()

    if not results:
        st.error("No data returned. Check internet connection.")
    else:
        df = pd.DataFrame(results)
        df["Score"] = (-df["Dist%"]/20)*0.5 + ((50-df["RSI"])/50)*0.3 + df["Confidence"]*0.2
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        df["Rank"] = df.index + 1
        df["Suggested"] = df.apply(lambda r: allocation_label(r, baseline, fg_index), axis=1)
        st.session_state["scan_results"] = df

if "scan_results" in st.session_state:
    df = st.session_state["scan_results"]
    COLS = ["Rank","Ticker","Price","MA200","Dist%","RSI","Vol%","Confidence","Signal","Knife","Suggested"]

    st.markdown("---")
    st.subheader("Scan Results")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Scanned",  len(df))
    k2.metric("BUY",  int((df["Action"]=="BUY").sum()))
    k3.metric("SELL", int((df["Action"]=="SELL").sum()))
    k4.metric("WAIT", int((df["Action"]=="WAIT").sum()))

    t_all, t_buy, t_sell, t_wait = st.tabs(["All","BUY","SELL","WAIT"])

    def colour(val):
        if "BUY"  in str(val): return "color:#00c853;font-weight:bold"
        if "SELL" in str(val): return "color:#ff1744;font-weight:bold"
        return "color:#ffd600"

    def show_table(data):
        if data.empty:
            st.info("No results.")
            return
        styled = (
            data[COLS].style
            .applymap(colour, subset=["Signal"])
            .format({"Price":"{:.2f}","MA200":"{:.2f}","Dist%":"{:+.1f}%",
                     "RSI":"{:.1f}","Vol%":"{:.2f}%","Confidence":"{:.2f}"})
            .background_gradient(subset=["Dist%"], cmap="RdYlGn")
            .background_gradient(subset=["RSI"],   cmap="RdYlGn_r")
        )
        st.dataframe(styled, use_container_width=True, height=480)
        for _, row in data.head(10).iterrows():
            st.markdown(f"**{row['Ticker']}** - [Yahoo]({row['Yahoo']}) | [ETF.com]({row['etf.com']}) | [justETF]({row['justETF']})")

    with t_all:  show_table(df)
    with t_buy:  show_table(df[df["Action"]=="BUY"].reset_index(drop=True))
    with t_sell: show_table(df[df["Action"]=="SELL"].reset_index(drop=True))
    with t_wait: show_table(df[df["Action"]=="WAIT"].reset_index(drop=True))

    csv = df[COLS+["Yahoo","etf.com","justETF"]].to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="scan_results.csv",
                       mime="text/csv", use_container_width=True)
