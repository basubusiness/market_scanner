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
    df = df[~df["ticker"].str.startswith("^")]
    df = df[df["ticker"].str.match(r"^[A-Z]{1,5}$")]
    return df

universe = load_universe()

st.sidebar.title(f"Config {APP_VERSION}")
st.sidebar.subheader("Filters")

asset_type = st.sidebar.multiselect("Asset Type", ["ETF", "Stock"], default=["ETF"])
stocks_selected = "Stock" in asset_type
etfs_selected   = "ETF"   in asset_type

country = []
sector  = []

def col_options(df, col):
    """Safe sorted unique non-empty values for a column."""
    if col not in df.columns:
        return []
    return sorted([v for v in df[col].dropna().unique() if str(v).strip()])

# ── ETF filters
etf_category_group = []
etf_category       = []
etf_currency       = []
etf_exchange       = []
etf_family         = []

if etfs_selected:
    eu = universe[universe["type"] == "ETF"]
    st.sidebar.markdown("**📦 ETF Filters**")

    opts = col_options(eu, "category_group")
    if opts:
        etf_category_group = st.sidebar.multiselect(
            "ETF Asset Class (empty = all)",
            opts,
            help="e.g. Equities, Fixed Income, Commodities"
        )

    opts = col_options(eu, "category")
    if opts:
        etf_category = st.sidebar.multiselect(
            "ETF Category (empty = all)",
            opts,
            help="e.g. Large Cap, Emerging Markets, Government Bonds"
        )

    opts = col_options(eu, "currency")
    if opts:
        etf_currency = st.sidebar.multiselect(
            "ETF Currency / Domicile proxy (empty = all)",
            opts,
            help="EUR ≈ UCITS/EU domiciled · USD ≈ US domiciled"
        )

    opts = col_options(eu, "exchange")
    if opts:
        etf_exchange = st.sidebar.multiselect(
            "ETF Exchange (empty = all)",
            opts,
            help="e.g. NYSE Arca, NASDAQ, LSE"
        )

    opts = col_options(eu, "family")
    if opts:
        etf_family = st.sidebar.multiselect(
            "ETF Family (empty = all)",
            opts,
            help="e.g. iShares, Vanguard, Invesco"
        )

# ── Stock filters
country = []
sector  = []
industry = []

if stocks_selected:
    su = universe[universe["type"] == "Stock"]
    st.sidebar.markdown("**📈 Stock Filters**")

    opts = col_options(su, "country")
    if opts:
        country = st.sidebar.multiselect(
            "Country (empty = all)",
            opts,
            default=["United States"] if "United States" in opts else [],
        )

    opts = col_options(su, "sector")
    if opts:
        sector = st.sidebar.multiselect("Sector (empty = all)", opts)

    opts = col_options(su, "industry_group")
    if opts:
        industry = st.sidebar.multiselect("Industry Group (empty = all)", opts)

# ── Build filtered universe
parts = []
if etfs_selected:
    e = universe[universe["type"] == "ETF"].copy()
    if etf_category_group and "category_group" in e.columns:
        e = e[e["category_group"].isin(etf_category_group)]
    if etf_category and "category" in e.columns:
        e = e[e["category"].isin(etf_category)]
    if etf_currency and "currency" in e.columns:
        e = e[e["currency"].isin(etf_currency)]
    if etf_exchange and "exchange" in e.columns:
        e = e[e["exchange"].isin(etf_exchange)]
    if etf_family and "family" in e.columns:
        e = e[e["family"].isin(etf_family)]
    parts.append(e)

if stocks_selected:
    s = universe[universe["type"] == "Stock"].copy()
    if country and "country" in s.columns:
        s = s[s["country"].isin(country)]
    if sector and "sector" in s.columns:
        s = s[s["sector"].isin(sector)]
    if industry and "industry_group" in s.columns:
        s = s[s["industry_group"].isin(industry)]
    parts.append(s)

filtered = pd.concat(parts) if parts else pd.DataFrame()
tickers  = list(dict.fromkeys(filtered["ticker"].tolist()))

MAX_TICKERS = 500
st.sidebar.caption(f"Universe: {len(universe):,} | Filtered: {len(filtered):,} | Scan: {min(len(tickers), MAX_TICKERS)}")
if len(tickers) > MAX_TICKERS:
    st.sidebar.warning(f"Capped at {MAX_TICKERS} — expect a longer scan")
    tickers = tickers[:MAX_TICKERS]

baseline = st.sidebar.number_input("Monthly Investment (EUR)", value=1000, min_value=100, step=100)

with st.sidebar.expander("🔬 Debug (advanced)"):
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
        price = float(close.iloc[-1])
        # Filter out illiquid/dead tickers
        if price < 0.50:
            return None
        # Check average volume if available
        if "Volume" in df.columns:
            avg_vol = df["Volume"].dropna().tail(20).mean()
            if avg_vol < 1000:
                return None
        ma200   = float(close.rolling(200).mean().iloc[-1])
        rsi     = float(calculate_rsi(close).iloc[-1])
        if rsi < 1 or rsi > 99:   # RSI=0 or 100 means bad data
            return None
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
    st.caption("[📈 Yahoo Finance VIX](https://finance.yahoo.com/quote/%5EVIX/)")
with col_fg:
    fg_index = st.slider("Fear & Greed (manual)", 0, 100, 50)
    st.caption("[🧠 CNN Fear & Greed Index](https://edition.cnn.com/markets/fear-and-greed)")

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
    attempted = 0
    prog = st.progress(0, text="Starting...")
    for i, t in enumerate(tickers):
        r = analyse_ticker(t, risk_mult)
        attempted += 1
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
        st.session_state["scan_attempted"] = attempted

if "scan_results" in st.session_state:
    df = st.session_state["scan_results"]
    COLS = ["Rank","Ticker","Price","MA200","Dist%","RSI","Vol%","Confidence","Signal","Knife","Suggested","Yahoo","etf.com","justETF"]

    st.markdown("---")
    st.subheader("Scan Results")
    attempted = st.session_state.get("scan_attempted", len(df))
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Attempted", attempted)
    k2.metric("Valid Data", len(df), delta=f"{len(df)-attempted} filtered out", delta_color="off")
    k3.metric("BUY",  int((df["Action"]=="BUY").sum()))
    k4.metric("SELL", int((df["Action"]=="SELL").sum()))
    k5.metric("WAIT", int((df["Action"]=="WAIT").sum()))

    t_all, t_buy, t_sell, t_wait = st.tabs(["All","BUY","SELL","WAIT"])

    def colour(val):
        if "BUY"  in str(val): return "color:#00c853;font-weight:bold"
        if "SELL" in str(val): return "color:#ff1744;font-weight:bold"
        return "color:#ffd600"

    def show_table(data):
        if data.empty:
            st.info("No results.")
            return
        display_cols = [c for c in COLS if c not in ("Links",)]
        styled = (
            data[display_cols].style
            .applymap(colour, subset=["Signal"])
            .format({"Price":"{:.2f}","MA200":"{:.2f}","Dist%":"{:+.1f}%",
                     "RSI":"{:.1f}","Vol%":"{:.2f}%","Confidence":"{:.2f}"})
            .background_gradient(subset=["Dist%"], cmap="RdYlGn")
            .background_gradient(subset=["RSI"],   cmap="RdYlGn_r")
        )
        st.dataframe(
            styled,
            use_container_width=True,
            height=480,
            column_config={
                "Yahoo":   st.column_config.LinkColumn("Yahoo",   display_text="📈 YF"),
                "etf.com": st.column_config.LinkColumn("ETF.com", display_text="📊 ETF"),
                "justETF": st.column_config.LinkColumn("justETF", display_text="🔍 jETF"),
            }
        )

    with t_all:  show_table(df)
    with t_buy:  show_table(df[df["Action"]=="BUY"].reset_index(drop=True))
    with t_sell: show_table(df[df["Action"]=="SELL"].reset_index(drop=True))
    with t_wait: show_table(df[df["Action"]=="WAIT"].reset_index(drop=True))

    csv = df[COLS].to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="scan_results.csv",
                       mime="text/csv", use_container_width=True)
