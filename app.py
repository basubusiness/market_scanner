import streamlit as st
st.set_page_config(page_title="Market Scanner v7", layout="wide")

APP_VERSION = "v7.0"

import yfinance as yf
import pandas as pd
import numpy as np
import financedatabase as fd
import requests

# ─────────────────────────────────────────────
# UNIVERSE
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="Loading market universe...")
def load_universe():
    etfs     = fd.ETFs().select().copy()
    equities = fd.Equities().select().copy()
    etfs["type"]     = "ETF"
    equities["type"] = "Stock"
    df = pd.concat([etfs, equities], axis=0).reset_index()
    df.rename(columns={df.columns[0]: "ticker"}, inplace=True)
    for col in ["country","sector","name","category_group","category","currency","exchange","family","industry_group"]:
        if col not in df.columns:
            df[col] = ""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("").astype(str).str.strip()
    df = df[df["ticker"].str.match(r"^[A-Z]{1,5}$")]
    return df

universe = load_universe()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def col_options(df, col):
    if col not in df.columns:
        return []
    return sorted([v for v in df[col].dropna().unique() if str(v).strip()])

def flatten_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs    = gain / loss.replace(0, 0.001)
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return float(macd.iloc[-1]), float(signal.iloc[-1])

@st.cache_data(ttl=1800, show_spinner=False)
def get_fg_index():
    """Try to fetch Fear & Greed from CNN via their internal API."""
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            score = data["fear_and_greed"]["score"]
            rating = data["fear_and_greed"]["rating"]
            return round(float(score)), str(rating).replace("_", " ").title()
    except Exception:
        pass
    return None, None

@st.cache_data(ttl=300)
def get_live_vix():
    try:
        d = flatten_df(yf.download("^VIX", period="2d", progress=False, auto_adjust=True))
        if not d.empty and "Close" in d.columns:
            return float(d["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return 20.0

def allocation_label(row, base, fg):
    score = (40 if fg < 35 else 0) + (30 if row["RSI"] < 40 else 0) + (30 if row["Dist%"] < 0 else 0)
    if row["Action"] == "AVOID":
        return f"⛔ Hold — Knife risk"
    if score >= 70: return f"EUR {base*2:,.0f} (Strong Buy)"
    if score >= 35: return f"EUR {base:,.0f} (Normal)"
    return f"EUR {base*0.5:,.0f} (Cautious)"

# ─────────────────────────────────────────────
# TICKER ANALYSIS
# ─────────────────────────────────────────────

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
        if price < 0.50:
            return None
        if "Volume" in df.columns:
            avg_vol = df["Volume"].dropna().tail(20).mean()
            if avg_vol < 1000:
                return None
        ma50  = float(close.rolling(50).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
        rsi   = float(calculate_rsi(close).iloc[-1])
        if rsi < 1 or rsi > 99:
            return None
        macd_val, macd_sig = calculate_macd(close)
        dist_ma  = ((price - ma200) / ma200) * 100
        week52h  = float(close.tail(252).max())
        week52l  = float(close.tail(252).min())
        dist_52h = ((price - week52h) / week52h) * 100
        vol_pct  = float(close.pct_change().rolling(20).std().iloc[-1] * 100)
        trend_down = bool(close.iloc[-1] < close.iloc[-5])
        trend_down_strong = bool(close.iloc[-1] < close.iloc[-10] < close.iloc[-20])
        conf = min(abs(dist_ma)/20,1)*0.5 + min(abs(50-rsi)/50,1)*0.5
        conf *= 1 - min(vol_pct/5, 0.5)
        return dict(
            price=price, ma50=ma50, ma200=ma200, rsi=rsi,
            macd=macd_val, macd_signal=macd_sig,
            dist_ma=dist_ma, week52h=week52h, week52l=week52l, dist_52h=dist_52h,
            vol=vol_pct, trend_down=trend_down, trend_down_strong=trend_down_strong,
            confidence=conf
        )
    except Exception:
        return None

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_pe_ratio(ticker):
    """Fetch PE ratio from yfinance info — slow, cached 24h."""
    try:
        info = yf.Ticker(ticker).info
        pe = info.get("trailingPE") or info.get("forwardPE")
        return round(float(pe), 1) if pe and pe > 0 else None
    except Exception:
        return None

def analyse_ticker(ticker, risk_mult, fetch_pe=False):
    raw = fetch_ticker_data(ticker)
    if raw is None:
        return None

    dm   = raw["dist_ma"]
    rsi  = raw["rsi"]
    conf = raw["confidence"]
    macd_bull = raw["macd"] > raw["macd_signal"]
    knife_threshold = -15 * (1 / risk_mult)

    # Falling knife: deeply oversold AND still falling strongly — AVOID
    is_knife = (dm < knife_threshold) and (rsi < 35) and raw["trend_down_strong"]

    # Reversal confirmation: was falling knife but MACD crossed bullish
    reversal_confirmed = is_knife and macd_bull

    if is_knife and not reversal_confirmed:
        action = "AVOID"
    elif dm < -10 and rsi < 40 and (macd_bull or reversal_confirmed):
        action = "BUY"
    elif dm < -10 and rsi < 40:
        action = "WATCH"   # oversold but no MACD confirmation yet
    elif dm > 10 and rsi > 70:
        action = "SELL"
    else:
        action = "WAIT"

    strength = "Strong" if conf > 0.7 else "Medium" if conf > 0.4 else "Weak"

    pe = fetch_pe(ticker) if fetch_pe else None

    return {
        "Ticker":   ticker,
        "Price":    round(raw["price"], 2),
        "MA50":     round(raw["ma50"], 2),
        "MA200":    round(raw["ma200"], 2),
        "Dist%":    round(dm, 1),
        "52W High": round(raw["dist_52h"], 1),
        "RSI":      round(rsi, 1),
        "MACD":     "📈 Bull" if macd_bull else "📉 Bear",
        "Vol%":     round(raw["vol"], 2),
        "Confidence": round(conf, 2),
        "PE":       pe if pe else "—",
        "Signal":   f"{action} ({strength})",
        "Action":   action,
        "Knife":    "⚠️ Knife" if (is_knife and not reversal_confirmed) else ("✅ Confirmed" if reversal_confirmed else "OK"),
        "Yahoo":    f"https://finance.yahoo.com/quote/{ticker}",
        "etf.com":  f"https://www.etf.com/{ticker}",
        "justETF":  f"https://www.justetf.com/en/search.html?query={ticker}",
    }

# ─────────────────────────────────────────────
# SIDEBAR — FILTERS (cascading)
# ─────────────────────────────────────────────

st.sidebar.title(f"Config {APP_VERSION}")
st.sidebar.subheader("Filters")

asset_type     = st.sidebar.multiselect("Asset Type", ["ETF", "Stock"], default=["ETF"])
stocks_selected = "Stock" in asset_type
etfs_selected   = "ETF"   in asset_type

etf_category_group = []
etf_category       = []
etf_currency       = []
etf_exchange       = []
etf_family         = []

if etfs_selected:
    # Start with all ETFs, cascade each filter into the next
    eu = universe[universe["type"] == "ETF"].copy()
    st.sidebar.markdown("**📦 ETF Filters**")

    opts = col_options(eu, "category_group")
    if opts:
        etf_category_group = st.sidebar.multiselect("Asset Class (empty=all)", opts,
            help="e.g. Equities, Fixed Income, Commodities")
    if etf_category_group:
        eu = eu[eu["category_group"].isin(etf_category_group)]

    opts = col_options(eu, "category")
    if opts:
        etf_category = st.sidebar.multiselect("Category (empty=all)", opts,
            help="e.g. Large Cap, Emerging Markets")
    if etf_category:
        eu = eu[eu["category"].isin(etf_category)]

    opts = col_options(eu, "currency")
    if opts:
        etf_currency = st.sidebar.multiselect("Currency / Domicile (empty=all)", opts,
            help="EUR ≈ UCITS/EU · USD ≈ US domiciled")
    if etf_currency:
        eu = eu[eu["currency"].isin(etf_currency)]

    opts = col_options(eu, "exchange")
    if opts:
        etf_exchange = st.sidebar.multiselect("Exchange (empty=all)", opts,
            help="e.g. NYSE Arca, NASDAQ, LSE, Euronext")
    if etf_exchange:
        eu = eu[eu["exchange"].isin(etf_exchange)]

    opts = col_options(eu, "family")
    if opts:
        etf_family = st.sidebar.multiselect("Fund Family (empty=all)", opts,
            help="e.g. iShares, Vanguard, Invesco")

country  = []
sector   = []
industry = []

if stocks_selected:
    # Cascade stock filters too
    su = universe[universe["type"] == "Stock"].copy()
    st.sidebar.markdown("**📈 Stock Filters**")

    opts = col_options(su, "country")
    if opts:
        country = st.sidebar.multiselect("Country (empty=all)", opts,
            default=["United States"] if "United States" in opts else [])
    if country:
        su = su[su["country"].isin(country)]

    opts = col_options(su, "sector")
    if opts:
        sector = st.sidebar.multiselect("Sector (empty=all)", opts)
    if sector:
        su = su[su["sector"].isin(sector)]

    opts = col_options(su, "industry_group")
    if opts:
        industry = st.sidebar.multiselect("Industry Group (empty=all)", opts)

# Build filtered ticker list
parts = []
if etfs_selected:
    e = universe[universe["type"] == "ETF"].copy()
    if etf_category_group: e = e[e["category_group"].isin(etf_category_group)]
    if etf_category:       e = e[e["category"].isin(etf_category)]
    if etf_currency:       e = e[e["currency"].isin(etf_currency)]
    if etf_exchange:       e = e[e["exchange"].isin(etf_exchange)]
    if etf_family:         e = e[e["family"].isin(etf_family)]
    parts.append(e)

if stocks_selected:
    s = universe[universe["type"] == "Stock"].copy()
    if country:  s = s[s["country"].isin(country)]
    if sector:   s = s[s["sector"].isin(sector)]
    if industry: s = s[s["industry_group"].isin(industry)]
    parts.append(s)

filtered = pd.concat(parts) if parts else pd.DataFrame()
tickers  = list(dict.fromkeys(filtered["ticker"].tolist()))

MAX_TICKERS = 1000
st.sidebar.caption(f"Universe: {len(universe):,} | Filtered: {len(filtered):,} | Will scan: {min(len(tickers), MAX_TICKERS):,}")
if len(tickers) > MAX_TICKERS:
    st.sidebar.warning(f"Capped at {MAX_TICKERS:,} tickers")
    tickers = tickers[:MAX_TICKERS]

baseline = st.sidebar.number_input("Monthly Investment (EUR)", value=1000, min_value=100, step=100)
fetch_pe = st.sidebar.checkbox("Fetch PE Ratios (slower scan)", value=False,
    help="Adds P/E ratio to results. Each ticker needs an extra API call — adds ~2s per ticker.")

with st.sidebar.expander("🔬 Debug (advanced)"):
    st.write("ETF columns:", [c for c in universe.columns if universe[universe["type"]=="ETF"][c].astype(str).str.strip().replace("","<NA>").dropna().shape[0] > 100])
    st.write("Queued tickers (first 10):", tickers[:10])

# ─────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────

st.title(f"Market Decision Engine {APP_VERSION}")

# Auto-fetch F&G
fg_auto, fg_rating = get_fg_index()

col_vix, col_fg = st.columns(2)
with col_vix:
    live_vix = get_live_vix()
    st.metric("Live VIX", f"{live_vix:.2f}")
    st.caption("[📈 Yahoo Finance VIX](https://finance.yahoo.com/quote/%5EVIX/)")

with col_fg:
    if fg_auto is not None:
        st.metric("Fear & Greed (CNN live)", f"{fg_auto} — {fg_rating}")
        fg_index = fg_auto
        st.caption("[🧠 CNN Fear & Greed Index](https://edition.cnn.com/markets/fear-and-greed)")
    else:
        fg_index = st.slider("Fear & Greed (manual — CNN unavailable)", 0, 100, 50)
        st.caption("[🧠 CNN Fear & Greed Index](https://edition.cnn.com/markets/fear-and-greed)")

risk_mult = 1.0 + ((live_vix / 20) * ((100 - fg_index) / 50))
st.sidebar.subheader("Market Context")
if   risk_mult > 2:   st.sidebar.error(f"🔥 High Fear ({risk_mult:.2f}x)")
elif risk_mult < 1.2: st.sidebar.info(f"😌 Calm ({risk_mult:.2f}x)")
else:                 st.sidebar.warning(f"⚖️ Normal ({risk_mult:.2f}x)")

# ─────────────────────────────────────────────
# SIGNAL LEGEND
# ─────────────────────────────────────────────

with st.expander("📖 Signal Guide & Financial Principles"):
    st.markdown("""
| Signal | Meaning | When to act |
|--------|---------|-------------|
| **BUY** | Price >10% below MA200, RSI<40, MACD bullish crossover confirmed | Consider buying — oversold with momentum turning |
| **WATCH** | Oversold (like BUY) but MACD not yet confirmed bullish | Wait for MACD confirmation before entering |
| **AVOID** | Falling knife — deeply below MA200, RSI<35, price still trending down strongly | Do NOT buy. Wait for trend reversal confirmation |
| **WAIT** | No clear signal in either direction | Hold cash or existing position |
| **SELL** | Price >10% above MA200, RSI>70 | Overbought — consider trimming or taking profit |

**Why MACD matters for BUY signals:**
RSI alone tells you something is oversold, but not *when* it will recover. MACD crossing above its signal line is a momentum confirmation that selling pressure is easing. Without it, you may catch a falling knife.

**Falling Knife risk:**
A stock down 30-50% from its MA200 with RSI<35 and still falling is one of the most dangerous setups. It *looks* cheap but may have fundamental reasons (earnings collapse, fraud, sector disruption). Always check the news before acting on any AVOID signal.

**P/E Ratio context (enable in sidebar):**
- A stock with RSI 30 but P/E of 80 is still expensive relative to earnings.
- P/E < 15: potentially undervalued · P/E 15–25: fair · P/E > 30: growth premium, higher risk
- ETFs typically show N/A for P/E — use RSI/MA200 signals instead.

**VIX interpretation:**
- VIX < 15: complacency — markets may be overextended
- VIX 15–25: normal volatility
- VIX > 30: high fear — historically good long-term entry points but short-term pain likely
- VIX > 40: extreme fear / crisis — maximum opportunity AND maximum risk
    """)

# ─────────────────────────────────────────────
# SCAN
# ─────────────────────────────────────────────

if len(tickers) == 0:
    st.error("No tickers found. Adjust filters.")
    st.stop()

c1, c2 = st.columns([4, 1])
with c1:
    run = st.button("🔄 Run Market Scan", type="primary", use_container_width=True)
with c2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.pop("scan_results", None)
        st.session_state.pop("scan_attempted", None)
        st.rerun()

if run:
    results  = []
    valid    = 0
    target   = min(len(tickers), MAX_TICKERS)
    prog     = st.progress(0, text="Starting...")
    status   = st.empty()

    for i, t in enumerate(tickers):
        r = analyse_ticker(t, risk_mult, fetch_pe=fetch_pe)
        if r:
            results.append(r)
            valid += 1
        pct = (i + 1) / target
        prog.progress(min(pct, 1.0), text=f"Scanning {t}… ({valid} valid so far)")
        if (i + 1) % 50 == 0:
            status.caption(f"✅ {valid} valid | ❌ {i+1-valid} skipped | 📋 {i+1}/{target} attempted")

    prog.empty()
    status.empty()

    if not results:
        st.error("No valid data returned.")
    else:
        df = pd.DataFrame(results)
        df["Score"] = (-df["Dist%"]/20)*0.5 + ((50-df["RSI"])/50)*0.3 + df["Confidence"]*0.2
        # Penalise AVOID in scoring
        df.loc[df["Action"] == "AVOID", "Score"] -= 2
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        df["Rank"] = df.index + 1
        df["Suggested"] = df.apply(lambda r: allocation_label(r, baseline, fg_index), axis=1)
        st.session_state["scan_results"]  = df
        st.session_state["scan_attempted"] = len(tickers)

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────

if "scan_results" in st.session_state:
    df       = st.session_state["scan_results"]
    attempted = st.session_state.get("scan_attempted", len(df))

    COLS = ["Rank","Ticker","Price","MA50","MA200","Dist%","52W High","RSI","MACD",
            "Vol%","Confidence","PE","Signal","Knife","Suggested","Yahoo","etf.com","justETF"]
    # Only keep cols that exist
    COLS = [c for c in COLS if c in df.columns]

    st.markdown("---")
    st.subheader("📊 Scan Results")

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Attempted",  attempted)
    k2.metric("Valid",      len(df))
    k3.metric("🟢 BUY",    int((df["Action"]=="BUY").sum()))
    k4.metric("👀 WATCH",  int((df["Action"]=="WATCH").sum()))
    k5.metric("⛔ AVOID",  int((df["Action"]=="AVOID").sum()))
    k6.metric("🔴 SELL",   int((df["Action"]=="SELL").sum()))

    t_all, t_buy, t_watch, t_avoid, t_sell, t_wait = st.tabs([
        "All","🟢 BUY","👀 WATCH","⛔ AVOID","🔴 SELL","🟡 WAIT"])

    def colour(val):
        v = str(val)
        if "BUY"   in v: return "color:#00c853;font-weight:bold"
        if "WATCH" in v: return "color:#00bcd4;font-weight:bold"
        if "AVOID" in v: return "color:#ff6d00;font-weight:bold"
        if "SELL"  in v: return "color:#ff1744;font-weight:bold"
        return "color:#ffd600"

    def show_table(data):
        if data.empty:
            st.info("No results in this category.")
            return
        show_cols = [c for c in COLS if c in data.columns]
        fmt = {"Price":"{:.2f}","MA50":"{:.2f}","MA200":"{:.2f}",
               "Dist%":"{:+.1f}%","52W High":"{:+.1f}%",
               "RSI":"{:.1f}","Vol%":"{:.2f}%","Confidence":"{:.2f}"}
        fmt = {k:v for k,v in fmt.items() if k in show_cols}
        styled = (
            data[show_cols].style
            .applymap(colour, subset=["Signal"])
            .format(fmt)
            .background_gradient(subset=["Dist%"], cmap="RdYlGn")
            .background_gradient(subset=["RSI"],   cmap="RdYlGn_r")
        )
        st.dataframe(
            styled,
            use_container_width=True,
            height=500,
            column_config={
                "Yahoo":   st.column_config.LinkColumn("Yahoo",   display_text="📈 YF"),
                "etf.com": st.column_config.LinkColumn("ETF.com", display_text="📊 ETF"),
                "justETF": st.column_config.LinkColumn("justETF", display_text="🔍 jETF"),
            }
        )

    with t_all:   show_table(df)
    with t_buy:   show_table(df[df["Action"]=="BUY"].reset_index(drop=True))
    with t_watch: show_table(df[df["Action"]=="WATCH"].reset_index(drop=True))
    with t_avoid: show_table(df[df["Action"]=="AVOID"].reset_index(drop=True))
    with t_sell:  show_table(df[df["Action"]=="SELL"].reset_index(drop=True))
    with t_wait:  show_table(df[df["Action"]=="WAIT"].reset_index(drop=True))

    st.markdown("---")
    csv = df[COLS].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv, file_name="scan_results.csv",
                       mime="text/csv", use_container_width=True)
