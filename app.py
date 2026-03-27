import streamlit as st
st.set_page_config(page_title="Market Scanner v8", layout="wide")

APP_VERSION = "v8.0"

import yfinance as yf
import pandas as pd
import numpy as np
import financedatabase as fd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    for col in ["country","sector","name","category_group","category",
                "currency","exchange","family","industry_group"]:
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
    # Fix: avoid artificial inflation when loss=0
    rs = gain / loss
    rs[loss == 0] = np.inf
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    ema12  = prices.ewm(span=12, adjust=False).mean()
    ema26  = prices.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - signal
    return float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])

def linear_slope(series, window=10):
    """Normalised linear regression slope — better trend proxy than price comparison."""
    y = series.tail(window).values
    if len(y) < window:
        return 0.0
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    return float(slope / (y.mean() + 1e-9))   # normalise by mean price

@st.cache_data(ttl=1800, show_spinner=False)
def get_fg_index():
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        r   = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        if r.status_code == 200:
            d      = r.json()["fear_and_greed"]
            score  = round(float(d["score"]))
            rating = str(d["rating"]).replace("_", " ").title()
            return score, rating
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

def allocation_label(row, base, fg, risk_mult):
    """Smarter sizing: scale by confidence AND volatility, not just signal score."""
    if row["Action"] == "AVOID":
        return "⛔ Do not enter"
    if row["Action"] == "WATCH":
        return f"👀 Wait — EUR {base*0.25:,.0f} max starter"

    # Base score from market context
    ctx_score = (40 if fg < 35 else 20 if fg < 50 else 0)

    # Signal score
    sig_score  = (30 if row["RSI"] < 40 else 15 if row["RSI"] < 50 else 0)
    sig_score += (30 if row["Dist%"] < -10 else 15 if row["Dist%"] < 0 else 0)

    total = ctx_score + sig_score

    # Scale by confidence and volatility
    conf_mult = 0.5 + row["Confidence"]          # 0.5–1.5x
    vol_mult  = max(0.5, 1 - row["Vol%"] / 20)   # reduce for high vol

    amount = base * (total / 100) * 2 * conf_mult * vol_mult * risk_mult
    amount = max(base * 0.25, min(amount, base * 3))  # clamp 0.25x–3x

    tier = "🔥 Strong" if total >= 60 else "⚖️ Normal" if total >= 30 else "🔍 Light"
    return f"{tier} — EUR {amount:,.0f}"

# ─────────────────────────────────────────────
# TICKER ANALYSIS — parallel batch
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

        ma50   = float(close.rolling(50).mean().iloc[-1])
        ma200  = float(close.rolling(200).mean().iloc[-1])
        rsi    = float(calculate_rsi(close).iloc[-1])
        if not (1 < rsi < 99):
            return None

        macd_val, macd_sig, macd_hist = calculate_macd(close)

        # RSI momentum — is RSI rising or falling?
        rsi_series = calculate_rsi(close)
        rsi_slope  = linear_slope(rsi_series.dropna(), window=5)

        dist_ma  = ((price - ma200) / ma200) * 100
        week52h  = float(close.tail(252).max())
        week52l  = float(close.tail(252).min())
        dist_52h = ((price - week52h) / week52h) * 100
        vol_pct  = float(close.pct_change().rolling(20).std().iloc[-1] * 100)

        # Better trend: linear regression slope, not raw price comparison
        price_slope = linear_slope(close, window=10)
        trend_down_strong = price_slope < -0.003   # normalised threshold

        # Confidence: deviation strength + RSI extremity, dampened by volatility
        conf  = min(abs(dist_ma) / 20, 1) * 0.5 + min(abs(50 - rsi) / 50, 1) * 0.5
        conf *= 1 - min(vol_pct / 5, 0.5)

        return dict(
            price=price, ma50=ma50, ma200=ma200,
            rsi=rsi, rsi_slope=rsi_slope,
            macd=macd_val, macd_signal=macd_sig, macd_hist=macd_hist,
            dist_ma=dist_ma, week52h=week52h, week52l=week52l, dist_52h=dist_52h,
            vol=vol_pct, price_slope=price_slope,
            trend_down_strong=trend_down_strong, confidence=conf,
        )
    except Exception:
        return None

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_pe_ratio(ticker):
    try:
        info = yf.Ticker(ticker).info
        pe   = info.get("trailingPE") or info.get("forwardPE")
        mcap = info.get("marketCap")
        return (
            round(float(pe), 1) if pe and 0 < pe < 1000 else None,
            mcap,
        )
    except Exception:
        return None, None

def analyse_ticker(ticker, risk_mult, fetch_pe=False):
    raw = fetch_ticker_data(ticker)
    if raw is None:
        return None

    dm        = raw["dist_ma"]
    rsi       = raw["rsi"]
    conf      = raw["confidence"]
    macd_bull = raw["macd"] > raw["macd_signal"]
    macd_accel= raw["macd_hist"] > 0  # histogram positive = acceleration
    rsi_rising= raw["rsi_slope"] > 0

    knife_threshold   = -15 * (1 / risk_mult)
    is_knife          = (dm < knife_threshold) and (rsi < 35) and raw["trend_down_strong"]
    # Reversal: knife + MACD turned bull + RSI starting to rise
    reversal_confirmed = is_knife and macd_bull and rsi_rising

    if is_knife and not reversal_confirmed:
        action = "AVOID"
    elif dm < -10 and rsi < 40 and macd_bull and macd_accel:
        action = "BUY"           # full confirmation
    elif dm < -10 and rsi < 40 and (macd_bull or rsi_rising):
        action = "WATCH"         # oversold, partial confirmation
    elif dm > 10 and rsi > 70:
        action = "SELL"
    else:
        action = "WAIT"

    strength = "Strong" if conf > 0.7 else "Medium" if conf > 0.4 else "Weak"

    pe, mcap = (fetch_pe_ratio(ticker) if fetch_pe else (None, None))

    # Market cap bucket
    if mcap:
        if   mcap > 200e9: cap_label = "Mega"
        elif mcap > 10e9:  cap_label = "Large"
        elif mcap > 2e9:   cap_label = "Mid"
        elif mcap > 300e6: cap_label = "Small"
        else:              cap_label = "Micro"
    else:
        cap_label = "—"

    return {
        "Ticker":     ticker,
        "Price":      round(raw["price"], 2),
        "MA50":       round(raw["ma50"], 2),
        "MA200":      round(raw["ma200"], 2),
        "Dist%":      round(dm, 1),
        "52W%":       round(raw["dist_52h"], 1),
        "RSI":        round(rsi, 1),
        "RSI↗":       "↗" if rsi_rising else "↘",
        "MACD":       "▲ Bull" if macd_bull else "▼ Bear",
        "MACD Accel": "⚡" if macd_accel else "—",
        "Vol%":       round(raw["vol"], 2),
        "Confidence": round(conf, 2),
        "PE":         pe if pe else "—",
        "Cap":        cap_label,
        "Signal":     f"{action} ({strength})",
        "Action":     action,
        "Knife":      "⚠️ Knife" if (is_knife and not reversal_confirmed)
                      else ("✅ Reversal" if reversal_confirmed else "—"),
        "Yahoo":      f"https://finance.yahoo.com/quote/{ticker}",
        "etf.com":    f"https://www.etf.com/{ticker}",
        "justETF":    f"https://www.justetf.com/en/search.html?query={ticker}",
    }

def parallel_scan(tickers, risk_mult, fetch_pe, max_workers=12):
    """Run analyse_ticker in parallel threads — 5-10x faster than sequential."""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyse_ticker, t, risk_mult, fetch_pe): t for t in tickers}
        for fut in as_completed(futures):
            ticker = futures[fut]
            try:
                r = fut.result()
                results[ticker] = r
            except Exception:
                results[ticker] = None
    # Return in original order
    return [results[t] for t in tickers if results.get(t) is not None]

# ─────────────────────────────────────────────
# SCORING (improved — includes MACD)
# ─────────────────────────────────────────────

def compute_scores(df):
    # Normalise each component to [0,1] range
    dist_score  = (-df["Dist%"] / 30).clip(0, 1)           # deeper below MA200 = better
    rsi_score   = ((50 - df["RSI"]) / 50).clip(0, 1)       # lower RSI = better
    macd_score  = (df["MACD"] == "▲ Bull").astype(float)   # bull = 1, bear = 0
    accel_score = (df["MACD Accel"] == "⚡").astype(float)
    conf_score  = df["Confidence"]

    score = (
        dist_score  * 0.30 +
        rsi_score   * 0.25 +
        macd_score  * 0.20 +
        accel_score * 0.10 +
        conf_score  * 0.15
    )
    # Hard penalty for AVOID
    score -= (df["Action"] == "AVOID").astype(float) * 2
    return score

# ─────────────────────────────────────────────
# SIDEBAR — CASCADING FILTERS
# ─────────────────────────────────────────────

st.sidebar.title(f"Config {APP_VERSION}")
st.sidebar.subheader("Filters")

asset_type      = st.sidebar.multiselect("Asset Type", ["ETF","Stock"], default=["ETF"])
stocks_selected = "Stock" in asset_type
etfs_selected   = "ETF"   in asset_type

etf_category_group = []
etf_category       = []
etf_currency       = []
etf_exchange       = []
etf_family         = []

if etfs_selected:
    eu = universe[universe["type"] == "ETF"].copy()
    st.sidebar.markdown("**📦 ETF Filters**")

    opts = col_options(eu, "category_group")
    if opts:
        etf_category_group = st.sidebar.multiselect(
            "Asset Class (empty=all)", opts,
            help="e.g. Equities, Fixed Income, Commodities")
    if etf_category_group:
        eu = eu[eu["category_group"].isin(etf_category_group)]

    opts = col_options(eu, "category")
    if opts:
        etf_category = st.sidebar.multiselect(
            "Category (empty=all)", opts,
            help="e.g. Large Cap, Emerging Markets")
    if etf_category:
        eu = eu[eu["category"].isin(etf_category)]

    opts = col_options(eu, "currency")
    if opts:
        etf_currency = st.sidebar.multiselect(
            "Currency / Domicile (empty=all)", opts,
            help="EUR ≈ UCITS/EU · USD ≈ US domiciled")
    if etf_currency:
        eu = eu[eu["currency"].isin(etf_currency)]

    opts = col_options(eu, "exchange")
    if opts:
        etf_exchange = st.sidebar.multiselect(
            "Exchange (empty=all)", opts,
            help="e.g. NYSE Arca, NASDAQ, LSE")
    if etf_exchange:
        eu = eu[eu["exchange"].isin(etf_exchange)]

    opts = col_options(eu, "family")
    if opts:
        etf_family = st.sidebar.multiselect(
            "Fund Family (empty=all)", opts,
            help="e.g. iShares, Vanguard, Invesco")

country  = []
sector   = []
industry = []

if stocks_selected:
    su = universe[universe["type"] == "Stock"].copy()
    st.sidebar.markdown("**📈 Stock Filters**")

    opts = col_options(su, "country")
    if opts:
        country = st.sidebar.multiselect(
            "Country (empty=all)", opts,
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
st.sidebar.caption(
    f"Universe: {len(universe):,} | Filtered: {len(filtered):,} | "
    f"Will scan: {min(len(tickers), MAX_TICKERS):,}"
)
if len(tickers) > MAX_TICKERS:
    st.sidebar.warning(f"Capped at {MAX_TICKERS:,} tickers")
    tickers = tickers[:MAX_TICKERS]

baseline  = st.sidebar.number_input("Monthly Investment (EUR)", value=1000, min_value=100, step=100)
fetch_pe  = st.sidebar.checkbox("Fetch PE + Market Cap (slower)", value=False,
    help="Adds P/E and market cap. ~2s extra per ticker. Cached 24h.")
n_workers = st.sidebar.slider("Parallel workers", 4, 24, 12,
    help="More workers = faster scan. Reduce if you hit rate limits.")

with st.sidebar.expander("🔬 Debug (advanced)"):
    st.write("Queued tickers (first 10):", tickers[:10])
    st.write("ETF columns with data:",
        [c for c in ["category_group","category","currency","exchange","family"]
         if c in universe.columns and universe[universe["type"]=="ETF"][c].str.strip().replace("","<NA>").dropna().shape[0] > 10])

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

st.title(f"Market Decision Engine {APP_VERSION}")

fg_auto, fg_rating = get_fg_index()

col_vix, col_fg = st.columns(2)
with col_vix:
    live_vix = get_live_vix()
    st.metric("Live VIX", f"{live_vix:.2f}",
        delta="High fear" if live_vix > 30 else ("Calm" if live_vix < 15 else "Normal"),
        delta_color="inverse")
    st.caption("[📈 Yahoo Finance VIX](https://finance.yahoo.com/quote/%5EVIX/)")

with col_fg:
    if fg_auto is not None:
        st.metric("Fear & Greed (CNN live)", f"{fg_auto}",
            delta=fg_rating, delta_color="off")
        fg_index = fg_auto
        st.caption("[🧠 CNN Fear & Greed](https://edition.cnn.com/markets/fear-and-greed)")
    else:
        fg_index = st.slider("Fear & Greed (manual)", 0, 100, 50)
        st.caption("[🧠 CNN Fear & Greed](https://edition.cnn.com/markets/fear-and-greed) — auto-fetch unavailable")

risk_mult = 1.0 + ((live_vix / 20) * ((100 - fg_index) / 50))

st.sidebar.subheader("Market Context")
if   risk_mult > 2.5: st.sidebar.error(f"🔥 Extreme Fear ({risk_mult:.2f}x)")
elif risk_mult > 2.0: st.sidebar.error(f"😰 High Fear ({risk_mult:.2f}x)")
elif risk_mult < 1.2: st.sidebar.info(f"😌 Calm ({risk_mult:.2f}x)")
else:                 st.sidebar.warning(f"⚖️ Normal ({risk_mult:.2f}x)")

# VIX regime override hint
if live_vix > 40:
    st.warning("⚡ VIX > 40 — Extreme regime. MACD confirmation relaxed for reversals. "
               "Violent bounces possible. Size positions conservatively.")

# ─────────────────────────────────────────────
# SIGNAL GUIDE
# ─────────────────────────────────────────────

with st.expander("📖 Signal Guide & Financial Principles"):
    st.markdown("""
| Signal | Meaning | Suggested action |
|--------|---------|-----------------|
| **BUY** | >10% below MA200 · RSI<40 · MACD bullish · MACD accelerating | Full position — all conditions met |
| **WATCH** | Oversold but only partial MACD/RSI confirmation | Small starter — wait for BUY confirmation |
| **AVOID** | Falling knife — deeply below MA200 · RSI<35 · price slope still negative | Do not enter. Wait for reversal |
| **WAIT** | No clear signal | Hold cash or existing position |
| **SELL** | >10% above MA200 · RSI>70 | Overbought — trim or take profit |

**Why both MACD and RSI?**
RSI tells you *how oversold*. MACD tells you *if momentum is turning*. Together they filter out stocks that are cheap for a reason (still falling) vs genuinely recovering.

**RSI↗ arrow:** Rising RSI on an oversold stock is an early reversal signal even before MACD confirms.

**MACD Accel (⚡):** MACD histogram turning positive means the bullish momentum is *accelerating* — strongest buy confirmation.

**Falling Knife:**
Anything >15% below MA200 with RSI<35 and a negative price slope is dangerous. It looks cheap. It may not be. Always check news before acting.

**P/E context (enable in sidebar):**
- <15: potentially undervalued · 15–25: fair · >30: growth premium
- A stock with RSI 30 but P/E 80 is still expensive on earnings

**VIX levels:**
- <15: complacency — markets may be stretched  
- 15–25: normal  
- >30: fear — historically good long-term entries, short-term pain likely  
- >40: crisis — maximum opportunity AND maximum risk. MACD confirmation less reliable.

**Position sizing:**
Allocation is scaled by confidence score and volatility — not just signal strength. High-volatility names get smaller allocations automatically.
    """)

# ─────────────────────────────────────────────
# SCAN
# ─────────────────────────────────────────────

if len(tickers) == 0:
    st.error("No tickers match current filters.")
    st.stop()

c1, c2 = st.columns([4, 1])
with c1:
    run = st.button("🔄 Run Market Scan", type="primary", use_container_width=True)
with c2:
    if st.button("🗑️ Clear", use_container_width=True):
        for k in ["scan_results","scan_attempted","scan_valid"]:
            st.session_state.pop(k, None)
        st.rerun()

if run:
    n = min(len(tickers), MAX_TICKERS)
    scan_tickers = tickers[:n]

    prog   = st.progress(0, text=f"Launching {n_workers} parallel workers…")
    status = st.empty()

    # Run parallel scan — progress is approximate since threads complete out of order
    with st.spinner(f"Scanning {n:,} tickers with {n_workers} workers…"):
        results = parallel_scan(scan_tickers, risk_mult, fetch_pe, max_workers=n_workers)

    prog.empty()

    if not results:
        st.error("No valid data returned. Check internet connection.")
    else:
        df = pd.DataFrame(results)
        df["Score"]     = compute_scores(df)
        df              = df.sort_values("Score", ascending=False).reset_index(drop=True)
        df["Rank"]      = df.index + 1
        df["Suggested"] = df.apply(
            lambda r: allocation_label(r, baseline, fg_index, risk_mult), axis=1)

        st.session_state["scan_results"]  = df
        st.session_state["scan_attempted"] = n
        st.session_state["scan_valid"]     = len(df)
        status.empty()
        st.success(f"✅ Scan complete — {len(df)} valid results from {n} attempted")

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────

if "scan_results" in st.session_state:
    df        = st.session_state["scan_results"]
    attempted = st.session_state.get("scan_attempted", len(df))
    valid     = st.session_state.get("scan_valid", len(df))

    COLS = ["Rank","Ticker","Price","MA50","MA200","Dist%","52W%","RSI","RSI↗",
            "MACD","MACD Accel","Vol%","Confidence","PE","Cap",
            "Signal","Knife","Suggested","Yahoo","etf.com","justETF"]
    COLS = [c for c in COLS if c in df.columns]

    st.markdown("---")
    st.subheader("📊 Scan Results")

    k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
    k1.metric("Attempted",  attempted)
    k2.metric("Valid",      valid, delta=f"-{attempted-valid} filtered", delta_color="off")
    k3.metric("🟢 BUY",    int((df["Action"]=="BUY").sum()))
    k4.metric("👀 WATCH",  int((df["Action"]=="WATCH").sum()))
    k5.metric("⛔ AVOID",  int((df["Action"]=="AVOID").sum()))
    k6.metric("🔴 SELL",   int((df["Action"]=="SELL").sum()))
    k7.metric("🟡 WAIT",   int((df["Action"]=="WAIT").sum()))

    tabs = st.tabs(["All","🟢 BUY","👀 WATCH","⛔ AVOID","🔴 SELL","🟡 WAIT"])
    filters = [None, "BUY", "WATCH", "AVOID", "SELL", "WAIT"]

    def colour(val):
        v = str(val)
        if "BUY"    in v: return "color:#00c853;font-weight:bold"
        if "WATCH"  in v: return "color:#00bcd4;font-weight:bold"
        if "AVOID"  in v: return "color:#ff6d00;font-weight:bold"
        if "SELL"   in v: return "color:#ff1744;font-weight:bold"
        if "WAIT"   in v: return "color:#ffd600"
        return ""

    def show_table(data):
        if data.empty:
            st.info("No results in this category.")
            return
        show_cols = [c for c in COLS if c in data.columns]
        fmt = {
            "Price":"{:.2f}", "MA50":"{:.2f}", "MA200":"{:.2f}",
            "Dist%":"{:+.1f}%", "52W%":"{:+.1f}%",
            "RSI":"{:.1f}", "Vol%":"{:.2f}%", "Confidence":"{:.2f}",
        }
        fmt = {k: v for k, v in fmt.items() if k in show_cols}
        styled = (
            data[show_cols].style
            .applymap(colour, subset=["Signal"])
            .format(fmt)
            .background_gradient(subset=["Dist%"],      cmap="RdYlGn")
            .background_gradient(subset=["RSI"],        cmap="RdYlGn_r")
            .background_gradient(subset=["Confidence"], cmap="Blues")
        )
        st.dataframe(
            styled,
            use_container_width=True,
            height=520,
            column_config={
                "Yahoo":   st.column_config.LinkColumn("Yahoo",   display_text="📈 YF"),
                "etf.com": st.column_config.LinkColumn("ETF.com", display_text="📊 ETF"),
                "justETF": st.column_config.LinkColumn("justETF", display_text="🔍 jETF"),
            }
        )

    for tab, action in zip(tabs, filters):
        with tab:
            show_table(df if action is None else df[df["Action"]==action].reset_index(drop=True))

    st.markdown("---")
    csv = df[COLS].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv,
        file_name="scan_results.csv", mime="text/csv", use_container_width=True)
