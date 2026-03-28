import streamlit as st
st.set_page_config(page_title="Market Scanner v8", layout="wide")

APP_VERSION = "v8.1"

import yfinance as yf
import pandas as pd
import numpy as np
import financedatabase as fd
import requests
try:
    import justetf_scraping
    JUSTETF_AVAILABLE = True
except ImportError:
    JUSTETF_AVAILABLE = False
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
    df = df.drop_duplicates(subset=["ticker", "type"], keep="first")
    return df

@st.cache_data(show_spinner="Loading justETF data (3400+ ETFs)...", ttl=86400*7)
def load_justetf_universe():
    """
    Fetch full ETF universe from justETF via justetf-scraping package.
    Returns a DataFrame indexed by ticker with rich metadata:
    domicile_country, ter, distribution_policy, fund_size_eur,
    replication, strategy, isin, name, currency.
    Cached for 7 days — ETF metadata rarely changes.
    """
    if not JUSTETF_AVAILABLE:
        return pd.DataFrame()
    try:
        df = justetf_scraping.load_overview()   # ~3 HTTP requests, ~3400 ETFs
        df = df.reset_index()                   # isin becomes a column
        # Rename to match our schema
        rename = {
            "ticker":              "ticker",
            "isin":                "jETF_isin",
            "name":                "jETF_name",
            "domicile_country":    "domicile",
            "ter":                 "ter",
            "distribution_policy": "dist_policy",
            "fund_size_eur":       "fund_size_eur",
            "replication":         "replication",
            "strategy":            "strategy",
            "currency":            "jETF_currency",
            "index_name":          "index_name",
        }
        keep = {k: v for k, v in rename.items() if k in df.columns}
        df = df[list(keep.keys())].rename(columns=keep)
        df["ticker"] = df["ticker"].fillna("").astype(str).str.strip().str.upper()
        df = df[df["ticker"].str.match(r"^[A-Z]{1,5}$")]
        df = df.drop_duplicates(subset=["ticker"], keep="first")
        return df
    except Exception as e:
        return pd.DataFrame()

universe = load_universe()

# ─────────────────────────────────────────────
# justETF ENRICHMENT — loaded on demand, cached 7 days
# ─────────────────────────────────────────────

@st.cache_data(ttl=86400*7, show_spinner=False)
def get_justetf_data():
    return load_justetf_universe()

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
    """
    4-layer F&G fetch:
    1. CNN dataviz API (no date suffix)
    2. CNN dataviz API (with today's date suffix — different endpoint behaviour)
    3. HTML scrape of money.cnn.com — parse "Greed Now:" text pattern
    4. alternative.me free API — independent source, same 0-100 scale
    """
    import re, datetime

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/html, */*",
        "Referer": "https://www.cnn.com/",
    }

    def rating_from_score(score):
        if score <= 25:  return "Extreme Fear"
        if score <= 44:  return "Fear"
        if score <= 55:  return "Neutral"
        if score <= 75:  return "Greed"
        return "Extreme Greed"

    # Layer 1 & 2: CNN dataviz API variants
    today = datetime.date.today().strftime("%Y-%m-%d")
    for url in [
        "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
        f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{today}",
        "https://production.dataviz.cnn.io/index/feargreed/graphdata",
    ]:
        try:
            r = requests.get(url, headers=headers, timeout=6)
            if r.status_code == 200:
                data = r.json()
                fg   = data.get("fear_and_greed") or {}
                if fg and "score" in fg:
                    score  = round(float(fg["score"]))
                    rating = str(fg.get("rating", "")).replace("_", " ").title() or rating_from_score(score)
                    return score, f"{rating} (CNN API)"
                # Also try historical endpoint structure
                hist = data.get("fear_and_greed_historical", {}).get("data", [])
                if hist:
                    latest = sorted(hist, key=lambda x: x.get("x", 0))[-1]
                    score  = round(float(latest.get("y", 50)))
                    return score, f"{rating_from_score(score)} (CNN historical)"
        except Exception:
            continue

    # Layer 3: HTML scrape of money.cnn.com
    try:
        r = requests.get("https://money.cnn.com/data/fear-and-greed/",
                         headers=headers, timeout=8)
        if r.status_code == 200:
            html = r.text
            # Pattern: "Greed Now: 42" in page source
            m = re.search(r"Greed Now:\s*(\d+(?:\.\d+)?)", html)
            if m:
                score = round(float(m.group(1)))
                return score, f"{rating_from_score(score)} (CNN page)"
            # Also try JSON embedded in page script tags
            m = re.search(r'"score"\s*:\s*(\d+(?:\.\d+)?)', html)
            if m:
                score = round(float(m.group(1)))
                return score, f"{rating_from_score(score)} (CNN embedded)"
    except Exception:
        pass

    # Layer 4: alternative.me — free, independent, same 0-100 scale
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1",
                         headers=headers, timeout=6)
        if r.status_code == 200:
            data  = r.json()
            entry = data.get("data", [{}])[0]
            score = round(float(entry.get("value", 50)))
            label = entry.get("value_classification", "").replace("_", " ").title()
            return score, f"{label or rating_from_score(score)} (alternative.me)"
    except Exception:
        pass

    return None, None

@st.cache_data(ttl=300)
def get_live_vix():
    try:
        d = flatten_df(yf.Ticker("^VIX").history(period="5d", auto_adjust=True))
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
        # Use Ticker.history(), NOT yf.download() — download() uses a shared
        # global dict (_DFS) causing data collisions in parallel threads
        df = flatten_df(yf.Ticker(ticker).history(period="1y", auto_adjust=True))
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
        info = yf.Ticker(ticker).fast_info
        # fast_info is more reliable and faster than .info
        mcap = getattr(info, "market_cap", None)
        # PE not in fast_info — fall through to full info
    except Exception:
        mcap = None

    pe = None
    try:
        info_full = yf.Ticker(ticker).info
        # Try multiple PE fields in order of preference
        for field in ["trailingPE", "forwardPE", "pegRatio"]:
            val = info_full.get(field)
            if val and isinstance(val, (int, float)) and 0 < float(val) < 2000:
                pe = round(float(val), 1)
                break
        # Get market cap from full info if fast_info failed
        if mcap is None:
            mcap = info_full.get("marketCap")
        # Also grab extra useful fields
        eps          = info_full.get("trailingEps")
        revenue_gr   = info_full.get("revenueGrowth")
        profit_mar   = info_full.get("profitMargins")
        div_yield    = info_full.get("dividendYield")
        beta         = info_full.get("beta")
        return {
            "pe":         pe,
            "mcap":       mcap,
            "eps":        round(float(eps), 2)        if eps        else None,
            "rev_growth": f"{revenue_gr*100:.1f}%"   if revenue_gr else None,
            "margin":     f"{profit_mar*100:.1f}%"   if profit_mar else None,
            "div_yield":  f"{div_yield*100:.2f}%"    if div_yield  else None,
            "beta":       round(float(beta), 2)       if beta       else None,
        }
    except Exception:
        return {"pe": pe, "mcap": mcap, "eps": None, "rev_growth": None,
                "margin": None, "div_yield": None, "beta": None}

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

    fund = fetch_pe_ratio(ticker) if fetch_pe else {}

    pe        = fund.get("pe")
    mcap      = fund.get("mcap")
    beta      = fund.get("beta")
    div_yield = fund.get("div_yield")
    margin    = fund.get("margin")
    rev_gr    = fund.get("rev_growth")

    # Market cap bucket
    if mcap:
        if   mcap > 200e9: cap_label = "Mega"
        elif mcap > 10e9:  cap_label = "Large"
        elif mcap > 2e9:   cap_label = "Mid"
        elif mcap > 300e6: cap_label = "Small"
        else:              cap_label = "Micro"
    else:
        cap_label = "—"

    def fmt(v): return v if v is not None else "—"

    return {
        "Ticker":     ticker,
        "Price":      round(raw["price"], 2),
        # justETF fields populated post-scan via merge (placeholders here)
        "Domicile":   "—",
        "TER":        "—",
        "Fund Size":  "—",
        "Dist Policy":"—",
        "Replication":"—",
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
        "PE":         fmt(pe),
        "Beta":       fmt(beta),
        "Div Yield":  fmt(div_yield),
        "Margin":     fmt(margin),
        "Rev Growth": fmt(rev_gr),
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

# ── justETF enrichment toggle ──────────────────
if JUSTETF_AVAILABLE:
    use_justetf = st.sidebar.toggle(
        "🇪🇺 Enrich ETFs with justETF data",
        value=False,
        help="Loads 3400+ ETFs from justETF with domicile, TER, fund size, "
             "replication method, distribution policy. Cached for 7 days. "
             "Adds ~5s on first load, instant thereafter."
    )
    if use_justetf:
        with st.sidebar:
            with st.spinner("Loading justETF data..."):
                jetf_df = get_justetf_data()
        if jetf_df.empty:
            st.sidebar.warning("⚠️ justETF fetch failed — using base universe only.")
            use_justetf = False
        else:
            st.sidebar.success(f"✅ justETF: {len(jetf_df):,} ETFs loaded")
    else:
        jetf_df = pd.DataFrame()
else:
    use_justetf = False
    jetf_df     = pd.DataFrame()
    st.sidebar.info("ℹ️ Install `justetf-scraping` for ETF enrichment.")

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

# ── justETF extra filters (shown only when enrichment is on)
jetf_domicile   = []
jetf_dist       = []
jetf_replication= []
jetf_min_size   = 0

if etfs_selected and use_justetf and not jetf_df.empty:
    st.sidebar.markdown("**🇪🇺 justETF Filters**")

    def je_opts(col):
        return sorted([v for v in jetf_df[col].dropna().unique() if str(v).strip()]) if col in jetf_df.columns else []

    opts = je_opts("domicile")
    if opts:
        jetf_domicile = st.sidebar.multiselect(
            "Domicile Country (empty=all)", opts,
            help="Ireland / Luxembourg = UCITS. USA = US-domiciled.")

    opts = je_opts("dist_policy")
    if opts:
        jetf_dist = st.sidebar.multiselect(
            "Distribution Policy (empty=all)", opts,
            help="Accumulating or Distributing")

    opts = je_opts("replication")
    if opts:
        jetf_replication = st.sidebar.multiselect(
            "Replication Method (empty=all)", opts,
            help="Physical Full / Physical Sampling / Swap-based")

    if "fund_size_eur" in jetf_df.columns:
        jetf_min_size = st.sidebar.number_input(
            "Min Fund Size (EUR m)", value=0, min_value=0, step=50,
            help="Filter out tiny/illiquid ETFs. 100m+ = reasonable liquidity.")

# Build filtered ticker list
parts = []
if etfs_selected:
    e = universe[universe["type"] == "ETF"].copy()
    # financedatabase filters
    if etf_category_group: e = e[e["category_group"].isin(etf_category_group)]
    if etf_category:       e = e[e["category"].isin(etf_category)]
    if etf_currency:       e = e[e["currency"].isin(etf_currency)]
    if etf_exchange:       e = e[e["exchange"].isin(etf_exchange)]
    if etf_family:         e = e[e["family"].isin(etf_family)]

    # justETF filters — merge then filter
    if use_justetf and not jetf_df.empty:
        e = e.merge(jetf_df, on="ticker", how="left")
        if jetf_domicile:
            e = e[e["domicile"].isin(jetf_domicile)]
        if jetf_dist:
            e = e[e["dist_policy"].isin(jetf_dist)]
        if jetf_replication:
            e = e[e["replication"].isin(jetf_replication)]
        if jetf_min_size > 0 and "fund_size_eur" in e.columns:
            e = e[e["fund_size_eur"].fillna(0) >= jetf_min_size]

    parts.append(e)

if stocks_selected:
    s = universe[universe["type"] == "Stock"].copy()
    if country:  s = s[s["country"].isin(country)]
    if sector:   s = s[s["sector"].isin(sector)]
    if industry: s = s[s["industry_group"].isin(industry)]
    parts.append(s)

filtered = pd.concat(parts) if parts else pd.DataFrame(columns=["ticker"])
tickers  = (
    list(dict.fromkeys(filtered["ticker"].str.upper().str.strip().tolist()))
    if not filtered.empty and "ticker" in filtered.columns
    else []
)

MAX_TICKERS = 1000
st.sidebar.caption(
    f"Universe: {len(universe):,} | Filtered: {len(filtered):,} | "
    f"Will scan: {min(len(tickers), MAX_TICKERS):,}"
)
if not asset_type:
    st.sidebar.warning("⚠️ Select at least one Asset Type to scan.")
elif len(tickers) == 0:
    st.sidebar.warning("⚠️ No tickers match the current filters.")
elif len(tickers) > MAX_TICKERS:
    st.sidebar.warning(f"Capped at {MAX_TICKERS:,} tickers")
    tickers = tickers[:MAX_TICKERS]

baseline  = st.sidebar.number_input("Monthly Investment (EUR)", value=1000, min_value=100, step=100)
fetch_pe  = st.sidebar.checkbox("Fetch PE + Market Cap (slower)", value=False,
    help="Adds P/E, Beta, Dividend Yield, Margin, Revenue Growth. ~2s extra per ticker. Cached 24h. Check this BEFORE running scan.")
n_workers = st.sidebar.slider("Parallel workers", 2, 16, 6,
    help="More workers = faster scan but increases yfinance data collision risk. "
         "Keep at 6 unless scanning ETFs (which are more stable). "
         "Reduce to 2-3 if you see many identical rows in results.")

with st.sidebar.expander("🔬 Debug (advanced)"):
    st.write("Queued tickers (first 10):", tickers[:10])
    st.write("ETF columns with data:",
        [c for c in ["category_group","category","currency","exchange","family"]
         if c in universe.columns and universe[universe["type"]=="ETF"][c].str.strip().replace("","<NA>").dropna().shape[0] > 10])

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

st.title(f"Market Decision Engine {APP_VERSION}")

def run_deep_dive():
        st.markdown("---")
        st.header("🔬 Deep Dive — Individual Asset Analysis")
        st.caption("Search any ticker or ISIN for a full breakdown with entry/exit timing.")

        # ── Search UI
        col_in, col_base = st.columns([3, 1])
        with col_in:
            user_input = st.text_input("Ticker or ISIN", value="", placeholder="e.g. VOO or IE00B3XXRP09").strip().upper()
        with col_base:
            dd_baseline = st.number_input("Monthly budget (EUR)", value=1000, min_value=100, step=100, key="dd_baseline")

        if not user_input:
            st.info("Enter a ticker symbol (e.g. SPY, VWRA) or a full ISIN (12 characters).")
            return

        # ── Resolve ticker from ISIN or search
        def is_isin(x):
            return len(x) == 12 and x[:2].isalpha() and x[2:].isalnum()

        ticker  = None
        isin    = None

        try:
            if is_isin(user_input):
                isin = user_input
                search = yf.Search(user_input, max_results=10)
            else:
                search = yf.Search(user_input, max_results=20)

            if hasattr(search, "quotes") and search.quotes:
                options = {
                    f"{r['symbol']}  —  {r.get('longname', r.get('shortname', ''))}  [{r.get('exchDisp','')}]": r["symbol"]
                    for r in search.quotes if "symbol" in r
                }
                if options:
                    chosen = st.selectbox("Select asset", list(options.keys()), key="dd_select")
                    ticker = options[chosen]
                    # Try to grab ISIN from result
                    chosen_raw = [r for r in search.quotes if r.get("symbol") == ticker]
                    if chosen_raw:
                        isin = isin or chosen_raw[0].get("isin")
                else:
                    ticker = user_input
            else:
                ticker = user_input
        except Exception:
            ticker = user_input

        if not ticker:
            st.error("Could not resolve ticker.")
            return

        if not st.button("🔍 Analyse", type="primary", key="dd_run"):
            return

        # ── Download data
        with st.spinner(f"Fetching data for {ticker}…"):
            try:
                raw_df = flatten_df(yf.Ticker(ticker).history(period="2y", auto_adjust=True))
            except Exception as e:
                st.error(f"Download failed: {e}")
                return

        if raw_df is None or raw_df.empty or "Close" not in raw_df.columns:
            st.error("No usable price data found. Try a different ticker or exchange listing.")
            return

        close = raw_df["Close"].dropna()

        if close.empty:
            st.error("Price series is empty after cleaning.")
            return
        if close.nunique() < 5:
            st.warning("⚠️ Price has barely moved — signals may be unreliable.")
        if close.isna().sum() > len(close) * 0.2:
            st.warning("⚠️ >20% missing data — indicators may be unreliable.")

        # ── Core indicators
        cur_p  = float(close.iloc[-1])
        ma50   = close.rolling(50).mean()
        ma200  = close.rolling(200).mean()

        # RSI (fixed formula)
        delta_p = close.diff()
        gain    = delta_p.where(delta_p > 0, 0.0).rolling(14).mean()
        loss    = (-delta_p.where(delta_p < 0, 0.0)).rolling(14).mean()
        rs      = gain / loss
        rs[loss == 0] = np.inf
        rsi_series = 100 - (100 / (1 + rs))
        rsi_val  = float(rsi_series.iloc[-1])
        rsi_prev = float(rsi_series.iloc[-2])
        rsi_rising = rsi_val > rsi_prev

        # MACD
        ema12     = close.ewm(span=12, adjust=False).mean()
        ema26     = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_sig
        macd_bull = float(macd_line.iloc[-1]) > float(macd_sig.iloc[-1])
        macd_accel= float(macd_hist.iloc[-1]) > 0

        # Trend
        ma200_slope = ((float(ma200.iloc[-1]) - float(ma200.iloc[-20])) / float(ma200.iloc[-20])) * 100
        trend_weak  = ma200_slope <= 0
        dist_ma200  = ((cur_p - float(ma200.iloc[-1])) / float(ma200.iloc[-1])) * 100

        # Volatility & trigger
        volatility        = float(close.pct_change().rolling(20).std().iloc[-1] * 100)
        trigger_threshold = max(1.0, volatility * 1.5)
        price_change      = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)

        # 52-week levels
        week52h   = float(close.tail(252).max())
        week52l   = float(close.tail(252).min())
        dist_52h  = ((cur_p - week52h) / week52h) * 100
        dist_52l  = ((cur_p - week52l) / week52l) * 100

        # VIX
        live_vix_dd, _ = get_live_vix(), None
        fg_auto_dd, fg_rating_dd = get_fg_index()
        fg_val = fg_auto_dd if fg_auto_dd is not None else 50

        # Entry state
        if rsi_val < 35 and trend_weak:
            entry_state = "WAIT"
        elif price_change > trigger_threshold and rsi_rising:
            entry_state = "TRIGGER"
        else:
            entry_state = "WATCH"

        # Exit state
        price_drop  = -price_change
        rsi_falling = not rsi_rising
        trend_strong = ma200_slope > 0
        if rsi_val > 65 and trend_strong:
            exit_state = "WAIT"
        elif price_drop > trigger_threshold and rsi_falling:
            exit_state = "TRIGGER"
        else:
            exit_state = "WATCH"

        # Scores
        buy_score  = (40 if fg_val < 35 else 0) + (30 if rsi_val < 40 else 0) + (30 if cur_p < float(ma200.iloc[-1]) else 0)
        sell_score = (40 if fg_val > 65 else 0) + (30 if rsi_val > 65 else 0) + (30 if cur_p > float(ma200.iloc[-1]) else 0)

        # Knife check
        price_slope_dd = linear_slope(close, window=10)
        is_knife_dd    = (dist_ma200 < -15) and (rsi_val < 35) and (price_slope_dd < -0.003)
        reversal_dd    = is_knife_dd and macd_bull and rsi_rising

        # ── Header row
        st.markdown("---")
        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric("Price",     f"{cur_p:.2f}")
        h2.metric("MA200",     f"{float(ma200.iloc[-1]):.2f}", delta=f"{dist_ma200:+.1f}%", delta_color="normal")
        h3.metric("RSI",       f"{rsi_val:.1f}", delta="↗ Rising" if rsi_rising else "↘ Falling", delta_color="off")
        h4.metric("MACD",      "▲ Bull" if macd_bull else "▼ Bear")
        h5.metric("52W High",  f"{dist_52h:+.1f}%")

        # Links
        link_col1, link_col2, link_col3 = st.columns(3)
        with link_col1:
            st.markdown(f"[📈 Yahoo Finance](https://finance.yahoo.com/quote/{ticker})")
        with link_col2:
            st.markdown(f"[📊 ETF.com](https://www.etf.com/{ticker})")
        with link_col3:
            if isin:
                st.markdown(f"[🔍 justETF (ISIN)](https://www.justetf.com/en/etf-profile.html?isin={isin})")
                st.caption(f"ISIN: `{isin}`")
            else:
                st.markdown(f"[🔍 justETF](https://www.justetf.com/en/search.html?query={ticker})")

        # ── Decision boxes
        st.markdown("### 🎯 Decision")
        dcol1, dcol2 = st.columns(2)

        with dcol1:
            st.markdown("#### 📥 Buy")
            if is_knife_dd and not reversal_dd:
                st.error("⛔ AVOID — Falling knife. Wait for reversal confirmation.")
            elif entry_state == "WAIT":
                st.warning("⏳ WAIT — Market still falling. Let it stabilize.")
            elif entry_state == "WATCH":
                if buy_score >= 70:
                    st.info(f"⚖️ PREPARE — Good setup, wait for confirmation (~EUR {dd_baseline:,.0f})")
                else:
                    st.info("🔍 WATCH — No strong entry yet.")
            else:  # TRIGGER
                if buy_score >= 70:
                    st.success(f"🔥 AGGRESSIVE BUY — EUR {dd_baseline*2:,.0f}")
                elif buy_score >= 40:
                    st.success(f"⚖️ STEADY BUY — EUR {dd_baseline:,.0f}")
                else:
                    st.warning(f"⚠️ LIGHT BUY — EUR {dd_baseline*0.5:,.0f}")

            if reversal_dd:
                st.success("✅ Reversal confirmed (was knife — MACD + RSI turned bullish)")

        with dcol2:
            st.markdown("#### 📤 Sell / Exit")
            if exit_state == "WAIT":
                st.info("🟡 HOLD — Momentum still strong.")
            elif exit_state == "WATCH":
                st.warning("🔵 WATCH — No confirmed downtrend yet.")
            else:
                st.error("🔴 SELL TRIGGER — Downtrend starting.")

            if sell_score >= 70:
                st.error("🚨 STRONG SELL — Reduce exposure significantly.")
            elif sell_score >= 40:
                st.warning("⚠️ PARTIAL SELL — Trim positions.")
            else:
                st.success("🟢 HOLD — No strong sell signal.")

        # ── Detailed signals
        st.markdown("### 🧠 Market Signals")
        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.metric("Fear & Greed", fg_auto_dd if fg_auto_dd else "Manual",
                      delta=fg_rating_dd if fg_rating_dd else "")
            with st.expander("What this means"):
                st.write("0–25 Extreme Fear (best opportunities)\n\n25–50 Fear\n\n50–75 Greed\n\n75–100 Extreme Greed (overvalued)")

        with s2:
            vix_dd = get_live_vix()
            st.metric("VIX", f"{vix_dd:.2f}")
            with st.expander("What this means"):
                st.write("<15 Complacency\n\n15–25 Normal\n\n25–30 Elevated\n\n>30 Fear / crisis")

        with s3:
            st.metric("RSI (14d)", f"{rsi_val:.1f}", delta="↗" if rsi_rising else "↘", delta_color="off")
            with st.expander("What this means"):
                st.write(f"<30 Oversold — potential rebound zone\n\n30–70 Neutral\n\n>70 Overbought\n\nCurrent: {rsi_val:.1f} ({'Rising' if rsi_rising else 'Falling'})")

        with s4:
            st.metric("MA200 Slope", f"{ma200_slope:+.2f}%")
            with st.expander("What this means"):
                st.write("Positive = uptrend intact\n\nNegative = downtrend\n\n|slope| < 0.5% = flat\n\n|slope| > 2% = strong trend")

        # ── Entry timing detail
        with st.expander("⏱ Entry Timing Detail"):
            st.markdown(f"""
    | Signal | Value | Threshold | Status |
    |--------|-------|-----------|--------|
    | Price change (1d) | `{price_change:+.2f}%` | `{trigger_threshold:.2f}%` | {'✅ Above trigger' if abs(price_change) > trigger_threshold else '⬇️ Below trigger'} |
    | RSI momentum | `{rsi_val:.1f}` | — | {'↗ Rising' if rsi_rising else '↘ Falling'} |
    | MA200 trend | `{ma200_slope:+.2f}%` | 0% | {'✅ Supportive' if not trend_weak else '⚠️ Weak'} |
    | MACD | {'▲ Bullish' if macd_bull else '▼ Bearish'} | — | {'✅' if macd_bull else '⚠️'} |
    | MACD Accel | {'⚡ Positive' if macd_accel else '— Neutral'} | — | {'✅' if macd_accel else '—'} |
    | Volatility (20d) | `{volatility:.2f}%` | — | — |
    | Dist from MA200 | `{dist_ma200:+.1f}%` | — | {'🟢 Below' if dist_ma200 < 0 else '🔴 Above'} |
    | 52W High dist | `{dist_52h:+.1f}%` | — | — |
    | 52W Low dist | `{dist_52l:+.1f}%` | — | — |
            """)

        # ── Chart
        st.markdown("### 📊 Price Chart (2Y)")
        chart_df = pd.DataFrame({
            "Price": close,
            "MA50":  ma50,
            "MA200": ma200,
        }).dropna(subset=["Price"])
        st.line_chart(chart_df, use_container_width=True)

        # ── MACD histogram chart
        st.markdown("### 📈 MACD")
        macd_df = pd.DataFrame({
            "MACD":      macd_line,
            "Signal":    macd_sig,
            "Histogram": macd_hist,
        }).tail(180).dropna()
        st.line_chart(macd_df, use_container_width=True)

        # ── RSI chart
        st.markdown("### 📉 RSI (14d)")
        rsi_df = rsi_series.tail(180).to_frame(name="RSI").dropna()
        st.line_chart(rsi_df, use_container_width=True)
        st.caption("Oversold zone: <30 · Overbought zone: >70")



_tab_scanner, _tab_dive = st.tabs(["🔭 Market Scanner", "🔬 Deep Dive — Individual Asset"])

with _tab_scanner:
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
            if   fg_auto <= 25: gauge = "🔴"
            elif fg_auto <= 44: gauge = "🟠"
            elif fg_auto <= 55: gauge = "🟡"
            elif fg_auto <= 75: gauge = "🟢"
            else:               gauge = "💚"
            st.metric(f"{gauge} Fear & Greed", f"{fg_auto} / 100",
                delta=fg_rating, delta_color="off")
            fg_index = fg_auto
            st.caption("[🧠 CNN Fear & Greed](https://edition.cnn.com/markets/fear-and-greed)")
        else:
            fg_index = st.slider("Fear & Greed (manual — all sources failed)", 0, 100, 50)
            st.warning("All F&G auto-sources failed. Enter manually above. "
                       "Check [CNN](https://edition.cnn.com/markets/fear-and-greed).")

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

    if not asset_type:
        st.info("👈 Select an Asset Type in the sidebar to get started.")
        st.stop()
    if len(tickers) == 0:
        st.warning("⚠️ No tickers match the current filters. Try broadening your selection.")
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

            # Drop duplicate tickers (safety net)
            df = df.drop_duplicates(subset=["Ticker"], keep="first")

            # Enrich with justETF metadata if available
            if use_justetf and not jetf_df.empty:
                jcols = {
                    "ticker":      "Ticker",
                    "domicile":    "Domicile",
                    "ter":         "TER",
                    "fund_size_eur":"Fund Size",
                    "dist_policy": "Dist Policy",
                    "replication": "Replication",
                }
                jcols = {k:v for k,v in jcols.items() if k in jetf_df.columns}
                j = jetf_df[list(jcols.keys())].rename(columns=jcols).copy()
                j["Ticker"] = j["Ticker"].str.upper()
                j["TER"] = j["TER"].apply(lambda x: f"{x:.2f}%" if pd.notna(x) and x else "—")
                j["Fund Size"] = j["Fund Size"].apply(lambda x: f"€{x:,.0f}m" if pd.notna(x) and x else "—")
                df = df.merge(j, on="Ticker", how="left", suffixes=("", "_jetf"))
                # Fill placeholders with merged values
                for col in ["Domicile","TER","Fund Size","Dist Policy","Replication"]:
                    jetf_col = col + "_jetf"
                    if jetf_col in df.columns:
                        df[col] = df[jetf_col].fillna(df[col])
                        df.drop(columns=[jetf_col], inplace=True)
                    df[col] = df[col].fillna("—")

            df["Score"] = compute_scores(df)
            df          = df.sort_values("Score", ascending=False).reset_index(drop=True)
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
                "MACD","MACD Accel","Vol%","Confidence",
                "PE","Beta","Div Yield","Margin","Rev Growth","Cap",
                "Domicile","TER","Fund Size","Dist Policy","Replication",
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

    # ═══════════════════════════════════════════════════════════════════
    # DEEP DIVE — Individual Asset Analysis (appended as a second page)
    # ═══════════════════════════════════════════════════════════════════

with _tab_dive:
    run_deep_dive()
