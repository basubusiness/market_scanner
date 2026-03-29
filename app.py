# ═══════════════════════════════════════════════════════════════════
# Market Decision Engine v10.0 — Dash + Railway
# ═══════════════════════════════════════════════════════════════════

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import yfinance as yf
import pandas as pd
import numpy as np
import financedatabase as fd
import requests
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time

try:
    import justetf_scraping
    JUSTETF_AVAILABLE = True
except ImportError:
    JUSTETF_AVAILABLE = False

APP_VERSION = "v10.2"
from datetime import datetime as _datetime
BUILD_TIME = _datetime.utcnow().strftime("%d %b %H:%M UTC")

# ───────────────────────────────────────────────────────────────────
# APP INIT
# ───────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Market Decision Engine",
)
server = app.server  # expose Flask server for gunicorn

# ───────────────────────────────────────────────────────────────────
# IN-MEMORY CACHE (replaces st.cache_data)
# ───────────────────────────────────────────────────────────────────

_cache = {}
_cache_lock = threading.Lock()
_active_scans = {}  # scan_id -> bool, False = cancelled

def cache_get(key):
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        val, expires = entry
        if expires and time.time() > expires:
            del _cache[key]
            return None
        return val

def cache_set(key, val, ttl=None):
    with _cache_lock:
        expires = (time.time() + ttl) if ttl else None
        _cache[key] = (val, expires)

# ───────────────────────────────────────────────────────────────────
# UNIVERSE LOADERS
# ───────────────────────────────────────────────────────────────────

def load_base_universe():
    cached = cache_get("universe")
    if cached is not None:
        return cached

    # Try pre-built universe.csv first (instant, pre-resolved symbols)
    import os
    csv_path = os.path.join(os.path.dirname(__file__), "universe.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            print(f"[universe] Loaded from universe.csv: {len(df):,} rows", flush=True)
            for col in ["country","sector","name","category_group","category",
                        "currency","yf_symbol","yf_suffix","isin"]:
                if col not in df.columns:
                    df[col] = ""
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].fillna("").astype(str).str.strip()
            df = df[df["ticker"].str.match(r"^[A-Z0-9]{2,6}$")]
            df = df.drop_duplicates(subset=["ticker","type"], keep="first")
            # Clean up mangled → symbols in yf_suffix (ISIN-resolved tickers)
            # These get handled via resolved_ cache, not suffix concatenation
            if "yf_suffix" in df.columns:
                df.loc[df["yf_suffix"].astype(str).str.startswith("→"), "yf_suffix"] = df.loc[
                    df["yf_suffix"].astype(str).str.startswith("→"), "yf_suffix"]  # keep as-is, handled in fetch
            cache_set("universe", df)
            return df
        except Exception as e:
            print(f"[universe] CSV load failed: {e}, falling back to live", flush=True)

    # Fall back to live loading
    print("[universe] Loading from financedatabase (slow)...", flush=True)
    etfs     = fd.ETFs().select().copy()
    equities = fd.Equities().select().copy()
    etfs["type"]     = "ETF"
    equities["type"] = "Stock"
    df = pd.concat([etfs, equities], axis=0).reset_index()
    df.rename(columns={df.columns[0]: "ticker"}, inplace=True)
    for col in ["country","sector","name","category_group","category",
                "currency","exchange","family","industry_group",
                "yf_symbol","yf_suffix","isin"]:
        if col not in df.columns:
            df[col] = ""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("").astype(str).str.strip()
    df = df[df["ticker"].str.match(r"^[A-Z]{2,5}$")]
    df = df[~df["ticker"].str.startswith("$")]
    df = df.drop_duplicates(subset=["ticker","type"], keep="first")
    cache_set("universe", df)
    return df

def load_justetf():
    cached = cache_get("justetf")
    if cached is not None:
        return cached
    if not JUSTETF_AVAILABLE:
        return pd.DataFrame()
    try:
        df = justetf_scraping.load_overview().reset_index()
        print(f"[justETF] raw cols: {list(df.columns)[:25]}", flush=True)
        print(f"[justETF] shape: {df.shape}", flush=True)
        # Auto-detect columns regardless of package version
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl == "ticker":                                col_map[c] = "ticker"
            elif cl == "isin":                                col_map[c] = "isin"
            elif "name" in cl and "index" not in cl:         col_map[c] = "jname"
            elif "domicile" in cl:                           col_map[c] = "domicile"
            elif cl in ("ter","expense_ratio"):              col_map[c] = "ter"
            elif "distribution" in cl or "policy" in cl:    col_map[c] = "dist_policy"
            elif "fund_size" in cl or "aum" in cl:          col_map[c] = "fund_size_eur"
            elif "replication" in cl:                        col_map[c] = "replication"
            elif "strategy" in cl:                           col_map[c] = "strategy"
        df = df.rename(columns=col_map)
        keep = [c for c in ["ticker","isin","jname","domicile","ter",
                             "dist_policy","fund_size_eur","replication","strategy"]
                if c in df.columns]
        df = df[keep].copy()
        # Infer dist_policy from ETF name if not provided by the package
        if "dist_policy" not in df.columns and "jname" in df.columns:
            def _infer_dist(name):
                n = str(name).lower()
                if any(x in n for x in [" acc", "(acc)", "accumulating","accumulation","thesaurierend","capitalisation"]):
                    return "Accumulating"
                if any(x in n for x in [" dist", "(dist)", " inc", "(inc)", "distributing","distribution","dividend","ausschüttend"]):
                    return "Distributing"
                return "Accumulating"  # majority of UCITS ETFs are accumulating
            df["dist_policy"] = df["jname"].apply(_infer_dist)
            print(f"[justETF] inferred dist_policy: {df['dist_policy'].value_counts().to_dict()}", flush=True)
        df["ticker"] = df["ticker"].fillna("").astype(str).str.strip().str.upper()
        df = df[df["ticker"].str.match(r"^[A-Z0-9]{1,6}$")]
        df = df.drop_duplicates(subset=["ticker"], keep="first")
        print(f"[justETF] loaded {len(df)} ETFs, cols: {list(df.columns)}", flush=True)
        if "domicile" in df.columns:
            print(f"[justETF] domicile sample: {df['domicile'].dropna().unique()[:8].tolist()}", flush=True)
        if "dist_policy" in df.columns:
            print(f"[justETF] dist sample: {df['dist_policy'].dropna().unique()[:5].tolist()}", flush=True)
        cache_set("justetf", df, ttl=86400*7)
        return df
    except Exception as e:
        print(f"[justETF] ERROR: {e}", flush=True)
        return pd.DataFrame()

# Load on startup (background thread so server starts fast)
# Load synchronously — universe must be ready before any callback runs
universe = load_base_universe()
jetf_df  = load_justetf()

def load_signals():
    import os
    path = os.path.join(os.path.dirname(__file__), "signals.csv")
    if not os.path.exists(path):
        print("[signals] signals.csv not found — using live yfinance scan", flush=True)
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        date_str = df["computed_at"].iloc[0] if "computed_at" in df.columns else "unknown"

        # Clean up — deduplicate tickers, remove warrants/SPACs/penny stocks
        orig_len = len(df)
        # Filter bad tickers
        def _bad_ticker(t):
            t = str(t).strip()
            if len(t) < 2: return True
            if t[-1] in ('W','U','R') and len(t) >= 4: return True  # warrants/units/rights
            if t.endswith(('WS','WT','WI','WD')): return True        # more warrant formats
            return False
        df = df[~df["ticker"].apply(_bad_ticker)]
        # Filter SPAC/shell names and leveraged/inverse ETPs
        if "name" in df.columns:
            spac_keywords = ["acquisition corp","acquisition co","blank check",
                             "special purpose","spac","shell company"]
            leveraged_keywords = ["2x","3x","4x","5x","-1x","-2x","-3x",
                                   "2x long","3x long","2x short","3x short",
                                   "daily 2x","daily 3x","ultra short","ultrashort",
                                   "leveraged","inverse daily","short daily"]
            def _is_spac(name):
                n = str(name).lower()
                return any(kw in n for kw in spac_keywords)
            def _is_leveraged(name):
                n = str(name).lower()
                return any(kw in n for kw in leveraged_keywords)
            df = df[~df["name"].apply(_is_spac)]
            df = df[~df["name"].apply(_is_leveraged)]
        # Remove penny stocks (price < $0.50) — but keep momentum-only signals (price=NaN)
        if "price" in df.columns:
            df = df[(df["price"].isna()) | (df["price"] >= 1.00)]
        # Keep best signal per ticker
        if "score" in df.columns:
            df = df.sort_values("score", ascending=False).drop_duplicates(
                subset=["ticker"], keep="first").reset_index(drop=True)

        actions = df["action"].value_counts().to_dict() if "action" in df.columns else {}
        print(f"[signals] Loaded {len(df):,} signals (was {orig_len:,}, computed: {date_str}) — "
              f"BUY:{actions.get('BUY',0)} WATCH:{actions.get('WATCH',0)} "
              f"SELL:{actions.get('SELL',0)} AVOID:{actions.get('AVOID',0)}", flush=True)
        return df
    except Exception as e:
        print(f"[signals] Load error: {e}", flush=True)
        return pd.DataFrame()

signals_df = load_signals()

# Pre-warm suffix cache for top justETF tickers in background
def _prewarm_suffix_cache():
    if jetf_df.empty:
        return
    top = jetf_df["ticker"].head(300).tolist()
    for t in top:
        if cache_get(f"sfx_{t}") is None:
            try:
                df = flatten_df(yf.Ticker(t + ".DE").history(period="3d", auto_adjust=True))
                if not df.empty and "Close" in df.columns and len(df["Close"].dropna()) >= 1:
                    cache_set(f"sfx_{t}", ".DE", ttl=86400*7)
            except Exception:
                pass

# Prewarm disabled — conflicts with scan API calls
# threading.Thread(target=_prewarm_suffix_cache, daemon=True).start()

# Pre-build name lookup from universe (instant, improves first scan speed)
_name_lookup = {}
if not universe.empty and "name" in universe.columns:
    for _, row in universe.iterrows():
        t = str(row.get("ticker","")).strip()
        n = str(row.get("name","")).strip()
        if t and n and n not in ("","nan","None"):
            _name_lookup[t] = n
if not jetf_df.empty:
    for _, row in jetf_df.iterrows():
        t = str(row.get("ticker","")).strip()
        n = str(row.get("jname","")).strip()
        i = str(row.get("isin","")).strip() if pd.notna(row.get("isin","")) else ""
        if t and n and n not in ("","nan","None"):
            _name_lookup[t] = (n, i)

# ───────────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ───────────────────────────────────────────────────────────────────

def flatten_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(window).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
    rs    = gain / loss
    rs[loss == 0] = np.inf
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    ema12  = prices.ewm(span=12, adjust=False).mean()
    ema26  = prices.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal, macd - signal

def linear_slope(series, window=10):
    y = series.tail(window).values
    if len(y) < window:
        return 0.0
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    return float(slope / (y.mean() + 1e-9))

# ───────────────────────────────────────────────────────────────────
# LIVE DATA
# ───────────────────────────────────────────────────────────────────

def get_fg_index():
    cached = cache_get("fg")
    if cached is not None:
        return cached
    headers = {"User-Agent": "Mozilla/5.0"}
    import datetime as dt
    today = dt.date.today().strftime("%Y-%m-%d")
    for url in [
        "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
        f"https://production.dataviz.cnn.io/index/fearandgreed/graphdata/{today}",
    ]:
        try:
            r = requests.get(url, headers=headers, timeout=6)
            if r.status_code == 200:
                fg = r.json().get("fear_and_greed", {})
                if "score" in fg:
                    s = round(float(fg["score"]))
                    label = fg.get("rating","").replace("_"," ").title()
                    result = (s, f"{label} (CNN)")
                    cache_set("fg", result, ttl=1800)
                    return result
        except Exception:
            continue
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", headers=headers, timeout=6)
        if r.status_code == 200:
            e = r.json().get("data", [{}])[0]
            s = round(float(e.get("value", 50)))
            label = e.get("value_classification","").replace("_"," ").title()
            result = (s, f"{label} (alt.me)")
            cache_set("fg", result, ttl=1800)
            return result
    except Exception:
        pass
    return (50, "Unavailable")

def get_live_vix():
    cached = cache_get("vix")
    if cached is not None:
        return cached
    try:
        d = flatten_df(yf.Ticker("^VIX").history(period="5d", auto_adjust=True))
        if not d.empty and "Close" in d.columns:
            val = float(d["Close"].dropna().iloc[-1])
            cache_set("vix", val, ttl=300)
            return val
    except Exception:
        pass
    return 20.0

# ───────────────────────────────────────────────────────────────────
# TICKER ANALYSIS
# ───────────────────────────────────────────────────────────────────

def _fetch_stooq(symbol):
    """Try Stooq as first data source — fast, no rate limits."""
    try:
        import pandas_datareader as pdr
        from datetime import timedelta
        # Convert yfinance suffix to Stooq format
        stooq_sym = symbol
        if symbol.endswith(".L"):   stooq_sym = symbol.replace(".L", ".UK")
        elif symbol.endswith(".AS"): stooq_sym = symbol.replace(".AS", ".NL")
        elif symbol.endswith(".PA"): stooq_sym = symbol.replace(".PA", ".FR")
        elif symbol.endswith(".MI"): stooq_sym = symbol.replace(".MI", ".IT")
        elif symbol.endswith(".SW"): stooq_sym = symbol.replace(".SW", ".CH")
        elif "." not in symbol:     stooq_sym = symbol  # skip .US — let yfinance handle ambiguous tickers
        end   = pd.Timestamp.today()
        start = end - pd.Timedelta(days=400)
        df = pdr.get_data_stooq(stooq_sym, start=start, end=end)
        if df.empty or "Close" not in df.columns:
            return pd.DataFrame()
        df = df.sort_index()
        if len(df["Close"].dropna()) >= 30:
            return flatten_df(df)
    except Exception:
        pass
    return pd.DataFrame()


def fetch_justetf_chart(isin):
    """
    Fetch full price history from justETF by ISIN.
    Returns DataFrame with Close column, or empty DataFrame.
    This is the most reliable source for UCITS ETFs.
    """
    if not isin or len(str(isin)) != 12:
        return pd.DataFrame()
    try:
        import justetf_scraping
        df = justetf_scraping.load_chart(str(isin), unclosed=True)
        if df is None or df.empty:
            return pd.DataFrame()
        # Rename quote → Close for compatibility
        if "quote" in df.columns:
            df = df.rename(columns={"quote": "Close"})
        elif "quote_with_dividends" in df.columns:
            df = df.rename(columns={"quote_with_dividends": "Close"})
        if "Close" not in df.columns:
            return pd.DataFrame()
        # Convert index to DatetimeIndex if needed
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        # Keep last 1 year
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=400)
        df = df[df.index >= cutoff]
        close = df["Close"].dropna()
        if len(close) < 30:
            return pd.DataFrame()
        print(f"[justETF] chart loaded for {isin}: {len(df)} days, "
              f"latest={close.iloc[-1]:.2f}", flush=True)
        return df
    except Exception as e:
        print(f"[justETF] chart error for {isin}: {e}", flush=True)
        return pd.DataFrame()


def fetch_ticker_data(ticker, isin=None):
    # Quick reject obvious invalid tickers before any API call
    if not ticker or ticker.startswith("$") or len(ticker) < 2:
        return None
    key = f"tick_{ticker}"
    cached = cache_get(key)
    if cached is not None:
        return None if cached == "FAILED" else cached
    try:
        # Check universe.csv pre-resolved symbol first
        known_sfx = cache_get(f"sfx_{ticker}")
        resolved_sym = cache_get(f"resolved_{ticker}")
        if known_sfx is None and resolved_sym is None and not universe.empty and "yf_symbol" in universe.columns:
            rows = universe[universe["ticker"] == ticker]
            if not rows.empty:
                yf_sym = str(rows.iloc[0].get("yf_symbol","")).strip()
                yf_sfx = str(rows.iloc[0].get("yf_suffix","")).strip()
                if yf_sym and yf_sym not in ("","nan","None"):
                    if yf_sfx.startswith("→"):
                        # Full symbol replacement (ISIN-resolved to different ticker)
                        resolved_sym = yf_sym
                        cache_set(f"resolved_{ticker}", yf_sym, ttl=86400*7)
                    else:
                        known_sfx = "" if yf_sfx in ("","nan","None") else yf_sfx
                        cache_set(f"sfx_{ticker}", known_sfx, ttl=86400*7)

        # Check if we already know the working suffix
        known_sfx = cache_get(f"sfx_{ticker}")
        resolved_sym = resolved_sym or cache_get(f"resolved_{ticker}")
        def _fetch_history(sym):
            """Fetch 1y history with timeout protection."""
            try:
                return flatten_df(yf.Ticker(sym).history(
                    period="1y", auto_adjust=True, timeout=10))
            except Exception:
                return pd.DataFrame()

        def _valid(df):
            return (not df.empty and "Close" in df.columns
                    and len(df["Close"].dropna()) >= 30)

        # Determine the symbol to use
        target_sym = resolved_sym or (ticker + known_sfx if known_sfx is not None else ticker)

        # Try Stooq first — fast, no rate limits
        df = _fetch_stooq(target_sym)

        # Fall back to yfinance if Stooq fails
        if not _valid(df):
            if resolved_sym:
                df = _fetch_history(resolved_sym)
            elif known_sfx is not None:
                df = _fetch_history(ticker + known_sfx)
            else:
                df = _fetch_history(ticker)
                if not _valid(df):
                    _sfx_list = [".DE",".DU",".L",".AS",".SG"] if isin else [".DE",".DU",".L",".AS",".PA",".MI",".SW",".SG",".F",".VI",".BR"]
                    for sfx in _sfx_list:
                        df2 = _fetch_history(ticker + sfx)
                        if _valid(df2):
                            df = df2
                        cache_set(f"sfx_{ticker}", sfx, ttl=86400*7)
                        break
            # Cache negative result for tickers that failed all suffixes
            if not _valid(df) and not isin:
                cache_set(key, "FAILED", ttl=3600)
                return None
            # Last resort: search by ISIN
            if not _valid(df) and isin:
                try:
                    sr = yf.Search(isin, max_results=3)
                    if hasattr(sr,"quotes") and sr.quotes:
                        sym2 = sr.quotes[0]["symbol"]
                        df2 = _fetch_history(sym2)
                        if _valid(df2):
                            df = df2
                            cache_set(f"sfx_{ticker}", f"→{sym2}", ttl=86400*7)
                except Exception:
                    pass
        if df.empty or "Close" not in df.columns:
            # For ETFs with ISIN — try justETF chart as last resort
            if isin and len(str(isin)) == 12:
                df = fetch_justetf_chart(isin)
                if df.empty or "Close" not in df.columns:
                    cache_set(key, "FAILED", ttl=3600)
                    return None
            else:
                cache_set(key, "FAILED", ttl=3600)
                return None
        close = df["Close"].dropna()
        if len(close) < 30:
            cache_set(key, "FAILED", ttl=3600)
            return None
        # Sanity: if data looks like wrong ticker, try justETF for ETFs
        _ma_check = float(close.rolling(200).mean().iloc[-1])
        if _ma_check > 0 and (float(close.iloc[-1]) / _ma_check) > 10 and isin:
            df_j = fetch_justetf_chart(isin)
            if not df_j.empty and len(df_j["Close"].dropna()) >= 30:
                df    = df_j
                close = df["Close"].dropna()
        price = float(close.iloc[-1])
        if price < 0.10:  # only filter near-zero prices
            return None
        # Sanity: if price is >10x MA200, likely wrong ticker (e.g. SXRN.US vs SXRN.DE)
        _ma200c = float(close.rolling(200).mean().iloc[-1])
        if _ma200c > 0 and (price / _ma200c) > 10:
            return None
        if "Volume" in df.columns:
            avg_vol = df["Volume"].dropna().tail(20).mean()
            # ETFs can have very low volume but still be valid — only filter true zeros
            if avg_vol == 0:
                return None
        ma50   = float(close.rolling(50).mean().iloc[-1])
        ma200  = float(close.rolling(200).mean().iloc[-1])
        rsi_s  = calculate_rsi(close)
        rsi    = float(rsi_s.iloc[-1])
        if not (1 < rsi < 99):
            return None
        macd_l, macd_sig, macd_h = calculate_macd(close)
        rsi_slope   = linear_slope(rsi_s.dropna(), window=5)
        dist_ma     = ((price - ma200) / ma200) * 100
        dist_52h    = ((price - float(close.tail(252).max())) / float(close.tail(252).max())) * 100
        vol_pct     = float(close.pct_change().rolling(20).std().iloc[-1] * 100)
        price_slope = linear_slope(close, window=10)
        conf  = min(abs(dist_ma)/20,1)*0.5 + min(abs(50-rsi)/50,1)*0.5
        conf *= 1 - min(vol_pct/5, 0.5)
        result = dict(
            price=price, ma50=ma50, ma200=ma200,
            rsi=rsi, rsi_slope=rsi_slope,
            macd=float(macd_l.iloc[-1]), macd_signal=float(macd_sig.iloc[-1]),
            macd_hist=float(macd_h.iloc[-1]),
            dist_ma=dist_ma, dist_52h=dist_52h,
            vol=vol_pct, price_slope=price_slope,
            trend_down_strong=(price_slope < -0.003),
            confidence=conf,
            # store full series for deep dive charts
            close=close, ma50_s=close.rolling(50).mean(),
            ma200_s=close.rolling(200).mean(),
            rsi_s=rsi_s, macd_l=macd_l, macd_sig=macd_sig, macd_h=macd_h,
        )
        cache_set(key, result, ttl=3600)
        return result
    except Exception:
        return None

def analyse_ticker(ticker, risk_mult, isin=None):
    raw = fetch_ticker_data(ticker, isin=isin)
    if raw is None:
        return None
    dm, rsi, conf = raw["dist_ma"], raw["rsi"], raw["confidence"]
    macd_bull  = raw["macd"] > raw["macd_signal"]
    macd_accel = raw["macd_hist"] > 0
    rsi_rising = raw["rsi_slope"] > 0
    # Knife threshold — use fixed -25% floor so extreme fear doesn't flag everything
    knife_thr  = max(-25, -15 * (1 / risk_mult))
    is_knife   = (dm < knife_thr) and (rsi < 30) and raw["trend_down_strong"]
    reversal   = is_knife and macd_bull and rsi_rising

    if is_knife and not reversal:
        action = "AVOID"
    elif dm < -10 and rsi < 45 and macd_bull and macd_accel:
        action = "BUY"
    elif dm < -10 and rsi < 45 and (macd_bull or rsi_rising):
        action = "WATCH"
    elif dm < -5 and rsi < 35:
        # Deep oversold even without MACD confirmation — worth watching
        action = "WATCH"
    elif dm > 10 and rsi > 70:
        action = "SELL"
    else:
        action = "WAIT"
    strength = "Strong" if conf > 0.7 else "Medium" if conf > 0.4 else "Weak"
    dist_s = min(-dm/30,1) if dm < 0 else 0
    rsi_s2 = max((50-rsi)/50, 0)
    score  = dist_s*0.30 + rsi_s2*0.25 + (0.20 if macd_bull else 0) + (0.10 if macd_accel else 0) + conf*0.15
    if action == "AVOID": score -= 2
    return {
        "Ticker": ticker,  # always show original ticker, not exchange-suffixed one
        "Price": round(raw["price"],2),
        "MA200": round(raw["ma200"],2), "Dist%": round(dm,1),
        "52W%": round(raw["dist_52h"],1), "RSI": round(rsi,1),
        "RSI↗": "↗" if rsi_rising else "↘",
        "MACD": "▲ Bull" if macd_bull else "▼ Bear",
        "MACD⚡": "⚡" if macd_accel else "—",
        "Vol%": round(raw["vol"],2), "Conf": round(conf,2),
        "Action": action, "Signal": f"{action} ({strength})",
        "Knife": "⚠️" if (is_knife and not reversal) else ("✅Rev" if reversal else ""),
        "Score": round(score,4),
    }

def resolve_yf_ticker(ticker, isin=None):
    """Find the correct yfinance ticker symbol for a given ETF ticker/ISIN.
    Returns the working yfinance symbol or None."""
    # Check suffix cache first
    sfx = cache_get(f"sfx_{ticker}")
    if sfx:
        return ticker + sfx

    def _has_price(sym):
        """Quick check if a symbol has valid price data."""
        try:
            t = yf.Ticker(sym)
            # Try fast_info first
            fi = t.fast_info
            for attr in ["last_price", "regularMarketPrice", "currentPrice"]:
                val = getattr(fi, attr, None)
                if val and float(val) > 0:
                    return True
            # Fall back to 3-day history
            df = flatten_df(t.history(period="3d", auto_adjust=True))
            return not df.empty and "Close" in df.columns and len(df["Close"].dropna()) >= 1
        except Exception:
            return False

    # Try bare ticker
    if _has_price(ticker):
        cache_set(f"sfx_{ticker}", "", ttl=86400*7)
        return ticker

    # Try suffixes — .DE first since justETF uses Xetra tickers
    for sfx in [".DE",".DU",".L",".AS",".PA",".MI",".SW",".SG",".F",".VI",".BR"]:
        if _has_price(ticker + sfx):
            cache_set(f"sfx_{ticker}", sfx, ttl=86400*7)
            return ticker + sfx

    # Last resort: search by ISIN
    if isin:
        try:
            sr = yf.Search(isin, max_results=3)
            if hasattr(sr,"quotes") and sr.quotes:
                sym = sr.quotes[0]["symbol"]
                cache_set(f"sfx_{ticker}", f"_resolved_{sym}", ttl=86400*7)
                return sym
        except Exception:
            pass

    return None

def get_name_isin(ticker):
    """
    Fast name/ISIN lookup — priority order:
    1. In-memory cache (instant)
    2. justETF DataFrame (instant, no API)
    3. financedatabase universe (instant, no API)
    4. yfinance .info (slow — only if nothing else found)
    """
    key = f"meta_{ticker}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    # Source 0: pre-built lookup dict (instant dict lookup)
    if ticker in _name_lookup:
        val = _name_lookup[ticker]
        if isinstance(val, tuple):
            cache_set(key, val, ttl=86400)
            return val
        result = (val[:40], "")
        cache_set(key, result, ttl=86400)
        return result

    # Source 1: justETF (has ISIN + clean names)
    if not jetf_df.empty and ticker in jetf_df["ticker"].values:
        row  = jetf_df[jetf_df["ticker"]==ticker].iloc[0]
        name = str(row.get("jname",""))[:40]
        isin = str(row.get("isin","")) if pd.notna(row.get("isin","")) else ""
        if name and name not in ("", "nan"):
            result = (name, isin)
            cache_set(key, result, ttl=86400)
            return result

    # Source 2: financedatabase universe "name" column (instant, no API)
    if not universe.empty and "name" in universe.columns:
        rows = universe[universe["ticker"] == ticker]
        if not rows.empty:
            name = str(rows.iloc[0].get("name",""))[:40]
            if name and name not in ("", "nan"):
                result = (name, "")
                cache_set(key, result, ttl=86400)
                return result

    # Source 3: yfinance .fast_info — lightweight, just gets basic metadata
    # Only called if sources 1+2 failed — skipped for bulk scans to stay fast
    # (name_map builder passes skip_slow=True for bulk)
    result = (ticker, "")
    cache_set(key, result, ttl=3600)  # shorter cache since name wasn't found
    return result

def get_name_isin_full(ticker):
    """Slower version that also tries yfinance — used in Deep Dive only."""
    name, isin = get_name_isin(ticker)
    if name != ticker:
        return name, isin
    try:
        info = yf.Ticker(ticker).info
        name = (info.get("longName") or info.get("shortName") or ticker)[:40]
        isin = info.get("isin","") or ""
        result = (name, isin)
        cache_set(f"meta_{ticker}", result, ttl=86400)
        return result
    except Exception:
        return ticker, ""

def fetch_pe_data(ticker):
    key = f"pe_{ticker}"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        info = yf.Ticker(ticker).info
        pe   = None
        for f in ["trailingPE","forwardPE"]:
            v = info.get(f)
            if v and 0 < float(v) < 2000:
                pe = round(float(v),1)
                break
        mcap = info.get("marketCap")
        beta = info.get("beta")
        div  = info.get("dividendYield")
        result = {
            "PE":    pe or "—",
            "Beta":  round(float(beta),2) if beta else "—",
            "Div%":  f"{div*100:.2f}%" if div else "—",
            "MCap":  (f"${mcap/1e9:.1f}B" if mcap > 1e9 else f"${mcap/1e6:.0f}M") if mcap else "—",
        }
        cache_set(key, result, ttl=86400)
        return result
    except Exception:
        return {"PE":"—","Beta":"—","Div%":"—","MCap":"—"}

# ───────────────────────────────────────────────────────────────────
# UNIVERSE BUILDER
# ───────────────────────────────────────────────────────────────────

PRESETS = {
    "🌍 All ETFs":             {"type":"ETF"},
    "🇪🇺 UCITS ETFs":          {"type":"ETF","domicile":["Ireland","Luxembourg"]},
    "🇺🇸 US ETFs":             {"type":"ETF","domicile":["United States"]},
    "📈 US Stocks":            {"type":"Stock","country":["United States"]},
    "🇬🇧 UK Stocks":           {"type":"Stock","country":["United Kingdom"]},
    "🇩🇪 German Stocks":       {"type":"Stock","country":["Germany"]},
    "🌐 Global Stocks":        {"type":"Stock","country":["United States","United Kingdom","Germany","France","Japan","Canada"]},
    "💰 Equity ETFs":          {"type":"ETF","category_group":["Equities"]},
    "🏦 Fixed Income ETFs":    {"type":"ETF","category_group":["Fixed Income"]},
    "🥇 Commodity ETFs":       {"type":"ETF","category_group":["Commodities"]},
    "🔧 Custom":               {"type":"custom"},
}

def build_tickers(preset_key, filters):
    if universe.empty:
        return []
    preset = PRESETS.get(preset_key, {"type":"ETF"})
    ptype  = preset.get("type","ETF")

    def safe_jcol(col):
        if jetf_df.empty or col not in jetf_df.columns:
            return []
        return sorted([v for v in jetf_df[col].dropna().unique() if str(v).strip()])

    parts = []

    # ETF branch
    etf_types = ["ETF"] if ptype == "ETF" else (filters.get("types",[]) if ptype=="custom" else [])
    if "ETF" in etf_types or ptype == "ETF":
        need_jetf = (
            "domicile" in preset or filters.get("domicile") or filters.get("dist_policy") or
            filters.get("replication") or filters.get("strategy") or
            filters.get("min_size",0) > 0 or filters.get("max_ter",2.0) < 2.0
        )

        if need_jetf and not jetf_df.empty:
            # Use justETF as PRIMARY source — it has 4000+ UCITS ETFs with correct metadata
            # financedatabase only has ~2800 and misses most UCITS ETFs
            e = jetf_df.copy()
            dom = preset.get("domicile") or filters.get("domicile",[])
            if dom and "domicile" in e.columns:
                e = e[e["domicile"].isin(dom)]
            if filters.get("dist_policy") and "dist_policy" in e.columns:
                e = e[e["dist_policy"].isin(filters["dist_policy"])]
            if filters.get("replication") and "replication" in e.columns:
                e = e[e["replication"].isin(filters["replication"])]
            if filters.get("strategy") and "strategy" in e.columns:
                e = e[e["strategy"].isin(filters["strategy"])]
            if filters.get("min_size",0) > 0 and "fund_size_eur" in e.columns:
                e = e[e["fund_size_eur"].fillna(0) >= filters["min_size"]]
            if "ter" in e.columns:
                if filters.get("min_ter",0) > 0:
                    e = e[e["ter"].fillna(0) >= filters["min_ter"]]
                if filters.get("max_ter",2.0) < 2.0:
                    e = e[e["ter"].fillna(999) <= filters["max_ter"]]
            # Apply category filter using financedatabase if requested
            cat = preset.get("category_group") or filters.get("category",[])
            if cat:
                fd_cats = universe[universe["type"]=="ETF"][["ticker","category_group"]]
                e = e.merge(fd_cats, on="ticker", how="left")
                e = e[e["category_group"].isin(cat)]
        else:
            # No justETF filters — use financedatabase universe + optional category filter
            e = universe[universe["type"]=="ETF"].copy()
            cat = preset.get("category_group") or filters.get("category",[])
            if cat:
                e = e[e["category_group"].isin(cat)]
            # Apply preset domicile filter via justETF join
            dom = preset.get("domicile",[])
            if dom and not jetf_df.empty and "domicile" in jetf_df.columns:
                e = e.merge(jetf_df[["ticker","domicile"]], on="ticker", how="inner")
                e = e[e["domicile"].isin(dom)]

        parts.append(e)

    # Stock branch
    stock_types = ["Stock"] if ptype == "Stock" else (filters.get("types",[]) if ptype=="custom" else [])
    if "Stock" in stock_types or ptype == "Stock":
        s = universe[universe["type"]=="Stock"].copy()
        country = preset.get("country") or filters.get("country",[])
        if country: s = s[s["country"].isin(country)]
        if filters.get("sector"): s = s[s["sector"].isin(filters["sector"])]
        # Filter out SPACs, warrants, units (tickers ending in U, W, R, +, =)
        s = s[~s["ticker"].str.match(r"^[A-Z]{2,4}[UWR]$")]
        # Sort: shorter tickers first (more likely to be real companies)
        # then alphabetically within same length
        s = s.iloc[s["ticker"].str.len().argsort(kind="stable")]
        parts.append(s)

    if not parts:
        return []
    filtered = pd.concat(parts)
    return list(dict.fromkeys(filtered["ticker"].str.upper().str.strip().tolist()))

# ───────────────────────────────────────────────────────────────────
# LAYOUT HELPERS
# ───────────────────────────────────────────────────────────────────

SIGNAL_COLORS = {
    "BUY":   "#00e676", "WATCH": "#00bcd4",
    "AVOID": "#ff6d00", "SELL":  "#ff1744", "WAIT": "#ffd600",
}

def badge(action):
    color = SIGNAL_COLORS.get(action, "#888")
    return html.Span(action, style={
        "background": color, "color": "#000", "fontWeight": "bold",
        "padding": "2px 8px", "borderRadius": "4px", "fontSize": "12px"
    })

def kpi_card(title, value, color="#fff"):
    return dbc.Card(dbc.CardBody([
        html.P(title, className="mb-0", style={"fontSize":"12px","color":"#aaa"}),
        html.H4(value, style={"color": color, "marginBottom":0}),
    ]), style={"background":"#1e1e2e","border":"1px solid #333"})

# ───────────────────────────────────────────────────────────────────
# LAYOUT
# ───────────────────────────────────────────────────────────────────

# Dark theme style for all Dropdown components
_DD = {"backgroundColor":"#1e1e2e","color":"#fff","border":"1px solid #555","marginBottom":"6px"}
_DD_STYLE = {"option":{"backgroundColor":"#1e1e2e","color":"#fff"},
             "control":{"backgroundColor":"#1e1e2e","borderColor":"#555","color":"#fff"},
             "singleValue":{"color":"#fff"},"placeholder":{"color":"#aaa"},
             "menu":{"backgroundColor":"#1e1e2e"},"input":{"color":"#fff"},
             "multiValue":{"backgroundColor":"#333"},
             "multiValueLabel":{"color":"#fff"}}

def _get_jcol_opts(col):
    """Get options from justETF df — with hard fallbacks."""
    if not jetf_df.empty and col in jetf_df.columns:
        vals = sorted([v for v in jetf_df[col].astype(str).str.strip().unique()
                       if v and v not in ("","nan","None")])
        if vals:
            return [{"label":v,"value":v} for v in vals]
    # Hard fallbacks per column
    fallbacks = {
        "domicile":    ["Ireland","Luxembourg","Germany","France","Netherlands","United States"],
        "dist_policy": ["Accumulating","Distributing"],
        "replication": ["Physical (Full)","Physical (Sampling)","Swap-based"],
        "strategy":    ["Long-only","Short & Leveraged","Active"],
    }
    return [{"label":v,"value":v} for v in fallbacks.get(col, [])]

def sidebar():
    def_jcol = _get_jcol_opts  # use module-level function
    country_opts = sorted([c for c in universe[universe["type"]=="Stock"]["country"].astype(str).str.strip().unique()
                          if c and c not in ("","nan","None")]) if not universe.empty else []
    sector_opts  = sorted([s for s in universe[universe["type"]=="Stock"]["sector"].astype(str).str.strip().unique()
                          if s and s not in ("","nan","None")]) if not universe.empty else []
    cat_opts     = sorted([c for c in universe[universe["type"]=="ETF"]["category_group"].astype(str).str.strip().unique()
                          if c and c not in ("","nan","None")]) if not universe.empty else []
    if not cat_opts:
        cat_opts = ["Equities","Fixed Income","Commodities","Real Estate",
                    "Alternatives","Cash","Currencies","Derivatives"]

    return html.Div([
        html.Div([
            html.Div([
                html.Span("📡 Market Decision Engine",
                          style={"color":"#fff","fontWeight":"bold","fontSize":"15px"}),
                html.Span(f" {APP_VERSION}",
                          style={"color":"#00bcd4","fontSize":"11px","marginLeft":"6px"}),
            ]),
            html.Small(f"deployed {BUILD_TIME}",
                       style={"color":"#444","fontSize":"10px"}),
            html.Div(id="cache-indicator"),
        ], className="mb-3"),
        html.Hr(style={"borderColor":"#444"}),

        # ── Live indicators
        dbc.Row([
            dbc.Col([html.P("VIX",style={"color":"#aaa","marginBottom":"2px","fontSize":"12px"}),
                     html.H4(id="vix-val", className="text-white")], width=6),
            dbc.Col([html.P("Fear & Greed",style={"color":"#aaa","marginBottom":"2px","fontSize":"12px"}),
                     html.H4(id="fg-val", className="text-white")], width=6),
        ]),
        html.Small(id="fg-label", className="text-muted"),
        html.Div(id="regime-badge", className="mt-2 mb-3"),
        dcc.Interval(id="live-interval", interval=300_000, n_intervals=0),

        html.Hr(style={"borderColor":"#444"}),

        # ── Preset
        html.Label("🎯 Universe preset", className="text-white fw-bold"),
        dcc.Dropdown(
            id="preset-dd",
            options=[{"label":k,"value":k} for k in PRESETS],
            value="🌍 All ETFs",
            clearable=False,
            style={**_DD,"marginBottom":"12px"},
        ),

        # ── Optional filters accordion
        dbc.Accordion([
            dbc.AccordionItem([

                # Asset type (custom only)
                html.Div([
                    html.Label("Asset Type", className="text-white small"),
                    dbc.Checklist(
                        id="filter-types",
                        options=[{"label":"ETF","value":"ETF"},{"label":"Stock","value":"Stock"}],
                        value=["ETF"], inline=True,
                        style={"color":"#fff"},
                    ),
                ], id="types-row", style={"display":"none"}, className="mb-2"),

                # ETF filters
                html.Div([
                    html.Label("📦 ETF Filters", className="text-info small fw-bold mt-2"),
                    html.Label("Domicile", className="text-white small"),
                    dcc.Dropdown(id="filter-domicile", options=def_jcol("domicile"),
                        multi=True, placeholder="Any", style=_DD, className="dash-dark-dd"),
                    html.Label("Distribution", className="text-white small"),
                    dcc.Dropdown(id="filter-dist", options=def_jcol("dist_policy"),
                        multi=True, placeholder="Any", style=_DD, className="dash-dark-dd"),
                    html.Label("Replication", className="text-white small"),
                    dcc.Dropdown(id="filter-replication", options=def_jcol("replication"),
                        multi=True, placeholder="Any", style=_DD, className="dash-dark-dd"),
                    html.Label("Strategy", className="text-white small"),
                    dcc.Dropdown(id="filter-strategy", options=def_jcol("strategy"),
                        multi=True, placeholder="Any", style=_DD, className="dash-dark-dd"),
                    html.Label("Asset Class", className="text-white small"),
                    dcc.Dropdown(id="filter-category", options=[{"label":v,"value":v} for v in cat_opts],
                        multi=True, placeholder="Any", style=_DD, className="dash-dark-dd"),
                    dbc.Row([
                        dbc.Col([html.Label("Min Size €m", className="text-white small"),
                                 dbc.Input(id="filter-minsize", type="number", value=0, min=0, step=50, size="sm")], width=6),
                        dbc.Col([html.Label("Max TER %", className="text-white small"),
                                 dbc.Input(id="filter-maxter", type="number", value=2.0, min=0, max=5, step=0.05, size="sm")], width=6),
                    ], className="mb-2"),
                ], id="etf-filters-row"),

                # Stock filters
                html.Div([
                    html.Label("📈 Stock Filters", className="text-info small fw-bold mt-2"),
                    html.Label("Country", className="text-white small"),
                    dcc.Dropdown(id="filter-country", options=[{"label":v,"value":v} for v in country_opts],
                        multi=True, placeholder="Any", style=_DD, className="dash-dark-dd"),
                    html.Label("Sector", className="text-white small"),
                    dcc.Dropdown(id="filter-sector", options=[{"label":v,"value":v} for v in sector_opts],
                        multi=True, placeholder="Any", style=_DD, className="dash-dark-dd"),
                ], id="stock-filters-row"),

            ], title="🔧 Optional Filters"),
        ], start_collapsed=False, className="mb-3"),

        # ── Settings
        html.Label("💰 Monthly Budget (EUR)", className="text-white small"),
        dbc.Input(id="budget-input", type="number", value=1000, min=100, step=100, className="mb-2", size="sm"),
        html.Label("⚡ Parallel Workers", className="text-white small"),
        dcc.Slider(id="workers-slider", min=2, max=12, step=1, value=6,
                   marks={2:"2",6:"6",12:"12"},
                   tooltip={"placement":"bottom"}),
        dbc.Checklist(id="fetch-pe-check",
            options=[{"label":" Fetch PE / Fundamentals (slower)","value":"pe"}],
            value=[], className="text-white mt-2 mb-3"),

        html.Hr(style={"borderColor":"#444"}),
        html.Small(id="scope-label", className="text-muted"),

        # ── VIX/F&G guide
        dbc.Accordion([dbc.AccordionItem([
            dcc.Markdown("""
**VIX** measures expected S&P 500 volatility:
- < 15 → Complacency
- 15–25 → Normal
- 25–30 → Elevated
- > 30 → Fear
- > 40 → Crisis

[📈 Live VIX](https://finance.yahoo.com/quote/%5EVIX/)

---

**Fear & Greed** (CNN composite):
- 0–25 → Extreme Fear 🔴 *(best entries)*
- 25–45 → Fear 🟠
- 45–55 → Neutral 🟡
- 55–75 → Greed 🟢
- 75–100 → Extreme Greed 💚 *(caution)*

[🧠 CNN F&G](https://edition.cnn.com/markets/fear-and-greed)
            """, className="text-muted", style={"fontSize":"12px"}),
        ], title="📖 VIX & F&G Guide")], start_collapsed=True),

    ], style={
        "width":"280px","minWidth":"280px","background":"#12121f",
        "padding":"16px","height":"100vh","overflowY":"auto",
        "borderRight":"1px solid #333","flexShrink":0,
    })

def scanner_tab():
    return html.Div([
        # Scan controls
        dbc.Row([
            dbc.Col(dbc.Button(
                "🔄 Run Scan",
                id="run-btn", color="danger", size="lg", className="w-100"
            ), width=10),
            dbc.Col(dbc.Button("🗑️", id="clear-btn", color="secondary",
                               size="lg", className="w-100", title="Clear results"),
                    width=2),
        ], className="mb-2 g-1"),
        # Hidden stop button (still needed for callback, triggered by overlay)
        html.Div(dbc.Button(id="stop-btn", style={"display":"none"})),

        # Scan status bar
        html.Div([
            dbc.Alert(id="scan-status-alert", color="info",
                      className="py-2 px-3 mb-2", style={"display":"none"}),
        ], id="scan-status-bar"),

        # Wrap all results in a Loading component
        dcc.Loading(
            id="results-loading",
            type="circle",
            color="#ff4444",
            children=[
                dbc.Row(id="kpi-row", className="mb-3 g-2"),
                html.Div(id="top-signals", className="mb-3"),
                html.Hr(style={"borderColor":"#333"}),
                dbc.Tabs([
                    dbc.Tab(label="All",       tab_id="all"),
                    dbc.Tab(label="🟢 BUY",   tab_id="BUY"),
                    dbc.Tab(label="👀 WATCH", tab_id="WATCH"),
                    dbc.Tab(label="⛔ AVOID", tab_id="AVOID"),
                    dbc.Tab(label="🔴 SELL",  tab_id="SELL"),
                    dbc.Tab(label="🟡 WAIT",  tab_id="WAIT"),
                ], id="signal-tabs", active_tab="all", className="mb-2"),
                html.Div(id="results-table"),
            ]
        ),

        # Download
        html.Div([
            dbc.Button("⬇️ Download CSV", id="dl-btn", color="secondary",
                       size="sm", className="mt-2"),
            dcc.Download(id="dl-csv"),
        ]),

        # Hidden stores
        dcc.Store(id="scan-store"),
        dcc.Store(id="dive-trigger"),
        dcc.Store(id="scan-running", data=False),
        # Progress polling
        dcc.Interval(id="progress-interval", interval=5000, disabled=False),
    ])

def deepdive_tab():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("Ticker or ISIN", className="text-white"),
                dbc.Input(id="dd-input", placeholder="e.g. VWRA or IE00B3RBWM25",
                          type="text", className="mb-2"),
            ], width=8),
            dbc.Col([
                html.Label("Budget (EUR)", className="text-white"),
                dbc.Input(id="dd-budget", type="number", value=1000, min=100, step=100),
            ], width=2),
            dbc.Col([
                html.Br(),
                dbc.Button("🔍 Analyse", id="dd-btn", color="danger",
                           className="w-100 mt-1"),
            ], width=2),
        ], className="mb-3"),

        html.Div(id="dd-from-scanner",className="mb-2"),
        html.Div(id="dd-results"),
    ])

app.layout = html.Div([
    # Scanning overlay — shown while scan runs
    html.Div([
        html.Div([
            dbc.Spinner(color="danger", size="lg"),
            html.H4("Scanning…", className="text-white mt-3 mb-2"),
            html.Div(id="overlay-progress-text",
                     className="text-muted mb-3",
                     style={"fontSize":"14px"}),
            dbc.Progress(id="overlay-progress-bar", value=0, max=100,
                         striped=True, animated=True, color="danger",
                         style={"width":"300px","height":"20px","marginBottom":"20px"}),
            dbc.Button("⏹ Stop Scan", id="overlay-stop-btn",
                       color="warning", size="lg"),
        ], style={
            "display":"flex","flexDirection":"column","alignItems":"center",
            "justifyContent":"center","height":"100%",
        }),
    ], id="scan-overlay", style={
        "display":"none",
        "position":"fixed","top":0,"left":0,"width":"100%","height":"100%",
        "background":"rgba(0,0,0,0.85)","zIndex":9999,
    }),

    # Main flex container
    html.Div([
        sidebar(),
        # Main content
        html.Div([
            dbc.Tabs([
                dbc.Tab(scanner_tab(), label="🔭 Market Scanner", tab_id="scanner"),
                dbc.Tab(deepdive_tab(), label="🔬 Deep Dive",     tab_id="deepdive"),
            ], id="main-tabs", active_tab="scanner"),
        ], style={"flex":"1","padding":"20px","overflowY":"auto","background":"#0d0d1a"}),
    ], style={"display":"flex","height":"100vh","overflow":"hidden"}),
], style={"fontFamily":"'Segoe UI', sans-serif","background":"#0d0d1a","color":"#fff"})

# ───────────────────────────────────────────────────────────────────
# CALLBACKS — Live data
# ───────────────────────────────────────────────────────────────────

@app.callback(
    Output("cache-indicator","children"),
    Input("live-interval","n_intervals"),
)
def update_cache_indicator(_):
    n_tickers = sum(1 for k in _cache if str(k).startswith("tick_"))
    n_meta    = sum(1 for k in _cache if str(k).startswith("meta_"))
    if n_tickers == 0:
        return html.Small("⚪ no cache yet",
                          style={"color":"#444","fontSize":"10px"})
    return html.Small(
        f"🟢 {n_tickers:,} tickers · {n_meta:,} names cached",
        style={"color":"#2e7d32","fontSize":"10px"})

@app.callback(
    Output("vix-val","children"), Output("fg-val","children"),
    Output("fg-label","children"), Output("regime-badge","children"),
    Input("live-interval","n_intervals"),
)
def update_live(_):
    vix = get_live_vix()
    fg, lbl = get_fg_index()
    rm = 1.0 + ((vix/20) * ((100-fg)/50))
    gauge = "🔴" if fg<=25 else "🟠" if fg<=44 else "🟡" if fg<=55 else "🟢" if fg<=75 else "💚"
    if   rm > 2.5: regime = dbc.Alert(f"🔥 Extreme Fear — {rm:.1f}x", color="danger",  className="py-1 px-2 mb-0")
    elif rm > 2.0: regime = dbc.Alert(f"😰 High Fear — {rm:.1f}x",    color="danger",  className="py-1 px-2 mb-0")
    elif rm < 1.2: regime = dbc.Alert(f"😌 Calm — {rm:.1f}x",         color="info",    className="py-1 px-2 mb-0")
    else:          regime = dbc.Alert(f"⚖️ Normal — {rm:.1f}x",        color="warning", className="py-1 px-2 mb-0")
    return f"{vix:.1f}", f"{gauge} {fg}", lbl, regime

# ───────────────────────────────────────────────────────────────────
# CALLBACKS — Show/hide filter sections
# ───────────────────────────────────────────────────────────────────

@app.callback(
    Output("types-row","style"),
    Output("etf-filters-row","style"),
    Output("stock-filters-row","style"),
    Output("filter-sector","options"),   # only sector cascades dynamically
    Input("preset-dd","value"),
    Input("filter-types","value"),
    Input("filter-country","value"),
)
def update_filter_sections(preset, types, selected_country):
    ptype   = PRESETS.get(preset,{}).get("type","ETF")
    custom  = ptype == "custom"
    etfs_on   = (custom and "ETF"   in (types or [])) or ptype == "ETF"
    stocks_on = (custom and "Stock" in (types or [])) or ptype == "Stock"

    types_style = {} if custom else {"display":"none"}
    etf_style   = {} if etfs_on   else {"display":"none"}
    stock_style = {} if stocks_on else {"display":"none"}

    # Use module-level jetf_df directly — capture at call time
    _jdf = jetf_df
    _univ = universe

    def jcol_opts(col):
        if _jdf is None or _jdf.empty or col not in _jdf.columns:
            return []
        vals = sorted([v for v in _jdf[col].astype(str).str.strip().unique()
                       if v and v not in ("", "nan", "None", "nan%")])
        return [{"label":v,"value":v} for v in vals]

    def uopts(col, filter_type=None, country_filter=None):
        if _univ is None or _univ.empty:
            return []
        df = _univ[_univ["type"]==filter_type] if filter_type else _univ
        if country_filter and "country" in df.columns:
            df = df[df["country"].isin(country_filter)]
        if col not in df.columns:
            return []
        vals = sorted([v for v in df[col].astype(str).str.strip().unique()
                       if v and v not in ("", "nan", "None")])
        return [{"label":v,"value":v} for v in vals]

    dom_opts   = jcol_opts("domicile")
    dist_opts  = jcol_opts("dist_policy")
    repl_opts  = jcol_opts("replication")
    strat_opts = jcol_opts("strategy")

    # Hard fallbacks for when package doesn't return these fields
    if not dist_opts:
        dist_opts = [{"label":"Accumulating","value":"Accumulating"},
                     {"label":"Distributing", "value":"Distributing"}]
    if not dom_opts:
        dom_opts = [{"label":v,"value":v} for v in
                    ["Ireland","Luxembourg","Germany","France","United States"]]
    if not repl_opts:
        repl_opts = [{"label":v,"value":v} for v in
                     ["Physical (Full)","Physical (Sampling)","Swap-based"]]

    cat_opts  = uopts("category_group", "ETF")
    if not cat_opts:
        cat_opts = [{"label":v,"value":v} for v in [
            "Equities","Fixed Income","Commodities","Real Estate",
            "Alternatives","Cash","Currencies","Derivatives"]]

    ctry_opts = uopts("country", "Stock")
    sect_opts = uopts("sector", "Stock", selected_country)

    return types_style, etf_style, stock_style, sect_opts

@app.callback(
    Output("scope-label","children"),
    Input("preset-dd","value"),
    Input("filter-types","value"),
    Input("filter-domicile","value"),
    Input("filter-dist","value"),
    Input("filter-replication","value"),
    Input("filter-strategy","value"),
    Input("filter-category","value"),
    Input("filter-country","value"),
    Input("filter-sector","value"),
    Input("filter-minsize","value"),
    Input("filter-maxter","value"),
)
def update_scope(preset, types, domicile, dist, repl, strategy, category,
                 country, sector, minsize, maxter):
    filters = dict(types=types or [], domicile=domicile or [], dist_policy=dist or [],
                   replication=repl or [], strategy=strategy or [], category=category or [],
                   country=country or [], sector=sector or [],
                   min_size=minsize or 0, max_ter=maxter or 2.0)
    tickers = build_tickers(preset, filters)
    active  = []
    if domicile:  active.append(f"Domicile: {', '.join(domicile)}")
    if dist:      active.append(f"Dist: {', '.join(dist)}")
    if country:   active.append(f"Country: {', '.join(country[:2])}")
    if sector:    active.append(f"Sector: {', '.join(sector[:2])}")
    label = f"{len(tickers):,} tickers in scope"
    if active:
        label += " · " + " · ".join(active)
    # Clear previous scan results when preset/filters change
    return label

# ───────────────────────────────────────────────────────────────────
# CALLBACKS — Scan
# ───────────────────────────────────────────────────────────────────

@app.callback(
    Output("scan-store","data"),
    Output("scan-status-alert","children"),
    Output("scan-status-alert","style"),
    Output("run-btn","disabled"),
    Output("scan-running","data"),
    Input("run-btn","n_clicks"),
    Input("clear-btn","n_clicks"),
    Input("stop-btn","n_clicks"),
    Input("overlay-stop-btn","n_clicks"),
    State("preset-dd","value"),
    State("filter-types","value"),
    State("filter-domicile","value"),
    State("filter-dist","value"),
    State("filter-replication","value"),
    State("filter-strategy","value"),
    State("filter-category","value"),
    State("filter-country","value"),
    State("filter-sector","value"),
    State("filter-minsize","value"),
    State("filter-maxter","value"),
    State("workers-slider","value"),
    State("fetch-pe-check","value"),
    State("budget-input","value"),
    prevent_initial_call=True,
)
def run_scan(run_clicks, clear_clicks, stop_clicks, overlay_stop_clicks, preset, types, domicile, dist,
             repl, strategy, category, country, sector,
             minsize, maxter, workers, fetch_pe, budget):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update, no_update, no_update
    triggered = ctx.triggered[0]["prop_id"]

    if "clear-btn" in triggered:
        # Clear render cache too
        with _cache_lock:
            render_keys = [k for k in list(_cache.keys()) if str(k).startswith("render_")]
            for k in render_keys:
                _cache.pop(k, None)
        return None, "Results cleared.", {"display":"none"}, False, False

    if "stop-btn" in triggered or "overlay-stop-btn" in triggered:
        for sid in list(_active_scans.keys()):
            _active_scans[sid] = False
        cache_set("current_scan_id", "stopped")
        return no_update, "⏹ Scan stopped.", {"display":"block"}, False, False

    if "run-btn" not in triggered:
        return no_update, no_update, no_update, no_update, no_update

    vix = get_live_vix()
    fg, _ = get_fg_index()
    risk_mult = 1.0 + ((vix/20) * ((100-fg)/50))

    filters = dict(
        types=types or [], domicile=domicile or [], dist_policy=dist or [],
        replication=repl or [], strategy=strategy or [], category=category or [],
        country=country or [], sector=sector or [],
        min_size=minsize or 0, max_ter=maxter or 2.0,
    )
    tickers = build_tickers(preset, filters)
    print(f"[scan] {len(tickers)} tickers, preset={preset}", flush=True)
    if not tickers:
        return None, "⚠️ No tickers match filters.", {}, False, False

    # ── FAST PATH: use pre-computed signals.csv if available ──────────
    if not signals_df.empty:
        sig = signals_df[signals_df["ticker"].isin(tickers)].copy()
        if len(sig) >= 10:  # any meaningful coverage → use signals
            print(f"[scan] Using pre-computed signals: {len(sig)}/{len(tickers)} covered", flush=True)
            budget_val = budget or 1000
            rows = []
            for _, r in sig.iterrows():
                action = r["action"]
                conf   = float(r.get("conf", 0.5))
                vol    = float(r.get("vol_pct", 2))
                dm     = float(r.get("dist_ma200", 0))
                rsi_v  = float(r.get("rsi", 50))
                if action == "AVOID":  alloc = "⛔ Skip"
                elif action == "WAIT": alloc = "—"
                elif action == "WATCH":
                    amt = budget_val * 0.25 * (0.5 + conf)
                    alloc = f"👀 €{amt:,.0f}"
                else:
                    ctx_s = (40 if fg<35 else 20 if fg<50 else 0)/100
                    sig_s = min(-dm/20,1)*0.5 + min((50-rsi_v)/50,1)*0.5 if dm<0 else 0
                    amt   = budget_val*(ctx_s+sig_s)*(0.5+conf)*max(0.5,1-vol/20)*risk_mult
                    amt   = max(budget_val*0.25, min(amt, budget_val*3))
                    tier  = "🔥" if amt>=budget_val*1.5 else "⚖️" if amt>=budget_val*0.8 else "🔍"
                    alloc = f"{tier} €{amt:,.0f}"
                # Fundamentals from signals.csv (pre-computed by OpenBB)
                pe   = r.get("pe_ratio")
                div  = r.get("div_yield")
                mcap = r.get("market_cap")
                beta = r.get("beta")
                src  = r.get("data_source","")
                # Format momentum returns if price data unavailable
                rsi_display  = r.get("rsi",50)  if pd.notna(r.get("rsi",None)) else "—"
                macd_display = ("▲ Bull" if r.get("macd_bull",0) else "▼ Bear") if pd.notna(r.get("macd_bull",None)) else "—"
                dist_display = r.get("dist_ma200",0) if pd.notna(r.get("dist_ma200",None)) else None
                # For justETF momentum signals, use ret_1m as proxy
                if dist_display is None and pd.notna(r.get("ret_1m",None)):
                    dist_display = r.get("ret_1m",0)
                rows.append({
                    "Ticker":  r["ticker"], "Name": r.get("name",""), "ISIN": r.get("isin",""),
                    "Price":   round(float(r["price"]),2) if pd.notna(r.get("price",None)) else "—",
                    "MA200":   round(float(r["ma200"]),2) if pd.notna(r.get("ma200",None)) and str(r.get("ma200","")) not in ("","nan","None") else "—",
                    "Dist%":   round(float(dist_display),2) if dist_display is not None else 0,
                    "52W%":    r.get("dist_52w",0) if pd.notna(r.get("dist_52w",None)) else "—",
                    "RSI":     rsi_display,
                    "RSI↗":   "↗" if r.get("rsi_rising",0) else "↘",
                    "MACD":    macd_display,
                    "MACD⚡":  "⚡" if r.get("macd_accel",0) else "—",
                    "Vol%":    r.get("vol_pct",0) if pd.notna(r.get("vol_pct",None)) else "—",
                    "Conf":    r.get("conf",0),
                    "Action":  action,
                    "Signal":  f"{action} ({'Strong' if conf>0.7 else 'Medium' if conf>0.4 else 'Weak'})",
                    "Knife":   "⚠️" if r.get("is_knife",0) and not r.get("reversal",0) else "",
                    "Score":   r.get("score",0), "Allocation": alloc,
                    "Source":  src,
                    "PE":      f"{pe:.1f}" if pe else "—",
                    "Beta":    f"{beta:.2f}" if beta else "—",
                    "Div%":    f"{div*100:.1f}%" if div else "—",
                    "MCap":    f"${mcap/1e9:.1f}B" if mcap else "—",
                })
            result_df = pd.DataFrame(rows)
            # Penalise momentum-only signals — rank them below price-data signals
            # Sort: by action priority, then price-data before momentum, then score
            action_order = {"BUY":0,"WATCH":1,"SELL":2,"AVOID":3,"WAIT":4}
            result_df["_action_n"]  = result_df["Action"].map(action_order).fillna(5)
            result_df["_has_price"] = result_df["Price"].apply(
                lambda x: 0 if str(x) in ("—","nan","None","") or x is None else 1)
            result_df = (result_df.sort_values(
                            ["_action_n","_has_price","Score"],
                            ascending=[True, False, False])
                                  .drop_duplicates(subset=["Ticker"], keep="first")
                                  .drop(columns=["_action_n","_has_price"])
                                  .reset_index(drop=True))
            result_df.insert(0, "Rank", result_df.index+1)
            # Pre-compute per-action rank for instant tab switching
            result_df["ActionRank"] = result_df.groupby("Action").cumcount() + 1
            computed = sig["computed_at"].iloc[0] if "computed_at" in sig.columns else "unknown"
            status = (f"⚡ {len(result_df)} pre-computed signals · "
                      f"BUY:{(result_df['Action']=='BUY').sum()} "
                      f"WATCH:{(result_df['Action']=='WATCH').sum()} "
                      f"SELL:{(result_df['Action']=='SELL').sum()} "
                      f"AVOID:{(result_df['Action']=='AVOID').sum()} "
                      f"· updated: {computed}")
            return (result_df.to_json(date_format="iso", orient="split"),
                    status, {"display":"block"}, False, False)
    # ── END FAST PATH — fall through to live yfinance scan ────────────
    ptype = PRESETS.get(preset, {}).get("type", "ETF")
    if ptype == "Stock":
        MAX_PER_SCAN = 500    # stocks: signals.csv is primary, live is fallback
    elif ptype == "ETF":
        MAX_PER_SCAN = 500    # ETFs still need suffix discovery
    else:
        MAX_PER_SCAN = 1000   # custom mix
    if len(tickers) > MAX_PER_SCAN:
        print(f"[scan] capping {len(tickers)} → {MAX_PER_SCAN} ({ptype})", flush=True)
        tickers = tickers[:MAX_PER_SCAN]

    # Parallel scan
    import uuid
    scan_id = str(uuid.uuid4())[:8]
    cache_set("current_scan_id", scan_id)
    _active_scans[scan_id] = True

    results = {}
    total = len(tickers)
    done  = 0
    cache_set(f"progress_{scan_id}", {"done": 0, "total": total, "valid": 0, "pct": 0}, ttl=300)

    with ThreadPoolExecutor(max_workers=int(workers or 6)) as ex:
        isin_map = {}
        if not jetf_df.empty and "ticker" in jetf_df.columns and "isin" in jetf_df.columns:
            isin_map = dict(zip(jetf_df["ticker"], jetf_df["isin"].fillna("")))
        futs = {ex.submit(analyse_ticker, t, risk_mult, isin_map.get(t,"")): t for t in tickers}
        for fut in as_completed(futs):
            if cache_get("current_scan_id") != scan_id or not _active_scans.get(scan_id, True):
                _active_scans.pop(scan_id, None)
                cache_set(f"progress_{scan_id}", None)
                return (no_update, "⏹ Scan cancelled.",
                        {"display":"block"}, False, False)
            t = futs[fut]
            try:
                r = fut.result()
                results[t] = r
            except Exception:
                results[t] = None
            done += 1
            valid_so_far = sum(1 for v in results.values() if v)
            pct = round(done / total * 100)
            if done % 50 == 0:
                print(f"[scan {scan_id}] {done}/{total} done, {valid_so_far} valid", flush=True)
            cache_set(f"progress_{scan_id}", {
                "done": done, "total": total, "valid": valid_so_far,
                "pct": pct, "ticker": t
            }, ttl=300)

    rows = [r for t in tickers if (r := results.get(t)) is not None]
    if not rows:
        return None, "❌ No data returned.", {}, False, False

    # Names from cache/universe — instant, no separate API calls needed
    valid_tickers = [r["Ticker"] for r in rows]
    name_map = {t: get_name_isin(t) for t in valid_tickers}

    # Optionally fetch PE
    pe_map = {}
    if "pe" in (fetch_pe or []):
        with ThreadPoolExecutor(max_workers=8) as ex:
            pfuts = {ex.submit(fetch_pe_data, t): t for t in valid_tickers}
            for f in as_completed(pfuts):
                t = pfuts[f]
                try: pe_map[t] = f.result()
                except Exception: pe_map[t] = {}

    # Build final rows
    budget = budget or 1000
    final = []
    for r in rows:
        t    = r["Ticker"]
        name, isin = name_map.get(t, (t,""))
        if not isin and not jetf_df.empty and "ticker" in jetf_df.columns:
            jm = jetf_df[jetf_df["ticker"]==t]
            if not jm.empty and "isin" in jm.columns:
                v = jm.iloc[0]["isin"]
                isin = str(v) if pd.notna(v) else ""
        pe   = pe_map.get(t, {})
        row  = {
            "Ticker": t, "Name": name, "ISIN": isin,
            **r,
            **{k: pe.get(k,"—") for k in ["PE","Beta","Div%","MCap"]},
        }
        # Allocation
        action = r["Action"]
        conf   = r["Conf"]
        vol    = r["Vol%"]
        dm     = r["Dist%"]
        rsi    = r["RSI"]
        if action == "AVOID":  alloc = "⛔ Skip"
        elif action == "WAIT": alloc = "—"
        elif action == "WATCH":
            amt = budget * 0.25 * (0.5 + conf)
            alloc = f"👀 €{amt:,.0f}"
        else:
            ctx_s = (40 if fg<35 else 20 if fg<50 else 0)/100
            sig_s = min(-dm/20,1)*0.5 + min((50-rsi)/50,1)*0.5 if dm<0 else 0
            amt   = budget*(ctx_s+sig_s)*(0.5+conf)*max(0.5,1-vol/20)*risk_mult
            amt   = max(budget*0.25, min(amt, budget*3))
            tier  = "🔥" if amt>=budget*1.5 else "⚖️" if amt>=budget*0.8 else "🔍"
            alloc = f"{tier} €{amt:,.0f}"
        row["Allocation"] = alloc
        final.append(row)

    df = pd.DataFrame(final).sort_values("Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index+1)

    status = (f"✅ {len(df)} valid / {len(tickers)} attempted · "
              f"BUY:{(df['Action']=='BUY').sum()} "
              f"WATCH:{(df['Action']=='WATCH').sum()} "
              f"SELL:{(df['Action']=='SELL').sum()} "
              f"AVOID:{(df['Action']=='AVOID').sum()}")

    _active_scans.pop(scan_id, None)
    return (df.to_json(date_format="iso", orient="split"),
            status, {"display":"block"}, False, False)

# ───────────────────────────────────────────────────────────────────
# CALLBACKS — Results display
# ───────────────────────────────────────────────────────────────────

@app.callback(
    Output("kpi-row","children"),
    Output("top-signals","children"),
    Output("results-table","children"),
    Input("scan-store","data"),
    Input("signal-tabs","active_tab"),
    prevent_initial_call=False,  # Must fire initially to show empty state
)
def render_results(store_data, active_tab):
    if not store_data:
        # Clean empty state — no spinner, no confusion
        return (
            [],
            [],
            html.Div([
                html.Div(style={"height":"60px"}),
                html.P("🔭 Select a preset and click Run Scan to begin.",
                       className="text-muted text-center mt-4",
                       style={"fontSize":"16px"}),
            ])
        )

    import io, hashlib
    # Cache parsed df — tab switches reuse it without re-parsing 4000-row JSON
    _ck = hashlib.md5(store_data.encode()).hexdigest()
    _cached = cache_get(f"render_{_ck}")
    if _cached is not None:
        df = _cached
    else:
        df = pd.read_json(io.StringIO(store_data), orient="split")
        cache_set(f"render_{_ck}", df, ttl=300)

    # KPIs
    kpis = dbc.Row([
        dbc.Col(kpi_card("Scanned",  len(df)),                              width=2),
        dbc.Col(kpi_card("🟢 BUY",  int((df["Action"]=="BUY").sum()),   "#00e676"), width=2),
        dbc.Col(kpi_card("👀 WATCH", int((df["Action"]=="WATCH").sum()), "#00bcd4"), width=2),
        dbc.Col(kpi_card("⛔ AVOID", int((df["Action"]=="AVOID").sum()), "#ff6d00"), width=2),
        dbc.Col(kpi_card("🔴 SELL",  int((df["Action"]=="SELL").sum()),  "#ff1744"), width=2),
        dbc.Col(kpi_card("🟡 WAIT",  int((df["Action"]=="WAIT").sum()),  "#ffd600"), width=2),
    ], className="g-2")

    # Top signals
    def top_card(action, label, color):
        sub = df[df["Action"]==action][["Rank","Ticker","Name","Dist%","RSI","Conf"]].head(8)
        if sub.empty:
            return dbc.Col(dbc.Card([dbc.CardHeader(label, style={"background":color,"color":"#000","fontWeight":"bold"}),
                                     dbc.CardBody(html.P("No signals", className="text-muted mb-0"))],
                                    style={"background":"#1e1e2e","border":f"1px solid {color}"}), width=4)
        tbl = dash_table.DataTable(
            data=sub.round(2).to_dict("records"),
            columns=[{"name":c,"id":c} for c in sub.columns],
            style_table={"overflowX":"auto"},
            style_cell={"background":"#1e1e2e","color":"#fff","border":"none","fontSize":"12px","padding":"4px 8px"},
            style_header={"background":"#12121f","color":"#aaa","fontWeight":"bold","border":"none"},
        )
        return dbc.Col(dbc.Card([
            dbc.CardHeader(label, style={"background":color,"color":"#000","fontWeight":"bold"}),
            dbc.CardBody(tbl, className="p-0"),
        ], style={"background":"#1e1e2e","border":f"1px solid {color}"}), width=4)

    top = dbc.Row([
        top_card("BUY",   "🟢 Top BUY",   "#00e676"),
        top_card("WATCH", "👀 WATCH",      "#00bcd4"),
        top_card("SELL",  "🔴 Top SELL",   "#ff1744"),
    ], className="g-2")

    # Full table — global rank, no re-ranking on tab switch
    sub = df if active_tab == "all" else df[df["Action"]==active_tab]

    SHOW_COLS = ["Rank","Ticker","Name","ISIN","Price","MA200","Dist%","52W%",
                 "RSI","RSI↗","MACD","MACD⚡","Vol%","Conf",
                 "PE","Beta","Div%","MCap",
                 "Signal","Knife","Allocation"]
    show_cols = [c for c in SHOW_COLS if c in sub.columns]

    # Round floats to avoid 2.8200000000000003 display issues
    sub = sub.copy()
    for col in ["Price","MA200","Dist%","52W%","RSI","Vol%","Conf","Score"]:
        if col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce").round(2)

    def style_signal(col_id, val):
        for action, color in SIGNAL_COLORS.items():
            if action in str(val):
                return color
        return None

    cond_styles = []
    for action, color in SIGNAL_COLORS.items():
        cond_styles.append({
            "if": {"filter_query": f'{{Signal}} contains "{action}"', "column_id": "Signal"},
            "color": color, "fontWeight": "bold",
        })

    tbl = dash_table.DataTable(
        id="main-table",
        data=sub[show_cols].round({"Price":2,"MA200":2,"Dist%":1,"52W%":1,"RSI":1,"Vol%":2,"Conf":2}).to_dict("records"),
        columns=[{"name":c,"id":c,"selectable": c=="Ticker"} for c in show_cols],
        row_selectable="single",
        selected_rows=[],
        sort_action="native",
        filter_action="native",
        page_size=25,
        style_table={"overflowX":"auto"},
        style_cell={
            "background":"#1a1a2e","color":"#fff",
            "border":"1px solid #2a2a3e",
            "fontSize":"12px","padding":"6px 10px",
            "whiteSpace":"nowrap","overflow":"hidden",
            "textOverflow":"ellipsis","maxWidth":"180px",
        },
        style_cell_conditional=[
            {"if":{"column_id":"Name"},  "maxWidth":"160px","minWidth":"120px"},
            {"if":{"column_id":"ISIN"},  "maxWidth":"110px","fontFamily":"monospace"},
            {"if":{"column_id":"Ticker"},"maxWidth":"70px","fontWeight":"bold"},
            {"if":{"column_id":"Signal"},"maxWidth":"130px"},
            {"if":{"column_id":"Allocation"},"maxWidth":"140px"},
        ],
        style_header={
            "background":"#12121f","color":"#00bcd4",
            "fontWeight":"bold","border":"1px solid #2a2a3e",
        },
        style_data_conditional=[
            {"if":{"row_index":"odd"},"background":"#161626"},
            *cond_styles,
            {"if":{"column_id":"Dist%","filter_query":"{Dist%} < 0"},"color":"#00e676"},
            {"if":{"column_id":"Dist%","filter_query":"{Dist%} > 0"},"color":"#ff6d00"},
            {"if":{"state":"selected"},"background":"#2a2a4e","border":"1px solid #00bcd4"},
        ],
        style_filter={"background":"#1a1a2e","color":"#fff","border":"1px solid #333"},
        tooltip_data=[{
            "Ticker": {"value": f"Click row then click 🔬 to deep dive", "type":"markdown"},
        } for _ in range(len(sub))],
    )

    # Row-click → deep dive button
    dive_btn = dbc.Button(
        "🔬 Deep Dive selected ticker →",
        id="goto-dive-btn",
        color="info", size="sm", className="mt-2",
        disabled=True,
    )

    table_section = html.Div([
        tbl,
        dbc.Row([
            dbc.Col(dive_btn, width="auto"),
            dbc.Col(html.Small("Select a row, then click to open in Deep Dive", className="text-muted mt-2"), width="auto"),
        ], className="mt-2 align-items-center"),
    ])

    return kpis, top, table_section

@app.callback(
    Output("goto-dive-btn","disabled"),
    Output("goto-dive-btn","children"),
    Input("main-table","selected_rows"),
    State("scan-store","data"),
)
def enable_dive_btn(selected_rows, store_data):
    if not selected_rows or not store_data:
        return True, "🔬 Deep Dive selected ticker →"
    df  = pd.read_json(store_data, orient="split")
    idx = selected_rows[0]
    if idx >= len(df):
        return True, "🔬 Deep Dive selected ticker →"
    ticker = df.iloc[idx]["Ticker"]
    return False, f"🔬 Deep Dive: {ticker} →"

@app.callback(
    Output("main-tabs","active_tab"),
    Output("dd-input","value"),
    Input("goto-dive-btn","n_clicks"),
    State("main-table","selected_rows"),
    State("scan-store","data"),
    prevent_initial_call=True,
)
def open_deep_dive(n_clicks, selected_rows, store_data):
    if not n_clicks or not selected_rows or not store_data:
        return no_update, no_update
    import io
    df     = pd.read_json(io.StringIO(store_data), orient="split")
    idx    = selected_rows[0]
    ticker = df.iloc[idx]["Ticker"]
    return "deepdive", ticker

@app.callback(
    Output("dd-from-scanner","children"),
    Input("dd-input","value"),
    State("main-tabs","active_tab"),
)
def show_scanner_context(ticker, tab):
    if tab == "deepdive" and ticker:
        return dbc.Alert(f"📡 **{ticker}** loaded from Scanner — click Analyse below.",
                         color="info", className="py-1")
    return ""

# ───────────────────────────────────────────────────────────────────
# CALLBACKS — Deep Dive
# ───────────────────────────────────────────────────────────────────

@app.callback(
    Output("dd-results","children"),
    Input("dd-btn","n_clicks"),
    State("dd-input","value"),
    State("dd-budget","value"),
    prevent_initial_call=True,
)
def run_deep_dive(n_clicks, user_input, budget):
    if not n_clicks or not user_input:
        return ""
    ticker = user_input.strip().upper()
    budget = budget or 1000

    isin = None
    def is_isin(x):
        return len(x)==12 and x[:2].isalpha() and x[2:].isalnum()

    if is_isin(ticker):
        isin = ticker
        resolved = False
        if not jetf_df.empty and "isin" in jetf_df.columns:
            match = jetf_df[jetf_df["isin"]==isin]
            if not match.empty:
                ticker = match.iloc[0]["ticker"]
                resolved = True
        if not resolved:
            try:
                sr = yf.Search(isin, max_results=5)
                if hasattr(sr,"quotes") and sr.quotes:
                    ticker = sr.quotes[0]["symbol"]
                    resolved = True
            except Exception:
                pass
        if not resolved:
            return dbc.Alert(
                f"Could not resolve ISIN {isin} to a ticker. Try the ticker symbol directly.",
                color="warning")

    # Look up correct yf_symbol from universe.csv FIRST
    resolved_yf = None
    if not universe.empty and "yf_symbol" in universe.columns:
        umatch = universe[universe["ticker"] == ticker]
        if not umatch.empty:
            sym = str(umatch.iloc[0].get("yf_symbol","")).strip()
            if sym and sym not in ("","nan","None"):
                resolved_yf = sym
            if not isin:
                isin = str(umatch.iloc[0].get("isin","")).strip() or None
    # Also check justETF for ISIN
    if not isin and not jetf_df.empty and "ticker" in jetf_df.columns:
        jmatch = jetf_df[jetf_df["ticker"] == ticker]
        if not jmatch.empty:
            isin = str(jmatch.iloc[0].get("isin","")).strip() or None

    # For ETFs with ISIN: use justETF chart as primary source (reliable EUR prices)
    # For stocks or ETFs without ISIN: fall back to yfinance via fetch_ticker_data
    raw = None
    is_etf = False
    if isin:
        # Check if this is an ETF (has justETF entry)
        if not jetf_df.empty and "isin" in jetf_df.columns:
            is_etf = not jetf_df[jetf_df["isin"] == isin].empty
        if is_etf:
            jetf_chart = fetch_justetf_chart(isin)
            if not jetf_chart.empty:
                # Convert justETF chart to same format as fetch_ticker_data output
                close   = jetf_chart["Close"].dropna()
                price   = float(close.iloc[-1])
                ma50_s  = close.rolling(50).mean()
                ma200_s = close.rolling(200).mean()
                ma200   = float(ma200_s.iloc[-1])
                dist_ma = ((price - ma200) / ma200) * 100
                delta   = close.diff()
                gain    = delta.where(delta>0,0).rolling(14).mean()
                loss    = (-delta.where(delta<0,0)).rolling(14).mean()
                rs      = gain / loss.replace(0,0.001)
                rsi_s   = 100 - (100/(1+rs))
                rsi_val2= float(rsi_s.iloc[-1])
                ema12   = close.ewm(span=12,adjust=False).mean()
                ema26   = close.ewm(span=26,adjust=False).mean()
                macd_l2 = ema12 - ema26
                macd_sig2 = macd_l2.ewm(span=9,adjust=False).mean()
                macd_h2 = macd_l2 - macd_sig2
                vol2    = float(close.pct_change().rolling(20).std().iloc[-1]*100)
                h52w    = float(close.tail(252).max())
                raw = {
                    "close": close, "ma50_s": ma50_s, "ma200_s": ma200_s,
                    "rsi_s": rsi_s, "macd_l": macd_l2, "macd_sig": macd_sig2,
                    "macd_h": macd_h2, "price": price, "dist_ma": dist_ma,
                    "rsi": rsi_val2, "rsi_slope": float(rsi_s.iloc[-1]-rsi_s.iloc[-2]),
                    "macd": float(macd_l2.iloc[-1]), "macd_signal": float(macd_sig2.iloc[-1]),
                    "macd_hist": float(macd_h2.iloc[-1]),
                    "confidence": min(abs(dist_ma)/20,1)*0.5 + min(abs(50-rsi_val2)/50,1)*0.5,
                    "vol": vol2, "dist_52h": ((price-h52w)/h52w)*100,
                    "ma200": ma200, "trend_down_strong": False,
                    "source": "justETF",
                }

    if raw is None:
        raw = fetch_ticker_data(resolved_yf or ticker, isin=isin)
    if raw is None:
        try:
            sr = yf.Search(ticker, max_results=3)
            if hasattr(sr,"quotes") and sr.quotes:
                sugg = ", ".join([q["symbol"] for q in sr.quotes[:3]])
                return dbc.Alert(f"No data for **{ticker}**. Try: {sugg}", color="warning")
        except Exception:
            pass
        return dbc.Alert(f"No data found for {ticker}.", color="danger")

    close    = raw["close"]
    ma50_s   = raw["ma50_s"]
    ma200_s  = raw["ma200_s"]
    rsi_s    = raw["rsi_s"]
    macd_l   = raw["macd_l"]
    macd_sig = raw["macd_sig"]
    macd_h   = raw["macd_h"]

    cur_p      = raw["price"]
    dist_ma    = raw["dist_ma"]
    rsi_val    = raw["rsi"]
    rsi_rising = raw["rsi_slope"] > 0
    macd_bull  = raw["macd"] > raw["macd_signal"]
    macd_accel = raw["macd_hist"] > 0
    conf       = raw["confidence"]
    vol        = raw["vol"]
    dist_52h   = raw["dist_52h"]
    is_knife   = raw["dist_ma"] < -15 and rsi_val < 35 and raw["trend_down_strong"]
    reversal   = is_knife and macd_bull and rsi_rising
    ma200_now  = raw["ma200"]
    vix        = get_live_vix()
    fg, fg_lbl = get_fg_index()

    # Buy/sell scores
    buy_score  = (40 if fg<35 else 0)+(30 if rsi_val<40 else 0)+(30 if dist_ma<0 else 0)
    sell_score = (40 if fg>65 else 0)+(30 if rsi_val>65 else 0)+(30 if dist_ma>0 else 0)

    if rsi_val<35 and dist_ma<0 and raw["trend_down_strong"]:
        entry="WAIT"
    elif rsi_rising and macd_bull:
        entry="TRIGGER"
    else:
        entry="WATCH"

    exit_s = "WAIT" if (rsi_val>65 and dist_ma>0) else ("TRIGGER" if (not rsi_rising and rsi_val>60) else "WATCH")

    # justETF metadata
    jetf_meta = {}
    name, isin_found = get_name_isin_full(ticker)
    isin = isin or isin_found
    if not jetf_df.empty and ticker in jetf_df["ticker"].values:
        jetf_meta = jetf_df[jetf_df["ticker"]==ticker].iloc[0].to_dict()

    # PE data
    pe_data = fetch_pe_data(ticker)

    # ── Price chart
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=close.index, y=close, name="Price", line=dict(color="#00bcd4",width=2)))
    fig_price.add_trace(go.Scatter(x=ma50_s.index, y=ma50_s, name="MA50", line=dict(color="#ffd600",width=1,dash="dash")))
    fig_price.add_trace(go.Scatter(x=ma200_s.index, y=ma200_s, name="MA200", line=dict(color="#ff6d00",width=1,dash="dash")))
    fig_price.update_layout(template="plotly_dark",paper_bgcolor="#1a1a2e",plot_bgcolor="#1a1a2e",
        margin=dict(l=0,r=0,t=30,b=0), height=280, title=f"{ticker} — {name}",
        legend=dict(orientation="h",y=1.1))

    # ── MACD + RSI
    fig_indicators = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5,0.5],
                                   vertical_spacing=0.05)
    fig_indicators.add_trace(go.Scatter(x=macd_l.index,   y=macd_l,   name="MACD",   line=dict(color="#00e676")), row=1, col=1)
    fig_indicators.add_trace(go.Scatter(x=macd_sig.index, y=macd_sig, name="Signal", line=dict(color="#ff6d00")), row=1, col=1)
    fig_indicators.add_trace(go.Bar(x=macd_h.index, y=macd_h, name="Hist",
        marker_color=["#00e676" if v>=0 else "#ff1744" for v in macd_h]), row=1, col=1)
    fig_indicators.add_trace(go.Scatter(x=rsi_s.index, y=rsi_s, name="RSI", line=dict(color="#00bcd4")), row=2, col=1)
    fig_indicators.add_hline(y=30, line_color="#00e676", line_dash="dash", row=2, col=1)
    fig_indicators.add_hline(y=70, line_color="#ff1744", line_dash="dash", row=2, col=1)
    fig_indicators.update_layout(template="plotly_dark",paper_bgcolor="#1a1a2e",plot_bgcolor="#1a1a2e",
        margin=dict(l=0,r=0,t=10,b=0), height=260,
        legend=dict(orientation="h",y=1.05))

    # ── Buy decision
    if is_knife and not reversal:
        buy_el = dbc.Alert("⛔ AVOID — Falling knife. Wait for reversal.", color="danger")
    elif entry=="WAIT":
        buy_el = dbc.Alert("⏳ WAIT — Still falling. Let it stabilise.", color="warning")
    elif entry=="WATCH":
        if buy_score>=70:
            buy_el = dbc.Alert(f"⚖️ PREPARE — Good setup (~€{budget:,.0f})", color="info")
        else:
            buy_el = dbc.Alert("🔍 WATCH — No strong entry yet.", color="secondary")
    else:
        if buy_score>=70:   buy_el = dbc.Alert(f"🔥 BUY — €{budget*2:,.0f}", color="success")
        elif buy_score>=40: buy_el = dbc.Alert(f"⚖️ BUY — €{budget:,.0f}", color="success")
        else:               buy_el = dbc.Alert(f"⚠️ LIGHT — €{budget*0.5:,.0f}", color="warning")
    if reversal:
        buy_el = html.Div([buy_el, dbc.Alert("✅ Reversal confirmed (MACD + RSI turned bullish)", color="success")])

    if sell_score>=70:   sell_el = dbc.Alert("🚨 STRONG SELL — Reduce significantly.", color="danger")
    elif sell_score>=40: sell_el = dbc.Alert("⚠️ TRIM — Partial reduction.", color="warning")
    else:                sell_el = dbc.Alert("🟢 HOLD — No sell signal.", color="success")

    # ── Metrics
    metrics = dbc.Row([
        dbc.Col(kpi_card("Price",      f"${cur_p:.2f}"), width=2),
        dbc.Col(kpi_card("vs MA200",   f"{dist_ma:+.1f}%",  "#00e676" if dist_ma<0 else "#ff6d00"), width=2),
        dbc.Col(kpi_card("RSI",        f"{rsi_val:.1f} {'↗' if rsi_rising else '↘'}"), width=2),
        dbc.Col(kpi_card("MACD",       "▲ Bull" if macd_bull else "▼ Bear", "#00e676" if macd_bull else "#ff6d00"), width=2),
        dbc.Col(kpi_card("52W High",   f"{dist_52h:+.1f}%"), width=2),
        dbc.Col(kpi_card("Conf",       f"{conf:.2f}"), width=2),
    ], className="g-2 mb-3")

    # justETF strip
    meta_parts = []
    if jetf_meta.get("domicile"):     meta_parts.append(f"🏳️ {jetf_meta['domicile']}")
    if jetf_meta.get("ter"):          meta_parts.append(f"💸 TER {jetf_meta['ter']:.2f}%")
    if jetf_meta.get("fund_size_eur"):meta_parts.append(f"📦 €{jetf_meta['fund_size_eur']:,.0f}m")
    if jetf_meta.get("dist_policy"):  meta_parts.append(f"💰 {jetf_meta['dist_policy']}")
    if jetf_meta.get("replication"):  meta_parts.append(f"🔄 {jetf_meta['replication']}")
    meta_strip = html.P("  ·  ".join(meta_parts), className="text-muted small") if meta_parts else None

    # Fundamentals strip
    fund_parts = []
    for k, v in pe_data.items():
        if v and v != "—":
            fund_parts.append(f"{k}: {v}")
    fund_strip = html.P("  ·  ".join(fund_parts), className="text-muted small") if fund_parts else None

    links = html.Div([
        dbc.Button("📈 Yahoo Finance",
            href=f"https://finance.yahoo.com/quote/{ticker}",
            target="_blank", color="link", size="sm"),
        dbc.Button("📊 ETF.com",
            href=f"https://www.etf.com/{ticker}",
            target="_blank", color="link", size="sm"),
        dbc.Button("🔍 justETF (ISIN)" if isin else "🔍 justETF",
            href=f"https://www.justetf.com/en/etf-profile.html?isin={isin}"
                 if isin else f"https://www.justetf.com/en/search.html?query={ticker}",
            target="_blank", color="link", size="sm"),
        *([ html.Code(isin, style={"marginLeft":"8px","color":"#aaa","fontSize":"11px",
            "background":"#1e1e2e","padding":"2px 6px","borderRadius":"4px"})
           ] if isin else []),
    ], className="mb-2")

    # Signal detail table
    price_chg = float((close.iloc[-1]-close.iloc[-2])/close.iloc[-2]*100)
    thr = max(1.0, vol*1.5)
    detail_df = pd.DataFrame([
        {"Indicator":"vs MA200",     "Value":f"{dist_ma:+.1f}%",   "Status":"🟢 Below" if dist_ma<0 else "🔴 Above"},
        {"Indicator":"RSI",          "Value":f"{rsi_val:.1f}",      "Status":"↗ Rising·Oversold" if (rsi_rising and rsi_val<30) else "↗ Rising" if rsi_rising else "↘ Falling"},
        {"Indicator":"MACD",         "Value":"▲ Bull" if macd_bull else "▼ Bear", "Status":"⚡ Accel" if macd_accel else "Flat"},
        {"Indicator":"1d Change",    "Value":f"{price_chg:+.2f}%", "Status":f"vs trigger {thr:.1f}%"},
        {"Indicator":"52W High",     "Value":f"{dist_52h:+.1f}%",  "Status":"—"},
        {"Indicator":"Volatility",   "Value":f"{vol:.1f}%",         "Status":"—"},
        {"Indicator":"Falling Knife","Value":"⚠️ YES" if is_knife else "✅ No", "Status":"✅ Reversal" if reversal else ""},
    ])
    detail_tbl = dash_table.DataTable(
        data=detail_df.to_dict("records"),
        columns=[{"name":c,"id":c} for c in detail_df.columns],
        style_table={"overflowX":"auto"},
        style_cell={"background":"#1a1a2e","color":"#fff","border":"1px solid #2a2a3e","fontSize":"12px","padding":"6px"},
        style_header={"background":"#12121f","color":"#00bcd4","fontWeight":"bold","border":"1px solid #2a2a3e"},
    )

    return html.Div([
        html.Hr(style={"borderColor":"#333"}),
        metrics,
        links,
        meta_strip,
        fund_strip,
        dbc.Row([
            dbc.Col([html.H6("📥 Buy Signal"), buy_el], width=6),
            dbc.Col([html.H6("📤 Sell Signal"), sell_el], width=6),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_price, config={"displayModeBar":False}), width=12),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_indicators, config={"displayModeBar":False}), width=12),
        ], className="mb-3"),
        dbc.Accordion([dbc.AccordionItem(detail_tbl, title="⏱ Full Signal Detail")], start_collapsed=True),
    ])

@app.callback(
    Output("run-btn","children"),
    Input("run-btn","disabled"),
)
def update_run_btn_label(is_disabled):
    if is_disabled:
        return [dbc.Spinner(size="sm", color="light",
                            style={"marginRight":"8px"}), "Scanning…"]
    return "🔄 Run Scan"


@app.callback(
    Output("filter-domicile","disabled"),
    Output("filter-dist","disabled"),
    Output("filter-replication","disabled"),
    Output("filter-strategy","disabled"),
    Output("filter-category","disabled"),
    Output("filter-country","disabled"),
    Output("filter-sector","disabled"),
    Output("preset-dd","disabled"),
    Input("run-btn","disabled"),
    prevent_initial_call=True,
)
def disable_filters_during_scan(btn_disabled):
    return [btn_disabled]*8

@app.callback(
    Output("scan-overlay","style"),
    Output("overlay-progress-bar","value"),
    Output("overlay-progress-text","children"),
    Input("progress-interval","n_intervals"),
    prevent_initial_call=True,
)
def update_overlay(_):
    hidden  = {"display":"none"}
    visible = {
        "display":"flex","position":"fixed","top":0,"left":0,
        "width":"100%","height":"100%",
        "background":"rgba(0,0,0,0.85)","zIndex":9999,
    }

    scan_id = cache_get("current_scan_id")
    if not scan_id or scan_id in ("stopped",""):
        return hidden, no_update, no_update

    prog = cache_get(f"progress_{scan_id}")
    if not prog:
        return visible, 0, "Starting…"

    pct   = prog.get("pct", 0)
    done  = prog.get("done", 0)
    total = prog.get("total", 1)
    valid = prog.get("valid", 0)

    if pct >= 100:
        return hidden, no_update, no_update

    txt = f"{done:,} / {total:,} checked  ·  {valid} valid so far"
    return visible, pct, txt

@app.callback(
    Output("dl-csv","data"),
    Input("dl-btn","n_clicks"),
    State("scan-store","data"),
    prevent_initial_call=True,
)
def download_csv(n, store_data):
    if not store_data:
        return no_update
    import io
    df = pd.read_json(io.StringIO(store_data), orient="split")
    return dcc.send_data_frame(df.to_csv, "scan_results.csv", index=False)

import os

@server.route("/clear-cache")
def clear_cache():
    """Manually clear all scan caches — call from browser to reset."""
    import json
    with _cache_lock:
        keys_cleared = [k for k in list(_cache.keys())
                       if str(k).startswith(("tick_","sfx_","resolved_",
                                             "progress_","current_scan",
                                             "render_","dd_"))]
        for k in keys_cleared:
            _cache.pop(k, None)
    return f"<pre>Cleared {len(keys_cleared)} keys. Refresh app now.</pre>"

@server.route("/scan-status")
def scan_status():
    """Live scan progress check."""
    import json
    scan_id = cache_get("current_scan_id")
    prog = cache_get(f"progress_{scan_id}") if scan_id else None
    failed_count = sum(1 for k in _cache if str(k).startswith("tick_")
                       and cache_get(k) == "FAILED")
    valid_count  = sum(1 for k in _cache if str(k).startswith("tick_")
                       and cache_get(k) not in (None, "FAILED"))
    return f"<pre>{json.dumps({'scan_id': scan_id, 'progress': prog, 'cached_valid': valid_count, 'cached_failed': failed_count}, indent=2)}</pre>"

@server.route("/debug")
def debug_info():
    """Quick health check — shows what data is loaded."""
    import json
    u_size  = len(universe) if not universe.empty else 0
    j_size  = len(jetf_df)  if not jetf_df.empty  else 0
    etf_u   = universe[universe["type"]=="ETF"]  if not universe.empty else pd.DataFrame()
    stk_u   = universe[universe["type"]=="Stock"] if not universe.empty else pd.DataFrame()

    cat_vals = []
    if not etf_u.empty and "category_group" in etf_u.columns:
        cat_vals = [v for v in etf_u["category_group"].astype(str).str.strip().unique()
                    if v and v not in ("","nan","None")][:20]

    ctry_vals = []
    if not stk_u.empty and "country" in stk_u.columns:
        ctry_vals = [v for v in stk_u["country"].astype(str).str.strip().unique()
                     if v and v not in ("","nan","None")][:10]

    # justETF column inspection
    jetf_cols = list(jetf_df.columns) if not jetf_df.empty else []
    jetf_dist_vals = []
    jetf_dom_vals  = []
    if not jetf_df.empty:
        for col in jetf_df.columns:
            sample = [v for v in jetf_df[col].dropna().astype(str).unique() if v.strip()][:5]
            if any(x in col.lower() for x in ["dist","policy","accum","div"]):
                jetf_dist_vals = [(col, sample)]
            if "domicile" in col.lower():
                jetf_dom_vals = [(col, sample)]

    # Sample justETF tickers for IE/LU domicile
    jetf_sample = []
    if not jetf_df.empty and "domicile" in jetf_df.columns:
        sample = jetf_df[jetf_df["domicile"].isin(["Ireland","Luxembourg"])].head(20)
        jetf_sample = sample[["ticker","isin","jname"]].to_dict("records") if not sample.empty else []

    info = {
        "universe_rows":     u_size,
        "jetf_ie_lu_sample": jetf_sample,
        "etf_rows":          len(etf_u),
        "stock_rows":        len(stk_u),
        "justetf_rows":      j_size,
        "justetf_available": JUSTETF_AVAILABLE,
        "justetf_columns":   jetf_cols,
        "justetf_dist_col":  jetf_dist_vals,
        "justetf_dom_col":   jetf_dom_vals,
        "etf_category_group_sample": cat_vals,
        "stock_country_sample":      ctry_vals,
        "name_lookup_size":  len(_name_lookup),
        "cache_keys":        list(_cache.keys())[:20],
    }
    return f"<pre>{json.dumps(info, indent=2)}</pre>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
