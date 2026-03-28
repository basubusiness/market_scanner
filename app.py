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
    df = df.drop_duplicates(subset=["ticker","type"], keep="first")
    cache_set("universe", df)  # permanent cache
    return df

def load_justetf():
    cached = cache_get("justetf")
    if cached is not None:
        return cached
    if not JUSTETF_AVAILABLE:
        return pd.DataFrame()
    try:
        df = justetf_scraping.load_overview().reset_index()
        rename = {
            "ticker": "ticker", "isin": "isin", "name": "jname",
            "domicile_country": "domicile", "ter": "ter",
            "distribution_policy": "dist_policy",
            "fund_size_eur": "fund_size_eur",
            "replication": "replication", "strategy": "strategy",
        }
        keep = {k: v for k, v in rename.items() if k in df.columns}
        df = df[list(keep.keys())].rename(columns=keep)
        df["ticker"] = df["ticker"].fillna("").astype(str).str.strip().str.upper()
        df = df[df["ticker"].str.match(r"^[A-Z]{1,5}$")]
        df = df.drop_duplicates(subset=["ticker"], keep="first")
        cache_set("justetf", df, ttl=86400*7)
        return df
    except Exception:
        return pd.DataFrame()

# Load on startup (background thread so server starts fast)
# Load synchronously — universe must be ready before any callback runs
universe = load_base_universe()
jetf_df  = load_justetf()

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

def fetch_ticker_data(ticker):
    key = f"tick_{ticker}"
    cached = cache_get(key)
    if cached is not None:
        return cached
    try:
        df    = flatten_df(yf.Ticker(ticker).history(period="1y", auto_adjust=True))
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"].dropna()
        if len(close) < 30:
            return None
        price = float(close.iloc[-1])
        if price < 0.50:
            return None
        if "Volume" in df.columns:
            if df["Volume"].dropna().tail(20).mean() < 1000:
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

def analyse_ticker(ticker, risk_mult):
    raw = fetch_ticker_data(ticker)
    if raw is None:
        return None
    dm, rsi, conf = raw["dist_ma"], raw["rsi"], raw["confidence"]
    macd_bull  = raw["macd"] > raw["macd_signal"]
    macd_accel = raw["macd_hist"] > 0
    rsi_rising = raw["rsi_slope"] > 0
    knife_thr  = -15 * (1 / risk_mult)
    is_knife   = (dm < knife_thr) and (rsi < 35) and raw["trend_down_strong"]
    reversal   = is_knife and macd_bull and rsi_rising
    if is_knife and not reversal:        action = "AVOID"
    elif dm < -10 and rsi < 40 and macd_bull and macd_accel: action = "BUY"
    elif dm < -10 and rsi < 40 and (macd_bull or rsi_rising): action = "WATCH"
    elif dm > 10 and rsi > 70:           action = "SELL"
    else:                                action = "WAIT"
    strength = "Strong" if conf > 0.7 else "Medium" if conf > 0.4 else "Weak"
    dist_s = min(-dm/30,1) if dm < 0 else 0
    rsi_s2 = max((50-rsi)/50, 0)
    score  = dist_s*0.30 + rsi_s2*0.25 + (0.20 if macd_bull else 0) + (0.10 if macd_accel else 0) + conf*0.15
    if action == "AVOID": score -= 2
    return {
        "Ticker": ticker, "Price": round(raw["price"],2),
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
        e = universe[universe["type"]=="ETF"].copy()
        cat = preset.get("category_group") or filters.get("category",[])
        if cat:
            e = e[e["category_group"].isin(cat)]
        # justETF filters
        need_jetf = (
            "domicile" in preset or filters.get("domicile") or filters.get("dist_policy") or
            filters.get("replication") or filters.get("strategy") or
            filters.get("min_size",0) > 0 or filters.get("max_ter",2.0) < 2.0
        )
        if need_jetf and not jetf_df.empty:
            jcols = [c for c in ["ticker","domicile","dist_policy","fund_size_eur","replication","strategy","ter"] if c in jetf_df.columns]
            e = e.merge(jetf_df[jcols], on="ticker", how="left")
            dom = preset.get("domicile") or filters.get("domicile",[])
            if dom and "domicile" in e.columns: e = e[e["domicile"].isin(dom)]
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
        parts.append(e)

    # Stock branch
    stock_types = ["Stock"] if ptype == "Stock" else (filters.get("types",[]) if ptype=="custom" else [])
    if "Stock" in stock_types or ptype == "Stock":
        s = universe[universe["type"]=="Stock"].copy()
        country = preset.get("country") or filters.get("country",[])
        if country: s = s[s["country"].isin(country)]
        if filters.get("sector"): s = s[s["sector"].isin(filters["sector"])]
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

def sidebar():
    def_jcol = lambda col: (sorted([v for v in jetf_df[col].dropna().unique() if str(v).strip()])
                             if not jetf_df.empty and col in jetf_df.columns else [])
    country_opts = sorted([c for c in universe[universe["type"]=="Stock"]["country"].unique() if c]) if not universe.empty else []
    sector_opts  = sorted([s for s in universe[universe["type"]=="Stock"]["sector"].unique()  if s]) if not universe.empty else []
    cat_opts     = sorted([c for c in universe[universe["type"]=="ETF"]["category_group"].unique() if c]) if not universe.empty else []

    return html.Div([
        html.H5("📡 Market Decision Engine", className="text-white mb-3"),
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
                    dcc.Dropdown(id="filter-domicile", options=[{"label":v,"value":v} for v in def_jcol("domicile")],
                        multi=True, placeholder="Any", style=_DD, className="dash-dark-dd"),
                    html.Label("Distribution", className="text-white small"),
                    dcc.Dropdown(id="filter-dist", options=[{"label":v,"value":v} for v in def_jcol("dist_policy")],
                        multi=True, placeholder="Any", style=_DD, className="dash-dark-dd"),
                    html.Label("Replication", className="text-white small"),
                    dcc.Dropdown(id="filter-replication", options=[{"label":v,"value":v} for v in def_jcol("replication")],
                        multi=True, placeholder="Any", style=_DD, className="dash-dark-dd"),
                    html.Label("Strategy", className="text-white small"),
                    dcc.Dropdown(id="filter-strategy", options=[{"label":v,"value":v} for v in def_jcol("strategy")],
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
        ], start_collapsed=True, className="mb-3"),

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
                [html.Span("🔄 Run Scan", id="run-btn-text"),
                 dbc.Spinner(size="sm", spinner_style={"marginLeft":"8px"}, id="run-spinner",
                             color="light", show_initially=False)],
                id="run-btn", color="danger", size="lg", className="w-100"
            ), width=10),
            dbc.Col(dbc.Button("🗑️ Clear", id="clear-btn", color="secondary",
                               size="lg", className="w-100"), width=2),
        ], className="mb-3"),

        # Scan status bar
        html.Div(id="scan-status-bar", children=[
            dbc.Alert(id="scan-status-alert", color="info",
                      className="py-2 px-3 mb-2", style={"display":"none"}),
        ]),

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
    # Dynamically update filter options based on preset + current selections
    Output("filter-domicile","options"),
    Output("filter-dist","options"),
    Output("filter-replication","options"),
    Output("filter-strategy","options"),
    Output("filter-category","options"),
    Output("filter-country","options"),
    Output("filter-sector","options"),
    Input("preset-dd","value"),
    Input("filter-types","value"),
    Input("filter-country","value"),  # cascade: country narrows sector
    Input("live-interval","n_intervals"),  # fires on page load too
)
def update_filter_sections(preset, types, selected_country, _n):
    ptype   = PRESETS.get(preset,{}).get("type","ETF")
    custom  = ptype == "custom"
    etfs_on   = (custom and "ETF"   in (types or [])) or ptype == "ETF"
    stocks_on = (custom and "Stock" in (types or [])) or ptype == "Stock"

    types_style = {} if custom else {"display":"none"}
    etf_style   = {} if etfs_on   else {"display":"none"}
    stock_style = {} if stocks_on else {"display":"none"}

    def jcol_opts(col):
        if jetf_df.empty or col not in jetf_df.columns:
            return []
        vals = sorted([v for v in jetf_df[col].astype(str).str.strip().unique()
                       if v and v not in ("", "nan", "None")])
        return [{"label":v,"value":v} for v in vals]

    def uopts(col, filter_type=None, country_filter=None):
        if universe.empty:
            return []
        df = universe[universe["type"]==filter_type] if filter_type else universe
        if country_filter and "country" in df.columns:
            df = df[df["country"].isin(country_filter)]
        if col not in df.columns:
            return []
        vals = sorted([v for v in df[col].astype(str).str.strip().unique()
                       if v and v not in ("", "nan", "None")])
        return [{"label":v,"value":v} for v in vals]

    dom_opts  = jcol_opts("domicile")
    dist_opts = jcol_opts("dist_policy")
    repl_opts = jcol_opts("replication")
    strat_opts= jcol_opts("strategy")
    cat_opts  = uopts("category_group", "ETF")
    ctry_opts = uopts("country", "Stock")
    sect_opts = uopts("sector", "Stock", selected_country)

    return (types_style, etf_style, stock_style,
            dom_opts, dist_opts, repl_opts, strat_opts, cat_opts,
            ctry_opts, sect_opts)

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
def run_scan(run_clicks, clear_clicks, preset, types, domicile, dist,
             repl, strategy, category, country, sector,
             minsize, maxter, workers, fetch_pe, budget):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update, no_update, no_update, no_update
    triggered = ctx.triggered[0]["prop_id"]

    if "clear-btn" in triggered:
        return None, "Results cleared."

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
    if not tickers:
        return None, "⚠️ No tickers match filters.", {}, False, False

    # Parallel scan
    results = {}
    with ThreadPoolExecutor(max_workers=int(workers or 6)) as ex:
        futs = {ex.submit(analyse_ticker, t, risk_mult): t for t in tickers}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                r = fut.result()
                results[t] = r
            except Exception:
                results[t] = None

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

    return df.to_json(date_format="iso", orient="split"), status, {}, False, False

# ───────────────────────────────────────────────────────────────────
# CALLBACKS — Results display
# ───────────────────────────────────────────────────────────────────

@app.callback(
    Output("kpi-row","children"),
    Output("top-signals","children"),
    Output("results-table","children"),
    Input("scan-store","data"),
    Input("signal-tabs","active_tab"),
)
def render_results(store_data, active_tab):
    if not store_data:
        return [], [], html.P("Run a scan to see results.", className="text-muted mt-4")

    df = pd.read_json(store_data, orient="split")

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

    # Full table
    sub = df if active_tab == "all" else df[df["Action"]==active_tab]

    SHOW_COLS = ["Rank","Ticker","Name","ISIN","Price","MA200","Dist%","52W%",
                 "RSI","RSI↗","MACD","MACD⚡","Vol%","Conf",
                 "PE","Beta","Div%","MCap",
                 "Signal","Knife","Allocation"]
    show_cols = [c for c in SHOW_COLS if c in sub.columns]

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
            "whiteSpace":"normal","maxWidth":"200px",
        },
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
        return no_update, no_update, no_update, no_update, no_update
    df     = pd.read_json(store_data, orient="split")
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

    # ISIN → ticker lookup
    isin = None
    def is_isin(x):
        return len(x)==12 and x[:2].isalpha() and x[2:].isalnum()
    if is_isin(ticker):
        isin = ticker
        # Try to find ticker from justETF
        if not jetf_df.empty and "isin" in jetf_df.columns:
            match = jetf_df[jetf_df["isin"]==ticker]
            if not match.empty:
                ticker = match.iloc[0]["ticker"]

    raw = fetch_ticker_data(ticker)
    if raw is None:
        return dbc.Alert(f"No data found for {ticker}. Try a different ticker.", color="danger")

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

    # Links
    links = html.Div([
        dbc.Button("📈 Yahoo Finance", href=f"https://finance.yahoo.com/quote/{ticker}",
                   target="_blank", color="link", size="sm"),
        dbc.Button("📊 ETF.com", href=f"https://www.etf.com/{ticker}",
                   target="_blank", color="link", size="sm"),
        dbc.Button("🔍 justETF",
                   href=f"https://www.justetf.com/en/etf-profile.html?isin={isin}" if isin
                        else f"https://www.justetf.com/en/search.html?query={ticker}",
                   target="_blank", color="link", size="sm"),
    ])

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
    Output("filter-domicile","disabled"),
    Output("filter-dist","disabled"),
    Output("filter-replication","disabled"),
    Output("filter-strategy","disabled"),
    Output("filter-category","disabled"),
    Output("filter-country","disabled"),
    Output("filter-sector","disabled"),
    Output("preset-dd","disabled"),
    Input("run-btn","disabled"),
)
def disable_filters_during_scan(btn_disabled):
    """Lock all filters while scan is running to prevent confusion."""
    return [btn_disabled] * 8

@app.callback(
    Output("dl-csv","data"),
    Input("dl-btn","n_clicks"),
    State("scan-store","data"),
    prevent_initial_call=True,
)
def download_csv(n, store_data):
    if not store_data:
        return no_update
    df = pd.read_json(store_data, orient="split")
    return dcc.send_data_frame(df.to_csv, "scan_results.csv", index=False)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
