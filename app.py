# ═══════════════════════════════════════════════════════════════════
# Market Decision Engine v10.3 — Dash + Railway
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

APP_VERSION = "v10.3"
from datetime import datetime as _datetime
BUILD_TIME = _datetime.utcnow().strftime("%d %b %H:%M UTC")

# ───────────────────────────────────────────────────────────────────
# APP INIT
# ───────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Market Decision Engine",
    index_string="""<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* ── Market Decision Engine — Precision Light Theme ── */
:root {
  --bg-base:       #f8f9fc;
  --bg-surface:    #ffffff;
  --bg-card:       #ffffff;
  --bg-hover:      #f1f4f9;
  --bg-selected:   #e8f0fe;

  --border:        #e2e6ed;
  --border-light:  #eef1f6;
  --border-focus:  #1a56db;

  --text-primary:  #0f172a;
  --text-secondary:#475569;
  --text-muted:    #94a3b8;
  --text-label:    #64748b;

  --blue-primary:  #1a56db;
  --blue-light:    #dbeafe;
  --blue-mid:      #3b82f6;

  /* Signal — only for data values */
  --sig-buy:       #0d9488;
  --sig-buy-bg:    #f0fdfa;
  --sig-buy-border:#99f6e4;
  --sig-watch:     #0284c7;
  --sig-watch-bg:  #f0f9ff;
  --sig-watch-border:#bae6fd;
  --sig-sell:      #dc2626;
  --sig-sell-bg:   #fff5f5;
  --sig-sell-border:#fecaca;
  --sig-avoid:     #d97706;
  --sig-avoid-bg:  #fffbeb;
  --sig-avoid-border:#fde68a;
  --sig-wait:      #64748b;

  --shadow-sm:     0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04);
  --shadow-md:     0 4px 12px rgba(15,23,42,0.08), 0 2px 6px rgba(15,23,42,0.05);
  --radius:        6px;
  --radius-sm:     4px;
}

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body {
  background: var(--bg-base) !important;
  color: var(--text-primary) !important;
  font-family: 'DM Sans', -apple-system, sans-serif !important;
  font-size: 14px !important;
  -webkit-font-smoothing: antialiased;
}
/* Force all text visible on light bg */
*, *::before, *::after { color: inherit; }
p, span, div, label, h1, h2, h3, h4, h5, h6, small, a {
  color: var(--text-primary) !important;
}
.text-muted, .text-secondary { color: var(--text-muted) !important; }
a { color: var(--blue-primary) !important; }

/* ── Sidebar ── */
#sidebar {
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border) !important;
  box-shadow: var(--shadow-sm) !important;
}
#sidebar h6, #sidebar .fw-bold {
  font-size: 10px !important;
  font-weight: 600 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--text-muted) !important;
}
#sidebar .text-muted, #sidebar small {
  color: var(--text-muted) !important;
  font-size: 11px !important;
}
#sidebar label, #sidebar .form-label, #sidebar .small.fw-medium, #sidebar .fw-medium {
  font-size: 11.5px !important;
  font-weight: 500 !important;
  color: #475569 !important;
  margin-bottom: 3px !important;
  display: block !important;
}
/* Nuclear option — force all sidebar text to be dark */
#sidebar * { color: #0f172a !important; }
#sidebar .text-muted, #sidebar small, #sidebar .text-secondary { color: #64748b !important; }
#sidebar .btn { color: #ffffff !important; }
#sidebar .accordion-button { color: #1a56db !important; }

/* ── Navbar / App title ── */
.navbar, [class*="navbar"] {
  background: var(--bg-surface) !important;
  border-bottom: 1px solid var(--border) !important;
}

/* ── Cards ── */
.card {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow-sm) !important;
}
.card-header {
  background: var(--bg-surface) !important;
  border-bottom: 1px solid var(--border-light) !important;
  padding: 10px 14px !important;
}

/* ── Signal tabs — clean underline ── */
#signal-tabs {
  border-bottom: 1px solid var(--border) !important;
  background: var(--bg-surface) !important;
}
#signal-tabs .nav-link {
  color: var(--text-muted) !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  padding: 9px 16px !important;
  margin-bottom: -1px !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  background: transparent !important;
  transition: all 0.15s ease !important;
}
#signal-tabs .nav-link:hover {
  color: var(--text-primary) !important;
  background: var(--bg-hover) !important;
}
#signal-tabs .nav-link.active {
  color: var(--blue-primary) !important;
  border-bottom-color: var(--blue-primary) !important;
  background: transparent !important;
  font-weight: 600 !important;
}

/* ── Status alert ── */
.alert {
  border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
  font-size: 11.5px !important;
}
.alert-info {
  background: #f0f9ff !important;
  border-color: #bae6fd !important;
  color: #0369a1 !important;
  font-size: 11px !important;
  padding: 6px 12px !important;
}

/* ── Run Scan button ── */
#run-btn {
  background: var(--blue-primary) !important;
  border-color: var(--blue-primary) !important;
  color: #fff !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  letter-spacing: 0.02em !important;
  border-radius: var(--radius) !important;
  transition: all 0.15s ease !important;
  box-shadow: 0 1px 4px rgba(26,86,219,0.25) !important;
}
#run-btn:hover {
  background: #1447c0 !important;
  border-color: #1447c0 !important;
  box-shadow: 0 2px 8px rgba(26,86,219,0.35) !important;
}

/* ── Table ── */
.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner th {
  background: #f8f9fc !important;
  color: var(--text-label) !important;
  font-size: 11px !important;
  font-weight: 600 !important;
  letter-spacing: 0.05em !important;
  text-transform: uppercase !important;
  border-bottom: 1px solid var(--border) !important;
  border-right: none !important;
  padding: 7px 10px !important;
}
.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td {
  background: var(--bg-surface) !important;
  color: var(--text-primary) !important;
  border-bottom: 1px solid var(--border-light) !important;
  border-right: none !important;
  font-size: 13px !important;
  padding: 6px 10px !important;
  font-family: 'DM Mono', monospace !important;
}
.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner tr:hover td {
  background: var(--bg-hover) !important;
}
.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner tr.selected td {
  background: var(--bg-selected) !important;
}
/* Filter row */
.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner .dash-filter input {
  background: #f8f9fc !important;
  border: 1px solid var(--border) !important;
  border-radius: 3px !important;
  color: var(--text-primary) !important;
  font-size: 11px !important;
  padding: 2px 6px !important;
}

/* ── Dropdowns ── */
.Select-control {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  min-height: 30px !important;
  font-size: 12px !important;
}
.Select-control:hover { border-color: var(--blue-mid) !important; }
.Select-menu-outer {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border) !important;
  box-shadow: var(--shadow-md) !important;
}
.Select-option { color: var(--text-primary) !important; font-size: 12px !important; }
.Select-option.is-focused { background: var(--bg-hover) !important; }
.Select-option.is-selected { background: var(--blue-light) !important; color: var(--blue-primary) !important; }
.Select-placeholder { color: var(--text-muted) !important; font-size: 12px !important; }
.Select-value-label { color: var(--text-primary) !important; font-size: 12px !important; }
.Select-multi-value-wrapper { padding: 2px !important; }

/* ── Accordion (Optional Filters) ── */
.accordion-button {
  background: var(--bg-surface) !important;
  color: var(--blue-primary) !important;
  font-size: 12px !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em !important;
  padding: 8px 12px !important;
  border: none !important;
  box-shadow: none !important;
}
.accordion-button:not(.collapsed) {
  background: var(--bg-hover) !important;
}
.accordion-body {
  background: var(--bg-surface) !important;
  padding: 10px 12px !important;
}
.accordion-item { border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }

/* ── Progress bar (scan overlay) ── */
.progress { background: var(--border) !important; border-radius: 3px !important; }
.progress-bar { background: var(--blue-primary) !important; }

/* ── Overlay ── */
#scan-overlay { backdrop-filter: blur(6px) !important; background: rgba(248,249,252,0.92) !important; }
#scan-overlay h4 { color: var(--text-primary) !important; }
#scan-overlay .text-muted { color: var(--text-secondary) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #c1c9d6; }

/* ── Number formatting — signal colors for data only ── */
.sig-buy   { color: var(--sig-buy)   !important; }
.sig-sell  { color: var(--sig-sell)  !important; }
.sig-watch { color: var(--sig-watch) !important; }
.sig-avoid { color: var(--sig-avoid) !important; }
.sig-wait  { color: var(--sig-wait)  !important; }

/* ── Inputs ── */
input[type="number"], .form-control {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-primary) !important;
  font-size: 12px !important;
  border-radius: var(--radius-sm) !important;
}

/* ── Badges ── */
.badge {
  font-family: 'DM Mono', monospace !important;
  font-size: 10px !important;
  font-weight: 500 !important;
  letter-spacing: 0.03em !important;
}

/* ── Deep Dive metric cards ── */
.metric-card {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 12px 16px !important;
}

/* ── Main tabs (Market Scanner / Deep Dive) ── */
#main-tabs .nav-link {
  color: var(--text-secondary) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 10px 18px !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  background: transparent !important;
}
#main-tabs .nav-link.active {
  color: var(--blue-primary) !important;
  border-bottom-color: var(--blue-primary) !important;
  background: transparent !important;
  font-weight: 600 !important;
}
#main-tabs { border-bottom: 1px solid var(--border) !important; }

/* ── Slider ── */
.rc-slider-track { background: var(--blue-primary) !important; }
.rc-slider-handle { border-color: var(--blue-primary) !important; }

/* ── Checkbox ── */
input[type="checkbox"] { accent-color: var(--blue-primary) !important; }

/* ── Regime badge (Extreme Fear etc) ── */
.regime-badge {
  font-size: 11px !important;
  font-weight: 600 !important;
  padding: 3px 10px !important;
  border-radius: 3px !important;
}

/* FLATLY overrides — keep everything light */
.navbar-dark, .navbar-dark .navbar-brand, .navbar-dark .nav-link { color: var(--text-primary) !important; }
.bg-primary { background: var(--blue-primary) !important; }
.btn-warning { background: #f59e0b !important; border-color: #f59e0b !important; color: #fff !important; }
.btn-secondary { background: #f1f4f9 !important; border-color: var(--border) !important; color: var(--text-secondary) !important; }
.btn-secondary:hover { background: #e2e6ed !important; }
.text-muted { color: var(--text-muted) !important; }
.border { border-color: var(--border) !important; }
/* Dash loading overlay */
._dash-loading { background: rgba(248,249,252,0.8) !important; }
/* Fix dropdown menus */
.dropdown-menu { background: var(--bg-surface) !important; border: 1px solid var(--border) !important; box-shadow: var(--shadow-md) !important; }
.dropdown-item { color: var(--text-primary) !important; font-size: 12px !important; }
.dropdown-item:hover { background: var(--bg-hover) !important; }
/* Spinner */
.spinner-border { color: var(--blue-primary) !important; }
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>""",
)

server = app.server  # expose Flask server for gunicorn

# ───────────────────────────────────────────────────────────────────
# IN-MEMORY CACHE (replaces st.cache_data)
# ───────────────────────────────────────────────────────────────────

_cache = {}
_cache_lock = threading.Lock()
_active_scans = {}  # scan_id -> bool, False = cancelled

# Disk cache for FMP data — survives Railway restarts, preserves quota
import os as _os_cache, json as _json_cache
_DISK_CACHE_FILE = "/tmp/mde_fmp_cache.json"

def _load_disk_cache():
    """Load persisted FMP cache from disk on startup."""
    try:
        if _os_cache.path.exists(_DISK_CACHE_FILE):
            with open(_DISK_CACHE_FILE, "r") as f:
                data = _json_cache.load(f)
            now = time.time()
            loaded = 0
            for k, (v, exp) in data.items():
                if exp is None or exp > now:   # skip expired
                    _cache[k] = (v, exp)
                    loaded += 1
            print(f"[cache] Loaded {loaded} FMP entries from disk", flush=True)
    except Exception as e:
        print(f"[cache] Disk load failed: {e}", flush=True)

def _save_disk_cache():
    """Persist FMP cache entries to disk (fmp_ prefix only)."""
    try:
        with _cache_lock:
            fmp_entries = {k: v for k, v in _cache.items()
                           if str(k).startswith("fmp_")}
        with open(_DISK_CACHE_FILE, "w") as f:
            _json_cache.dump(fmp_entries, f)
    except Exception as e:
        print(f"[cache] Disk save failed: {e}", flush=True)

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
    # Persist FMP entries to disk immediately
    if str(key).startswith("fmp_"):
        threading.Thread(target=_save_disk_cache, daemon=True).start()

# Load disk cache at startup
_load_disk_cache()

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
        # Filter extreme dist_ma200 — likely bad data (>95% below MA200 is almost impossible)
        if "dist_ma200" in df.columns:
            df = df[(df["dist_ma200"].isna()) | (df["dist_ma200"] > -95)]
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

# ── Backfill value_grade if column missing ───────────────────────────
# Uses extended columns if present (new builder), falls back to basic ones.
if not signals_df.empty and "value_grade" not in signals_df.columns:
    def _backfill_value_grades(df):
        grades, scores = [], []
        for _, r in df.iterrows():
            try:
                asset_type = str(r.get("type","Stock"))
                if asset_type != "Stock":
                    grades.append(None); scores.append(None); continue
                def _f(col):
                    v = r.get(col)
                    try: return float(v) if pd.notna(v) and v not in ("","nan","None") else None
                    except: return None
                # Use whatever columns are available — new builder has more
                pe  = _f("pe_ratio")
                pb  = _f("pb_ratio")
                div = _f("div_yield")
                mc  = _f("market_cap")
                roe = _f("roe")
                de  = _f("debt_equity")
                fcf = _f("fcf_yield")
                rev = _f("rev_growth")
                peg = _f("peg")
                fund = {
                    "fmp_pe_ttm":    pe,
                    "fmp_pb":        pb,
                    "fmp_div_yield": div,
                    "fmp_mcap":      mc,
                    "fmp_roe":       roe,
                    "fmp_debt_eq":   de,
                    "fmp_fcf_yield": fcf,
                    "fmp_rev_growth":rev,
                    "fmp_peg":       peg,
                }
                s, g, _, _ = compute_value_score(fund)
                grades.append(g); scores.append(s)
            except Exception:
                grades.append(None); scores.append(None)
        df = df.copy()
        df["value_grade"] = grades
        df["value_score"] = scores
        graded = [g for g in grades if g]
        from collections import Counter
        dist = Counter(graded)
        print(f"[signals] Value grades backfilled: {len(graded):,} stocks "
              f"(A:{dist.get('A',0)} B:{dist.get('B',0)} "
              f"C:{dist.get('C',0)} D:{dist.get('D',0)})", flush=True)
        return df
    signals_df = _backfill_value_grades(signals_df)

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
        elif "." not in symbol:     stooq_sym = symbol + ".US"  # US stocks need .US suffix on Stooq
        end   = pd.Timestamp.today()
        start = end - pd.Timedelta(days=800)
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
        cutoff = pd.Timestamp.today() - pd.Timedelta(days=800)
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


def fetch_ticker_data(ticker, isin=None, force_refresh=False):
    # Quick reject obvious invalid tickers before any API call
    if not ticker or ticker.startswith("$") or len(ticker) < 2:
        return None
    key = f"tick_{ticker}"
    if force_refresh:
        # Bust all cached keys for this ticker so we get truly fresh data
        with _cache_lock:
            base = ticker.split(".")[0]
            for k in list(_cache.keys()):
                if k in (key, f"sfx_{ticker}", f"resolved_{ticker}", f"meta_{ticker}") \
                        or k.startswith(f"yfund_{ticker}") \
                        or k.startswith(f"conv_{ticker}"):
                    _cache.pop(k, None)
            # Also bust FMP cache keys for this ticker's base symbol
            for k in list(_cache.keys()):
                if k.startswith(f"fmp_") and f"/{base}" in k:
                    _cache.pop(k, None)
        print(f"[refresh] Busted cache for {ticker}", flush=True)
    else:
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
        _ma_check = close.rolling(200).mean().iloc[-1]
        if pd.notna(_ma_check) and float(_ma_check) > 0 and (float(close.iloc[-1]) / float(_ma_check)) > 10 and isin:
            df_j = fetch_justetf_chart(isin)
            if not df_j.empty and len(df_j["Close"].dropna()) >= 30:
                df    = df_j
                close = df["Close"].dropna()
        price = float(close.iloc[-1])
        if price < 0.10:  # only filter near-zero prices
            return None
        # Sanity: if price is >10x MA200, likely wrong ticker (e.g. SXRN.US vs SXRN.DE)
        # Skip this check if MA200 is NaN (stock listed < 200 days ago — perfectly valid)
        _ma200c = close.rolling(200).mean().iloc[-1]
        if pd.notna(_ma200c) and float(_ma200c) > 0 and (price / float(_ma200c)) > 10:
            return None
        if "Volume" in df.columns:
            avg_vol = df["Volume"].dropna().tail(20).mean()
            # ETFs can have very low volume but still be valid — only filter true zeros
            if avg_vol == 0:
                return None
        ma50_raw  = close.rolling(50).mean().iloc[-1]
        ma200_raw = close.rolling(200).mean().iloc[-1]
        # For recently listed stocks (< 50 or < 200 days), fall back to price itself
        # so dist_ma and downstream calculations remain meaningful
        ma50  = float(ma50_raw)  if pd.notna(ma50_raw)  else float(price)
        ma200 = float(ma200_raw) if pd.notna(ma200_raw) else float(price)
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
        ma200_s_full = close.rolling(200).mean()
        # MA200 slope: normalised linear slope over last 60 bars
        # > +0.001 = rising trend, < -0.001 = falling trend, < -0.003 = steep decline
        _ma200_tail = ma200_s_full.dropna().tail(60)
        if len(_ma200_tail) >= 20:
            _y = _ma200_tail.values
            _x = np.arange(len(_y))
            _slope_raw = np.polyfit(_x, _y, 1)[0]
            ma200_slope = float(_slope_raw / (abs(_y.mean()) + 1e-9))
        else:
            ma200_slope = 0.0

        result = dict(
            price=price, ma50=ma50, ma200=ma200,
            rsi=rsi, rsi_slope=rsi_slope,
            macd=float(macd_l.iloc[-1]), macd_signal=float(macd_sig.iloc[-1]),
            macd_hist=float(macd_h.iloc[-1]),
            dist_ma=dist_ma, dist_52h=dist_52h,
            vol=vol_pct, price_slope=price_slope,
            trend_down_strong=(price_slope < -0.003),
            confidence=conf,
            ma200_slope=ma200_slope,
            # store full series for deep dive charts
            close=close, ma50_s=close.rolling(50).mean(),
            ma200_s=ma200_s_full,
            rsi_s=rsi_s, macd_l=macd_l, macd_sig=macd_sig, macd_h=macd_h,
        )
        cache_set(key, result, ttl=3600)
        # Write fresh signal back to in-memory signals_df
        global signals_df
        try:
            from datetime import date
            new_row = {
                "ticker": ticker, "data_source": "live_scan",
                "price": result.get("price"), "ma200": result.get("ma200"),
                "dist_ma200": result.get("dist_ma"), "rsi": result.get("rsi"),
                "rsi_rising": int(result.get("rsi_slope",0) > 0),
                "macd_bull": int(result.get("macd",0) > result.get("macd_signal",0)),
                "macd_accel": int(result.get("macd_hist",0) > 0),
                "vol_pct": result.get("vol"), "conf": result.get("confidence"),
                "action": result.get("action","WAIT"), "score": result.get("score",0),
                "is_knife": int(result.get("is_knife",False)),
                "reversal": int(result.get("reversal",False)),
                "computed_at": date.today().isoformat(),
            }
            if not signals_df.empty and "ticker" in signals_df.columns:
                signals_df = signals_df[signals_df["ticker"] != ticker]
                signals_df = pd.concat(
                    [pd.DataFrame([new_row]), signals_df], ignore_index=True)
        except Exception:
            pass
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

# ── FMP API helper ────────────────────────────────────────────────
import os as _os
_FMP_KEY = _os.environ.get("FMP_API_KEY", "")

def _get_fmp_key():
    """Always read from env — picks up Railway variable without restart."""
    return _os.environ.get("FMP_API_KEY", "") or _FMP_KEY

def _fmp_get(path, params=None):
    """Call FMP API. Returns JSON or None. Cached 1h."""
    key = _get_fmp_key()
    if not key:
        return None
    cache_key = f"fmp_{path}_{str(params)}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        import requests as _req
        base = "https://financialmodelingprep.com/api/v3"
        p = dict(apikey=key, **(params or {}))
        r = _req.get(f"{base}/{path}", params=p, timeout=6)
        if r.status_code == 200:
            data = r.json()
            # 24h TTL — fundamentals don't change intraday, preserves quota
            cache_set(cache_key, data, ttl=86400)
            return data
    except Exception:
        pass
    return None

def fetch_fmp_fundamentals(ticker):
    """
    Fetch FMP second-opinion fundamentals for a ticker.
    Returns dict with valuation, analyst targets, earnings date.
    """
    result = {}
    base_ticker = ticker.split(".")[0]  # strip exchange suffix

    # Quote / key metrics
    data = _fmp_get(f"quote/{base_ticker}")
    if data and isinstance(data, list) and data:
        q = data[0]
        def _f(v):
            try: return float(v) if v is not None else None
            except: return None
        result["fmp_price"]       = _f(q.get("price"))
        result["fmp_pe"]          = _f(q.get("pe"))
        result["fmp_eps"]         = _f(q.get("eps"))
        result["fmp_target"]      = _f(q.get("priceAvg50"))
        result["fmp_mcap"]        = _f(q.get("marketCap"))
        result["fmp_volume"]      = _f(q.get("volume"))
        result["fmp_52w_high"]    = _f(q.get("yearHigh"))
        result["fmp_52w_low"]     = _f(q.get("yearLow"))

    # Key metrics (PE, dividend, debt/equity etc)
    km = _fmp_get(f"key-metrics-ttm/{base_ticker}")
    if km and isinstance(km, list) and km:
        k = km[0]
        def _f(v):
            try: return float(v) if v is not None else None
            except: return None
        result["fmp_pe_ttm"]       = _f(k.get("peRatioTTM"))
        result["fmp_fwd_pe"]       = _f(k.get("priceToEarningsRatioTTM"))
        result["fmp_pb"]           = _f(k.get("pbRatioTTM"))
        result["fmp_div_yield"]    = _f(k.get("dividendYieldTTM"))
        result["fmp_debt_eq"]      = _f(k.get("debtToEquityTTM"))
        result["fmp_fcf_yield"]    = _f(k.get("freeCashFlowYieldTTM"))
        result["fmp_roe"]          = _f(k.get("roeTTM"))
        result["fmp_ev_ebitda"]    = _f(k.get("enterpriseValueOverEBITDATTM"))

    # Analyst estimates / price target
    pt = _fmp_get(f"price-target-consensus/{base_ticker}")
    if pt and isinstance(pt, list) and pt:
        p = pt[0]
        def _f(v):
            try: return float(v) if v is not None else None
            except: return None
        result["fmp_target_high"]  = _f(p.get("targetHigh"))
        result["fmp_target_low"]   = _f(p.get("targetLow"))
        result["fmp_target_mean"]  = _f(p.get("targetConsensus"))
        result["fmp_target_med"]   = _f(p.get("targetMedian"))

    # Earnings calendar
    ec = _fmp_get(f"earning_calendar/{base_ticker}")
    if ec and isinstance(ec, list):
        upcoming = [e for e in ec if e.get("date","") >= str(pd.Timestamp.today().date())]
        if upcoming:
            result["fmp_next_earnings"] = upcoming[0].get("date","")

    # Financial growth (revenue + earnings YoY)
    fg = _fmp_get(f"financial-growth/{base_ticker}", {"limit": 1})
    if fg and isinstance(fg, list) and fg:
        g = fg[0]
        def _f(v):
            try: return float(v) if v is not None else None
            except: return None
        result["fmp_rev_growth"]  = _f(g.get("revenueGrowth"))
        result["fmp_eps_growth"]  = _f(g.get("epsgrowth") or g.get("epsGrowth"))
        result["fmp_net_inc_growth"] = _f(g.get("netIncomeGrowth"))

    # PEG ratio — PE / earnings growth rate
    # Use TTM PE and TTM EPS growth; fall back to FWD PE / analyst growth
    _pe_for_peg  = result.get("fmp_pe_ttm") or result.get("fmp_pe")
    _eps_g       = result.get("fmp_eps_growth")
    if _pe_for_peg and _eps_g and _eps_g > 0.01:
        result["fmp_peg"] = round(_pe_for_peg / (_eps_g * 100), 2)
    else:
        result["fmp_peg"] = None

    return result


def compute_value_score(fmp):
    """
    Compute a 0-100 value score from FMP fundamentals.
    Returns (score, grade, breakdown_dict).
    Gracefully handles missing data — missing metrics contribute 0.
    """
    if not fmp:
        return 0, "N/A", {}

    def _f(v):
        try: return float(v)
        except: return None

    pe        = _f(fmp.get("fmp_pe_ttm") or fmp.get("fmp_pe"))
    peg       = _f(fmp.get("fmp_peg"))
    pb        = _f(fmp.get("fmp_pb"))
    fcf_yield = _f(fmp.get("fmp_fcf_yield"))   # already a ratio e.g. 0.06 = 6%
    roe       = _f(fmp.get("fmp_roe"))          # ratio e.g. 0.18 = 18%
    debt_eq   = _f(fmp.get("fmp_debt_eq"))
    rev_growth= _f(fmp.get("fmp_rev_growth"))   # ratio e.g. 0.12 = 12%
    div_yield = _f(fmp.get("fmp_div_yield"))
    eps_growth= _f(fmp.get("fmp_eps_growth"))

    breakdown = {}
    total_weight = 0
    weighted_sum = 0

    def score_metric(name, value, weight, thresholds):
        """
        thresholds: list of (max_value, score_0_to_1) sorted ascending.
        Returns weighted contribution and records in breakdown.
        """
        nonlocal total_weight, weighted_sum
        if value is None:
            breakdown[name] = None
            return
        # Find score by threshold
        s = 0.0
        for thresh, pts in thresholds:
            if value <= thresh:
                s = pts
                break
        else:
            s = thresholds[-1][1]  # use last bucket if above all thresholds
        breakdown[name] = round(s * 100)
        weighted_sum  += s * weight
        total_weight  += weight

    # ── PE ratio (25%) — lower is cheaper ────────────────────────────
    # Negative PE = losing money = 0 pts
    if pe is not None and pe > 0:
        score_metric("PE", pe, 25, [
            (10,  1.00), (15,  0.85), (20, 0.70),
            (25,  0.55), (35,  0.35), (50, 0.15), (999, 0.0),
        ])
    elif pe is not None and pe <= 0:
        breakdown["PE"] = 0
        total_weight += 25  # penalise loss-making companies

    # ── PEG ratio (20%) — < 1 means growth is cheap ──────────────────
    if peg is not None and peg > 0:
        score_metric("PEG", peg, 20, [
            (0.5, 1.00), (0.8, 0.85), (1.0, 0.70),
            (1.5, 0.45), (2.0, 0.20), (999, 0.0),
        ])

    # ── P/B ratio (15%) — below book value is a gift ─────────────────
    if pb is not None and pb > 0:
        score_metric("P/B", pb, 15, [
            (1.0, 1.00), (1.5, 0.80), (2.5, 0.55),
            (4.0, 0.25), (7.0, 0.10), (999, 0.0),
        ])

    # ── FCF yield (15%) — higher = more cash generated per $ of price ─
    if fcf_yield is not None:
        fcf_pct = fcf_yield * 100  # convert to %
        score_metric("FCF Yield", fcf_pct, 15, [
            (-999, 0.0),   # negative FCF = bad
            (0,    0.05),
            (2,    0.30),  (4,   0.55), (6,   0.75),
            (8,    0.90),  (999, 1.00),
        ])

    # ── ROE (10%) — return on equity ─────────────────────────────────
    if roe is not None:
        roe_pct = roe * 100
        score_metric("ROE", roe_pct, 10, [
            (0,   0.0),  (5,  0.20), (10, 0.45),
            (15,  0.65), (20, 0.80), (30, 0.95), (999, 1.00),
        ])

    # ── Debt/Equity (10%) — lower is safer ───────────────────────────
    if debt_eq is not None and debt_eq >= 0:
        score_metric("D/E", debt_eq, 10, [
            (0.1, 1.00), (0.3, 0.85), (0.7, 0.65),
            (1.5, 0.35), (3.0, 0.10), (999, 0.0),
        ])

    # ── Revenue growth (5%) ───────────────────────────────────────────
    if rev_growth is not None:
        rev_pct = rev_growth * 100
        score_metric("Rev Growth", rev_pct, 5, [
            (-999, 0.0), (0, 0.10), (5, 0.40),
            (10,   0.65), (15, 0.85), (999, 1.00),
        ])

    # ── Dividend yield bonus (extra 5 pts if >2%, not in main weight) ─
    div_bonus = 0
    if div_yield is not None and div_yield > 0.02:
        div_bonus = min(5, round(div_yield * 100))  # up to 5 bonus points

    if total_weight == 0:
        return 0, "N/A", breakdown, 0

    raw_score = (weighted_sum / total_weight) * 100 + div_bonus
    score = min(100, round(raw_score))

    grade = (
        "A" if score >= 75 else
        "B" if score >= 55 else
        "C" if score >= 35 else
        "D"
    )
    # Coverage: number of metrics that had data
    coverage = sum(1 for v in breakdown.values() if v is not None)
    return score, grade, breakdown, coverage


def fetch_fmp_value_batch(tickers, max_workers=8):
    """
    Fetch FMP fundamentals for a list of tickers in parallel.
    Skips tickers already in cache — preserves daily quota.
    Returns dict: {ticker: fmp_data}
    """
    if not _get_fmp_key():
        return {}

    results = {}
    to_fetch = []

    for t in tickers:
        # Check if quote endpoint is already cached for this ticker
        base = t.split(".")[0]
        cached_quote = cache_get(f"fmp_quote/{base}_None")
        if cached_quote is not None:
            # Already cached — reconstruct via fetch (hits cache, no API call)
            results[t] = fetch_fmp_fundamentals(t)
        else:
            to_fetch.append(t)

    cache_hits = len(tickers) - len(to_fetch)
    if cache_hits:
        print(f"[FMP batch] {cache_hits} cache hits, "
              f"{len(to_fetch)} need fresh fetch", flush=True)

    if to_fetch:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(fetch_fmp_fundamentals, t): t for t in to_fetch}
            for fut in as_completed(futs):
                t = futs[fut]
                try:
                    results[t] = fut.result()
                except Exception:
                    results[t] = {}

    return results


def fetch_pe_data(ticker):
    key = f"pe_{ticker}"
    cached = cache_get(key)
    if cached is not None:
        return cached

    def _safe(v, mn=0, mx=5000):
        try:
            f = float(v)
            return f if mn < f < mx else None
        except:
            return None

    info = {}
    t = yf.Ticker(ticker)

    # fast_info: lightweight, rarely hangs — get market cap + price
    try:
        fi = t.fast_info
        mcap_fast = getattr(fi, "market_cap", None)
        info["marketCap"] = mcap_fast
    except Exception:
        pass

    # Full .info: can hang on obscure tickers — run in thread with 8s timeout
    import concurrent.futures as _cf
    def _get_info():
        try:
            return t.info or {}
        except Exception:
            return {}
    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_get_info)
            full = fut.result(timeout=8)
        info.update(full)
    except Exception:
        pass  # timeout or error — use whatever fast_info gave us

    pe   = _safe(info.get("trailingPE"), 0, 2000) or _safe(info.get("forwardPE"), 0, 2000)
    mcap = _safe(info.get("marketCap"), 0, 1e15)
    beta = _safe(info.get("beta"), -5, 10)
    div  = _safe(info.get("dividendYield"), 0, 1)

    result = {
        "PE":   round(pe, 1) if pe else "—",
        "Beta": round(beta, 2) if beta else "—",
        "Div%": f"{div*100:.2f}%" if div else "—",
        "MCap": (f"${mcap/1e9:.1f}B" if mcap >= 1e9 else f"${mcap/1e6:.0f}M") if mcap else "—",
    }
    cache_set(key, result, ttl=3600)
    return result

def fetch_yf_fundamentals(ticker, timeout=5):
    """
    Fetch full fundamental metrics from yfinance for value scoring.
    Covers all 7 metrics: PE, PEG, P/B, FCF yield, ROE, D/E, rev growth.
    Uses 5s timeout — partial results returned on timeout.
    Cached 24h to match FMP TTL.
    """
    cache_key = f"yfund_{ticker}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    import concurrent.futures as _cf

    def _get_info():
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            # Also get cashflow for FCF yield computation
            try:
                cf = t.cashflow
                if cf is not None and not cf.empty:
                    # Free cash flow = Operating CF - Capex
                    op_cf_row  = [r for r in cf.index if "Operating" in str(r)]
                    capex_row  = [r for r in cf.index if "Capital" in str(r)]
                    if op_cf_row and capex_row:
                        op_cf  = float(cf.loc[op_cf_row[0]].iloc[0])
                        capex  = float(cf.loc[capex_row[0]].iloc[0])
                        info["_fcf"] = op_cf + capex  # capex is negative
            except Exception:
                pass
            return info
        except Exception:
            return {}

    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_get_info)
            info = fut.result(timeout=timeout)
    except Exception:
        info = {}

    def _sf(v, mn=-1e15, mx=1e15):
        try:
            f = float(v)
            return f if mn <= f <= mx else None
        except:
            return None

    pe         = _sf(info.get("trailingPE"),   0, 2000)
    fwd_pe     = _sf(info.get("forwardPE"),     0, 2000)
    pb         = _sf(info.get("priceToBook"),   0, 500)
    roe        = _sf(info.get("returnOnEquity"),-5, 50)
    de         = _sf(info.get("debtToEquity"),   0, 100)
    rev_growth = _sf(info.get("revenueGrowth"), -5, 50)
    eps_growth = _sf(info.get("earningsGrowth"),-5, 50)
    div_yield  = _sf(info.get("dividendYield"),  0, 1)
    mcap       = _sf(info.get("marketCap"),      0, 1e15)

    # FCF yield = FCF / market cap
    fcf_yield = None
    fcf_raw   = info.get("_fcf") or info.get("freeCashflow")
    if fcf_raw and mcap and mcap > 0:
        try:
            fcf_yield = float(fcf_raw) / mcap
        except Exception:
            pass

    # PEG: use yfinance trailingPegRatio if available, else compute
    peg = _sf(info.get("trailingPegRatio"), 0, 20)
    if peg is None and pe and eps_growth and eps_growth > 0.01:
        peg = round(pe / (eps_growth * 100), 2)

    result = {
        "fmp_pe_ttm":    pe or fwd_pe,
        "fmp_peg":       peg,
        "fmp_pb":        pb,
        "fmp_fcf_yield": fcf_yield,
        "fmp_roe":       roe,
        "fmp_debt_eq":   de / 100 if de else None,  # yf returns as %, normalise
        "fmp_rev_growth":rev_growth,
        "fmp_eps_growth":eps_growth,
        "fmp_div_yield": div_yield,
        "fmp_mcap":      mcap,
        "_source":       "yfinance",
    }
    cache_set(cache_key, result, ttl=86400)
    return result


def fetch_yf_fundamentals_batch(tickers, max_workers=8, timeout=5):
    """
    Parallel yfinance fundamental fetch with per-ticker timeout.
    Returns {ticker: fundamentals_dict}, timed_out: list
    """
    results   = {}
    timed_out = []

    # Check cache first
    to_fetch = []
    for t in tickers:
        cached = cache_get(f"yfund_{t}")
        if cached is not None:
            results[t] = cached
        else:
            to_fetch.append(t)

    if to_fetch:
        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(fetch_yf_fundamentals, t, timeout): t
                    for t in to_fetch}
            for fut in _cf.as_completed(futs, timeout=timeout + 2):
                t = futs[fut]
                try:
                    results[t] = fut.result()
                except Exception:
                    results[t] = {}
                    timed_out.append(t)

        # Any futures that didn't complete = timed out
        for fut, t in futs.items():
            if t not in results:
                results[t] = {}
                timed_out.append(t)

    return results, timed_out


def fetch_conviction_signals(ticker, timeout=5):
    """
    Fetch conviction overlay signals from yfinance:
    - Insider buying (net purchases last 6m)
    - Short interest (% of float)
    - Earnings beat history (last 4 quarters)
    - Analyst consensus (recommendation mean)
    Returns dict with conviction_score (0-100) and grade.
    Cached 24h.
    """
    cache_key = f"conv_{ticker}"
    cached = cache_get(cache_key)
    if cached is not None:
        return cached

    import concurrent.futures as _cf

    def _fetch():
        result = {}
        t = yf.Ticker(ticker)

        # Analyst consensus (1=Strong Buy → 5=Strong Sell)
        try:
            info = t.info or {}
            rec  = info.get("recommendationMean")
            n_analysts = info.get("numberOfAnalystOpinions", 0)
            if rec and n_analysts:
                result["analyst_mean"]    = float(rec)
                result["analyst_count"]   = int(n_analysts)
                result["analyst_buy_pct"] = max(0, round((3.0 - float(rec)) / 2.0 * 100))
        except Exception:
            pass

        # Short interest
        try:
            si = info.get("shortPercentOfFloat") or info.get("shortRatio")
            if si:
                result["short_pct"] = float(si) * 100 if float(si) < 1 else float(si)
        except Exception:
            pass

        # Earnings surprise history — last 4 quarters
        try:
            eh = t.earnings_history
            if eh is not None and not eh.empty and "surprisePercent" in eh.columns:
                surprises = eh["surprisePercent"].dropna().tail(4).tolist()
                result["earnings_surprises"]   = [round(s, 2) for s in surprises]
                result["earnings_beat_count"]  = sum(1 for s in surprises if s > 0)
                result["earnings_beat_avg_pct"]= round(sum(surprises)/len(surprises), 2) if surprises else None
        except Exception:
            pass

        # Insider purchases
        try:
            ins = t.insider_purchases
            if ins is not None and not ins.empty:
                # Net shares purchased (positive = buying, negative = selling)
                if "Shares" in ins.columns and "Transaction" in ins.columns:
                    buys  = ins[ins["Transaction"].str.contains("Purchase", na=False)]["Shares"].sum()
                    sells = ins[ins["Transaction"].str.contains("Sale",     na=False)]["Shares"].sum()
                    result["insider_net_shares"] = int(buys - sells)
                    result["insider_buying"]     = buys > sells
        except Exception:
            pass

        return result

    try:
        with _cf.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_fetch)
            raw = fut.result(timeout=timeout)
    except Exception:
        raw = {}

    # ── Compute conviction score 0–100 ───────────────────────────────
    score  = 50  # neutral baseline
    labels = []

    # Analyst consensus (±20 pts)
    am = raw.get("analyst_mean")
    if am is not None:
        if   am <= 1.5: score += 20; labels.append("Strong analyst Buy")
        elif am <= 2.2: score += 12; labels.append("Analyst Buy")
        elif am <= 2.8: score +=  4
        elif am <= 3.5: score -=  8; labels.append("Analyst Hold")
        else:           score -= 20; labels.append("Analyst Sell")

    # Short interest (±15 pts)
    si = raw.get("short_pct")
    if si is not None:
        if   si > 30: score -= 15; labels.append(f"High short {si:.0f}%")
        elif si > 15: score -=  8; labels.append(f"Elevated short {si:.0f}%")
        elif si <  3: score +=  5

    # Earnings beat streak (±15 pts)
    beats = raw.get("earnings_beat_count", 0)
    avg   = raw.get("earnings_beat_avg_pct")
    if beats == 4:  score += 15; labels.append("4/4 earnings beats")
    elif beats == 3:score += 10; labels.append("3/4 earnings beats")
    elif beats <= 1:score -=  8; labels.append("Poor earnings history")

    # Insider buying (±15 pts)
    if raw.get("insider_buying") is True:
        score += 15; labels.append("Insider buying")
    elif raw.get("insider_net_shares", 0) < 0:
        score -=  8; labels.append("Insider selling")

    score = max(0, min(100, score))
    grade = "🟢 HIGH" if score >= 70 else "🟡 MED" if score >= 45 else "🔴 LOW"

    conviction = {
        **raw,
        "conviction_score": score,
        "conviction_grade": grade,
        "conviction_labels": labels,
    }
    cache_set(cache_key, conviction, ttl=86400)
    return conviction


def fetch_conviction_batch(tickers, max_workers=6, timeout=5):
    """Parallel conviction fetch. Only called for tickers passing value threshold."""
    results = {}
    import concurrent.futures as _cf

    # Cache check first
    to_fetch = [t for t in tickers if cache_get(f"conv_{t}") is None]
    for t in tickers:
        c = cache_get(f"conv_{t}")
        if c is not None:
            results[t] = c

    if to_fetch:
        with _cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(fetch_conviction_signals, t, timeout): t
                    for t in to_fetch}
            for fut in _cf.as_completed(futs, timeout=timeout + 2):
                t = futs[fut]
                try:
                    results[t] = fut.result()
                except Exception:
                    results[t] = {"conviction_grade": "—", "conviction_score": 50}

        for fut, t in [(f, futs[f]) for f in futs]:
            if t not in results:
                results[t] = {"conviction_grade": "—", "conviction_score": 50}

    return results


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
    "BUY":   "#0d9488", "WATCH": "#0284c7",
    "AVOID": "#d97706", "SELL":  "#dc2626", "WAIT": "#64748b",
}
SIGNAL_BG = {
    "BUY":   "#f0fdfa", "WATCH": "#f0f9ff",
    "AVOID": "#fffbeb", "SELL":  "#fef2f2",
    "WAIT":  "#f8fafc",
}

def badge(action):
    color = SIGNAL_COLORS.get(action, "#4a5568")
    bg    = SIGNAL_BG.get(action, "rgba(255,255,255,0.06)")
    return html.Span(action, style={
        "background": bg, "color": color, "fontWeight": "600",
        "padding": "2px 10px", "borderRadius": "3px", "fontSize": "11px",
        "letterSpacing": "0.05em", "border": f"1px solid {color}22",
    })

def kpi_card(title, value, color="#e8edf5"):
    return dbc.Card(dbc.CardBody([
        html.P(title, className="mb-0", style={"fontSize":"10px","color":"#94a3b8",
                                                "textTransform":"uppercase","letterSpacing":"0.08em"}),
        html.H4(value, style={"color": color, "marginBottom":0, "fontWeight":"700"}),
    ]), style={"background":"#ffffff","border":"1px solid #e2e6ed","borderRadius":"6px"})

# ───────────────────────────────────────────────────────────────────
# LAYOUT
# ───────────────────────────────────────────────────────────────────

# Dark theme style for all Dropdown components
_DD = {"backgroundColor":"#f8f9fc","color":"#0f172a","border":"1px solid rgba(255,255,255,0.08)","marginBottom":"6px"}
_DD_STYLE = {"option":{"backgroundColor":"#f8f9fc","color":"#0f172a"},
             "control":{"backgroundColor":"#f8f9fc","borderColor":"rgba(255,255,255,0.08)","color":"#0f172a"},
             "singleValue":{"color":"#0f172a"},"placeholder":{"color":"#94a3b8"},
             "menu":{"backgroundColor":"#f8f9fc"},"input":{"color":"#0f172a"},
             "multiValue":{"backgroundColor":"#111827"},
             "multiValueLabel":{"color":"#0f172a"}}

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
                          style={"color":"#0f172a","fontWeight":"bold","fontSize":"15px"}),
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
            dbc.Col([html.P("VIX",style={"color":"#94a3b8","marginBottom":"2px","fontSize":"12px"}),
                     html.H4(id="vix-val", className="")], width=6),
            dbc.Col([html.P("Fear & Greed",style={"color":"#94a3b8","marginBottom":"2px","fontSize":"12px"}),
                     html.H4(id="fg-val", className="")], width=6),
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
                    html.Label("Asset Type", className="small fw-medium"),
                    dbc.Checklist(
                        id="filter-types",
                        options=[{"label":"ETF","value":"ETF"},{"label":"Stock","value":"Stock"}],
                        value=["ETF"], inline=True,
                        style={"color":"#0f172a"},
                    ),
                ], id="types-row", style={"display":"none"}, className="mb-2"),

                # ETF filters
                html.Div([
                    html.Label("📦 ETF Filters", className="text-info small fw-bold mt-2"),
                    html.Label("Domicile", className="small fw-medium"),
                    dcc.Dropdown(id="filter-domicile", options=def_jcol("domicile"),
                        multi=True, placeholder="Any", style=_DD, className="dash-dark-dd"),
                    html.Label("Distribution", className="small fw-medium"),
                    dcc.Dropdown(id="filter-dist", options=def_jcol("dist_policy"),
                        multi=True, placeholder="Any", style=_DD, className="dash-dark-dd"),
                    html.Label("Replication", className="small fw-medium"),
                    dcc.Dropdown(id="filter-replication", options=def_jcol("replication"),
                        multi=True, placeholder="Any", style=_DD, className="dash-dark-dd"),
                    html.Label("Strategy", className="small fw-medium"),
                    dcc.Dropdown(id="filter-strategy", options=def_jcol("strategy"),
                        multi=True, placeholder="Any", style=_DD, className="dash-dark-dd"),
                    html.Label("Asset Class", className="small fw-medium"),
                    dcc.Dropdown(id="filter-category", options=[{"label":v,"value":v} for v in cat_opts],
                        multi=True, placeholder="Any", style=_DD, className="dash-dark-dd"),
                    dbc.Row([
                        dbc.Col([html.Label("Min Size €m", className="small fw-medium"),
                                 dbc.Input(id="filter-minsize", type="number", value=0, min=0, step=50, size="sm")], width=6),
                        dbc.Col([html.Label("Max TER %", className="small fw-medium"),
                                 dbc.Input(id="filter-maxter", type="number", value=2.0, min=0, max=5, step=0.05, size="sm")], width=6),
                    ], className="mb-2"),
                ], id="etf-filters-row"),

                # Stock filters
                html.Div([
                    html.Label("📈 Stock Filters", className="text-info small fw-bold mt-2"),
                    html.Label("Country", className="small fw-medium"),
                    dcc.Dropdown(id="filter-country", options=[{"label":v,"value":v} for v in country_opts],
                        multi=True, placeholder="Any", style=_DD, className="dash-dark-dd"),
                    html.Label("Sector", className="small fw-medium"),
                    dcc.Dropdown(id="filter-sector", options=[{"label":v,"value":v} for v in sector_opts],
                        multi=True, placeholder="Any", style=_DD, className="dash-dark-dd"),
                ], id="stock-filters-row"),

            ], title="🔧 Optional Filters"),
        ], start_collapsed=False, className="mb-3"),

        # ── Settings
        html.Label("💰 Monthly Budget (EUR)", className="small fw-medium"),
        dbc.Input(id="budget-input", type="number", value=1000, min=100, step=100, className="mb-2", size="sm"),
        html.Label("⚡ Parallel Workers", className="small fw-medium"),
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

[📈 Live VIX ↗](https://finance.yahoo.com/quote/%5EVIX/)

---

**Fear & Greed** (CNN composite):
- 0–25 → Extreme Fear 🔴 *(best entries)*
- 25–45 → Fear 🟠
- 45–55 → Neutral 🟡
- 55–75 → Greed 🟢
- 75–100 → Extreme Greed 💚 *(caution)*

[🧠 CNN F&G ↗](https://edition.cnn.com/markets/fear-and-greed)
            """, className="text-muted", style={"fontSize":"12px"},
                link_target="_blank"),
        ], title="📖 VIX & F&G Guide")], start_collapsed=True),

    ], style={
        "width":"280px","minWidth":"280px","background":"#f8f9fc",
        "padding":"16px","height":"100vh","overflowY":"auto",
        "borderRight":"1px solid #e2e6ed","flexShrink":0,
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
                    dbc.Tab(label="All",        tab_id="all"),
                    dbc.Tab(label="🟢 BUY",    tab_id="BUY"),
                    dbc.Tab(label="👀 WATCH",  tab_id="WATCH"),
                    dbc.Tab(label="⛔ AVOID",  tab_id="AVOID"),
                    dbc.Tab(label="🔴 SELL",   tab_id="SELL"),
                    dbc.Tab(label="🟡 WAIT",   tab_id="WAIT"),
                    dbc.Tab(label="💎 Value A", tab_id="VALUE_A", id="tab-value-a"),
                    dbc.Tab(label="🔷 Value B", tab_id="VALUE_B", id="tab-value-b"),
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
                html.Label("Ticker or ISIN", className=""),
                dbc.Input(id="dd-input", placeholder="e.g. VWRA or IE00B3RBWM25",
                          type="text", className="mb-2"),
            ], width=8),
            dbc.Col([
                html.Label("Budget (EUR)", className=""),
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
        dcc.Store(id="dd-scanner-store", data={}),
    ])

def value_screener_tab():
    """Layout for the Value Screener tab."""
    return html.Div([
        # ── Candidate loader bar ─────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.Span("🎯 Load from scanner cache: ", className="small text-muted me-2"),
                dbc.Button("📡 Stock candidates (BUY/WATCH dips)",
                           id="vs-candidates-btn", color="outline-primary",
                           size="sm", className="me-2"),
                html.Span(id="vs-candidate-info",
                          className="small text-muted",
                          style={"fontFamily":"'DM Mono',monospace"}),
                html.Span(id="vs-fmp-status",
                          className="small ms-3",
                          style={"fontFamily":"'DM Mono',monospace"}),
                dbc.Button("🔑 Test FMP key", id="vs-test-fmp-btn",
                           color="outline-secondary", size="sm",
                           className="ms-2"),
            ], width=12),
        ], className="mb-2"),

        dbc.Row([
            dbc.Col([
                html.Label("Tickers to screen (comma-separated)", className="fw-semibold"),
                dbc.Textarea(
                    id="vs-tickers",
                    placeholder="AAPL, MSFT, GOOGL, META, AMZN, NVDA, TSLA, BRK-B, JPM, JNJ, V, PG, UNH, HD, MA ...",
                    rows=3, className="mb-2",
                    style={"fontFamily":"'DM Mono',monospace","fontSize":"12px"},
                ),
            ], width=8),
            dbc.Col([
                html.Label([
                    "Min Value Score  ",
                    html.Span("ⓘ", id="vs-score-tip",
                              style={"cursor":"pointer","color":"#94a3b8","fontSize":"12px"}),
                ], className="fw-semibold"),
                dbc.Tooltip(
                    "Score 0–100 from PE, PEG, P/B, FCF yield, ROE, D/E, revenue growth. "
                    "Low score may mean sparse FMP data, not a bad stock. "
                    "Set to 0 to see everything. Check 'Coverage' column.",
                    target="vs-score-tip", placement="left",
                ),
                dbc.Input(id="vs-min-score", type="number", value=20, min=0, max=100, step=5,
                          className="mb-2"),
                html.Label("Require tech signal", className="fw-semibold"),
                dbc.Select(id="vs-tech-filter",
                    options=[
                        {"label":"Any","value":"any"},
                        {"label":"BUY only","value":"BUY"},
                        {"label":"BUY or WATCH","value":"BUY_WATCH"},
                    ], value="any", className="mb-2"),
            ], width=2),
            dbc.Col([
                html.Br(),
                dbc.Button("🔍 Screen", id="vs-btn", color="danger",
                           className="w-100 mt-1 mb-2"),
                dbc.Button("📋 S&P 500 Top 50", id="vs-preset-btn", color="outline-secondary",
                           size="sm", className="w-100 mb-2"),
                dbc.Button("🔄 Refresh Live", id="vs-refresh-btn", color="outline-warning",
                           size="sm", className="w-100",
                           title="Force re-fetch live prices for the tickers above (bypasses cache)"),
            ], width=2),
        ], className="mb-3"),
        html.Div(id="vs-refresh-status", className="mb-1"),
        html.Div(id="vs-status", className="mb-2"),
        html.Div(id="vs-results"),
        dcc.Store(id="vs-store", data={}),
        dcc.Store(id="vs-refresh-store", data={}),
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
                dbc.Tab(scanner_tab(),        label="🔭 Market Scanner", tab_id="scanner"),
                dbc.Tab(deepdive_tab(),        label="🔬 Deep Dive",      tab_id="deepdive"),
                dbc.Tab(value_screener_tab(), label="💎 Value Screen",   tab_id="valuescreen"),
            ], id="main-tabs", active_tab="scanner"),
        ], style={"flex":"1","padding":"20px","overflowY":"auto","background":"#f8f9fc"}),
    ], style={"display":"flex","height":"100vh","overflow":"hidden"}),
], style={"fontFamily":"'Segoe UI', sans-serif","background":"#f8f9fc","color":"#0f172a"})

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
    Output("filter-sector","options"),
    Output("filter-country","options"),
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

    _jdf  = jetf_df
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

    if not dist_opts:
        dist_opts = [{"label":"Accumulating","value":"Accumulating"},
                     {"label":"Distributing", "value":"Distributing"}]
    if not dom_opts:
        dom_opts = [{"label":v,"value":v} for v in
                    ["Ireland","Luxembourg","Germany","France","United States"]]
    if not repl_opts:
        repl_opts = [{"label":v,"value":v} for v in
                     ["Physical (Full)","Physical (Sampling)","Swap-based"]]

    cat_opts = uopts("category_group", "ETF")
    if not cat_opts:
        cat_opts = [{"label":v,"value":v} for v in [
            "Equities","Fixed Income","Commodities","Real Estate",
            "Alternatives","Cash","Currencies","Derivatives"]]

    # Smart country options — filter by preset's country hint if available
    preset_country = PRESETS.get(preset, {}).get("country", [])
    if preset_country:
        # Only show countries relevant to this preset
        ctry_opts = [{"label":v,"value":v} for v in preset_country]
    else:
        # Show all countries that actually have signals in signals_df
        if not signals_df.empty and "country" in signals_df.columns:
            sig_countries = sorted([v for v in signals_df["country"].dropna().unique()
                                    if v and v not in ("","nan","None")])
            ctry_opts = [{"label":v,"value":v} for v in sig_countries]
        else:
            ctry_opts = uopts("country", "Stock")

    # Sector options cascade from selected country
    sect_opts = uopts("sector", "Stock", selected_country)

    return types_style, etf_style, stock_style, sect_opts, ctry_opts

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
                    "PE":      f"{pe:.1f}" if pe and str(pe) != 'nan' else "—",
                    "Beta":    f"{beta:.2f}" if beta and str(beta) != 'nan' else "—",
                    "Div%":    f"{div*100:.1f}%" if div and str(div) != 'nan' else "—",
                    "MCap":    f"${mcap/1e9:.1f}B" if mcap and str(mcap) != 'nan' else "—",
                    "VGrade":  str(r.get("value_grade","")) if pd.notna(r.get("value_grade",None)) and str(r.get("value_grade","")) not in ("","nan","None") else "—",
                    "VScore":  int(r["value_score"]) if pd.notna(r.get("value_score",None)) and str(r.get("value_score","")) not in ("","nan","None") else None,
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
    PANEL_STYLES = {
        "BUY":   {"color": "#0d9488", "bg": "#f0fdfa",  "border": "#99f6e4"},
        "WATCH": {"color": "#0284c7", "bg": "#f0f9ff",  "border": "#bae6fd"},
        "SELL":  {"color": "#dc2626", "bg": "#fef2f2",  "border": "#fecaca"},
    }
    PANEL_LABELS = {"BUY": "● Top BUY", "WATCH": "◎ WATCH", "SELL": "● Top SELL"}

    def top_card(action):
        ps    = PANEL_STYLES[action]
        label = PANEL_LABELS[action]
        sub   = df[df["Action"]==action][["Rank","Ticker","Name","Dist%","RSI","Conf"]].head(8)
        if sub.empty:
            return dbc.Col(dbc.Card([
                dbc.CardHeader(label, style={"background":ps["bg"],"color":ps["color"],
                                             "fontWeight":"700","fontSize":"10px",
                                             "letterSpacing":"0.1em","textTransform":"uppercase",
                                             "borderBottom":f"1px solid {ps['border']}",
                                             "padding":"8px 12px"}),
                dbc.CardBody(html.P("No signals", className="text-muted mb-0", style={"fontSize":"12px"}))],
                style={"background":ps["bg"],"border":f"1px solid {ps['border']}","borderRadius":"6px"}), width=4)
        tbl = dash_table.DataTable(
            data=sub.round(2).to_dict("records"),
            columns=[{"name":c,"id":c} for c in sub.columns],
            style_table={"overflowX":"auto"},
            style_cell={"background":"transparent","color":"#0f172a",
                       "border":"none","fontSize":"11px","padding":"3px 8px",
                       "fontFamily":"DM Mono,monospace","textAlign":"left"},
            style_header={"background":"transparent","color":"#94a3b8",
                         "fontWeight":"600","border":"none","fontSize":"9px",
                         "textTransform":"uppercase","letterSpacing":"0.07em",
                         "textAlign":"left"},
        )
        return dbc.Col(dbc.Card([
            dbc.CardHeader(label, style={"background":"transparent","color":ps["color"],
                                         "fontWeight":"600","fontSize":"11px",
                                         "letterSpacing":"0.08em","textTransform":"uppercase",
                                         "borderBottom":f"1px solid {ps['border']}","padding":"10px 14px"}),
            dbc.CardBody(tbl, className="p-0"),
        ], style={"background":ps["bg"],"border":f"1px solid {ps['border']}","borderRadius":"6px"}), width=4)

    top = dbc.Row([
        top_card("BUY"),
        top_card("WATCH"),
        top_card("SELL"),
    ], className="g-2")

    # Full table — global rank, no re-ranking on tab switch
    if active_tab == "all":
        sub = df
    elif active_tab == "VALUE_A":
        if "VGrade" in df.columns:
            sub = df[df["VGrade"] == "A"]
            if sub.empty:
                # No A grades in this scan — show message
                return kpis, top, dbc.Alert(
                    "💎 No Grade A stocks in current scan. "
                    "Run build_signals.py locally and commit the new signals.csv to populate this tab. "
                    "Grade A = Value Score ≥ 75 (PE < 15, P/B < 1.5, FCF yield > 6%, ROE > 20%).",
                    color="info", className="mt-3")
        else:
            return kpis, top, dbc.Alert(
                "💎 Value grades not yet computed. "
                "Run build_signals.py locally and commit signals.csv, then redeploy.",
                color="info", className="mt-3")
    elif active_tab == "VALUE_B":
        if "VGrade" in df.columns:
            sub = df[df["VGrade"].isin(["A","B"])]
            if sub.empty:
                return kpis, top, dbc.Alert(
                    "🔷 No Grade A or B stocks in current scan. "
                    "Run build_signals.py locally and commit the new signals.csv.",
                    color="info", className="mt-3")
        else:
            return kpis, top, dbc.Alert(
                "🔷 Value grades not yet computed. "
                "Run build_signals.py locally and commit signals.csv, then redeploy.",
                color="info", className="mt-3")
    else:
        sub = df[df["Action"] == active_tab]

    SHOW_COLS = ["Rank","Ticker","Name","ISIN","Price","MA200","Dist%","52W%",
                 "RSI","RSI↗","MACD","MACD⚡","Vol%","Conf",
                 "PE","Beta","Div%","MCap","VGrade",
                 "Signal","Knife","Allocation"]
    show_cols = [c for c in SHOW_COLS if c in sub.columns]

    # Round floats to avoid 2.8200000000000003 display issues
    sub = sub.copy()
    for col in ["Price","MA200","Dist%","52W%","RSI","Vol%","Conf","Score"]:
        if col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce").round(2)
    # Replace NaN strings with clean dash
    for col in ["PE","Beta","Div%","MCap"]:
        if col in sub.columns:
            sub[col] = sub[col].replace({"nan%":"—","$nanB":"—","nan":"—",float("nan"):"—"})
            sub[col] = sub[col].fillna("—")

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
            "background":"#1a1a2e","color":"#0f172a",
            "border":"1px solid #2a2a3e",
            "fontSize":"12px","padding":"6px 10px",
            "whiteSpace":"nowrap","overflow":"hidden",
            "textOverflow":"ellipsis","maxWidth":"180px",
        },
        style_cell_conditional=[
            {"if":{"column_id":"Name"},   "maxWidth":"180px","minWidth":"120px",
             "textAlign":"left","overflow":"hidden","textOverflow":"ellipsis"},
            {"if":{"column_id":"Ticker"}, "maxWidth":"72px","fontWeight":"700",
             "textAlign":"left","color":"#0f172a"},
            {"if":{"column_id":"Signal"}, "maxWidth":"130px","textAlign":"left"},
            {"if":{"column_id":"ISIN"},   "maxWidth":"115px","fontFamily":"'DM Mono',monospace",
             "fontSize":"11px","textAlign":"left"},
            {"if":{"column_id":"Allocation"},"maxWidth":"140px","textAlign":"left"},
            {"if":{"column_id":"MACD"},   "textAlign":"left"},
        ],
        style_header={
            "background":"#f8f9fc","color":"#64748b",
            "fontWeight":"600","border":"none",
            "borderBottom":"1px solid #e2e6ed",
        },
        style_data_conditional=[
            {"if":{"row_index":"odd"},"background":"#fafbfc"},
            *cond_styles,
            {"if":{"column_id":"Dist%","filter_query":"{Dist%} < 0"},"color":"#0d9488","fontWeight":"600"},
            {"if":{"column_id":"Dist%","filter_query":"{Dist%} > 0"},"color":"#d97706","fontWeight":"600"},
            {"if":{"column_id":"RSI","filter_query":"{RSI} < 35"},"color":"#0d9488","fontWeight":"600"},
            {"if":{"column_id":"RSI","filter_query":"{RSI} > 65"},"color":"#dc2626","fontWeight":"600"},
            {"if":{"state":"selected"},"background":"#e8f0fe","border":"1px solid #1a56db"},
        ],
        style_filter={"background":"#f8f9fc","color":"#0f172a",
                      "border":"none","borderBottom":"1px solid #e2e6ed"},
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
    State("signal-tabs","active_tab"),
)
def enable_dive_btn(selected_rows, store_data, active_tab):
    if not selected_rows or not store_data:
        return True, "🔬 Deep Dive selected ticker →"
    import io
    df  = pd.read_json(io.StringIO(store_data), orient="split")
    # Filter to active tab first — same as render_results does
    sub = df if active_tab == "all" or not active_tab else df[df["Action"]==active_tab]
    idx = selected_rows[0]
    if idx >= len(sub):
        return True, "🔬 Deep Dive selected ticker →"
    ticker = sub.iloc[idx]["Ticker"]
    return False, f"🔬 Deep Dive: {ticker} →"

@app.callback(
    Output("main-tabs","active_tab"),
    Output("dd-input","value"),
    Input("goto-dive-btn","n_clicks"),
    State("main-table","selected_rows"),
    State("scan-store","data"),
    State("signal-tabs","active_tab"),
    prevent_initial_call=True,
)
def open_deep_dive(n_clicks, selected_rows, store_data, active_tab):
    if not n_clicks or not selected_rows or not store_data:
        return no_update, no_update
    import io
    df  = pd.read_json(io.StringIO(store_data), orient="split")
    sub = df if active_tab == "all" or not active_tab else df[df["Action"]==active_tab]
    idx = selected_rows[0]
    if idx >= len(sub):
        return no_update, no_update
    ticker = sub.iloc[idx]["Ticker"]
    return "deepdive", ticker

@app.callback(
    Output("dd-from-scanner","children"),
    Input("dd-scanner-store","data"),
)
def show_scanner_context(store):
    if store and store.get("ticker"):
        return dbc.Alert(
            f"📡 **{store['ticker']}** loaded from Scanner — click Analyse below.",
            color="info", className="py-1")
    return ""


@app.callback(
    Output("dd-scanner-store","data"),
    Input("goto-dive-btn","n_clicks"),
    State("main-table","selected_rows"),
    State("scan-store","data"),
    State("signal-tabs","active_tab"),
    prevent_initial_call=True,
)
def update_scanner_store(goto_clicks, selected_rows, store_data, active_tab):
    if not goto_clicks or not selected_rows or not store_data:
        return {}
    import io
    df  = pd.read_json(io.StringIO(store_data), orient="split")
    sub = df if active_tab == "all" or not active_tab else df[df["Action"]==active_tab]
    idx = selected_rows[0]
    if idx < len(sub):
        return {"ticker": sub.iloc[idx]["Ticker"]}
    return {}

# ───────────────────────────────────────────────────────────────────
# CALLBACKS — Deep Dive
# ───────────────────────────────────────────────────────────────────

def _price_currency(yf_sym):
    """Infer currency symbol from yfinance ticker suffix."""
    s = str(yf_sym or "")
    eur_sfx = (".DE", ".L", ".AS", ".PA", ".MI", ".BR", ".VI",
               ".LS", ".ST", ".CO", ".HE", ".OL", ".MC", ".WA")
    if any(s.endswith(sfx) for sfx in eur_sfx):
        return "€"
    if s.endswith(".SW"):
        return "CHF "
    if s.endswith(".AX"):
        return "A$"
    if s.endswith(".HK"):
        return "HK$"
    if s.endswith(".T"):
        return "¥"
    return "$"


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
    try:
        return _run_deep_dive_inner(n_clicks, user_input, budget)
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(f"[deep_dive ERROR] {exc}", flush=True)
        print(tb, flush=True)
        return dbc.Alert([
            html.Strong("Deep Dive error — "),
            html.Span(str(exc)),
            html.Br(),
            html.Small(tb[:500], style={"fontFamily":"monospace","fontSize":"10px",
                                         "color":"#94a3b8","whiteSpace":"pre-wrap"}),
        ], color="danger")


def _fundamental_score_adjustment(pe=None, pb=None, div=None,
                                   eps=None, mc=None, beta=None,
                                   asset_type="Stock"):
    """Score delta [-0.15, +0.15] from fundamentals. Zero when data missing."""
    if asset_type == "ETF":
        return 0.0
    delta = 0.0
    if pe  is not None:
        if   pe <= 0:       delta -= 0.06
        elif pe < 12:       delta += 0.04
        elif pe < 25:       delta += 0.02
        elif pe >= 50:      delta -= 0.04
    if pb  is not None:
        if   pb <= 0:       delta -= 0.02
        elif pb < 1.5:      delta += 0.02
        elif pb > 4:        delta -= 0.03
    if eps is not None:
        delta += 0.02 if eps > 0 else -0.03
    if div is not None:
        delta += 0.02 if div > 0.04 else (0.01 if div > 0.01 else 0)
    if mc  is not None:
        if   mc < 50e6:     delta -= 0.04
        elif mc < 300e6:    delta -= 0.02
        elif mc > 10e9:     delta += 0.01
    if beta is not None:
        if   beta > 3.0:    delta -= 0.03
        elif beta > 2.0:    delta -= 0.01
        elif 0.4 < beta < 1.5: delta += 0.01
    return max(-0.15, min(0.15, round(delta, 4)))


def _run_deep_dive_inner(n_clicks, user_input, budget):
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
                # MA200 slope for justETF path
                _ma200_tail2 = ma200_s.dropna().tail(60)
                if len(_ma200_tail2) >= 20:
                    _y2 = _ma200_tail2.values
                    _x2 = np.arange(len(_y2))
                    _sr2 = np.polyfit(_x2, _y2, 1)[0]
                    _ma200_slope2 = float(_sr2 / (abs(_y2.mean()) + 1e-9))
                else:
                    _ma200_slope2 = 0.0
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
                    "ma200_slope": _ma200_slope2,
                    "source": "justETF",
                }
                # Write to tick_ cache so subsequent calls (scanner writeback, re-open) hit cache
                cache_set(f"tick_{ticker}", raw, ttl=3600)

    if raw is None:
        raw = fetch_ticker_data(resolved_yf or ticker, isin=isin)
    if raw is None:
        try:
            sr = yf.Search(ticker, max_results=8)
            if hasattr(sr,"quotes") and sr.quotes:
                # Prefer exact match or US-listed symbols — filter out foreign exchange duplicates
                quotes = sr.quotes
                # Prioritise: exact symbol match first, then symbols without dot (US), then others
                def _rank(q):
                    sym = q.get("symbol","")
                    if sym == ticker: return 0
                    if "." not in sym: return 1
                    if sym.startswith(ticker + "."): return 2
                    return 3
                quotes_sorted = sorted(quotes, key=_rank)
                sugg = ", ".join([q["symbol"] for q in quotes_sorted[:3]])
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

    cur_p       = raw["price"]
    curr_sym    = _price_currency(resolved_yf or ticker)
    dist_ma     = raw["dist_ma"]
    rsi_val     = raw["rsi"]
    rsi_rising  = raw["rsi_slope"] > 0
    macd_bull   = raw["macd"] > raw["macd_signal"]
    macd_accel  = raw["macd_hist"] > 0
    conf        = raw["confidence"]
    vol         = raw["vol"]
    dist_52h    = raw["dist_52h"]
    is_knife    = raw["dist_ma"] < -15 and rsi_val < 35 and raw["trend_down_strong"]
    reversal    = is_knife and macd_bull and rsi_rising
    ma200_now   = raw["ma200"]
    ma200_slope = raw.get("ma200_slope", 0.0)

    # ── MA200 slope interpretation ────────────────────────────────────
    # Slope is normalised: +0.001 per bar = rising trend, -0.003 = steep decline
    ma200_trend = (
        "rising"     if ma200_slope >  0.001 else
        "falling"    if ma200_slope < -0.001 else
        "flat"
    )
    ma200_steep_down = ma200_slope < -0.003

    vix        = get_live_vix()
    fg, fg_lbl = get_fg_index()

    # Fetch fundamentals early so score writeback can use them
    pe_data_early = fetch_pe_data(resolved_yf or ticker)
    _pe   = pe_data_early.get("PE")
    _beta = pe_data_early.get("Beta")
    _div  = pe_data_early.get("Div%")
    _mc   = pe_data_early.get("MCap")
    def _parse_fund(v):
        try:
            s = str(v).replace("$","").replace("B","e9").replace("M","e6").replace("%","")
            return float(s)
        except: return None
    fund_delta_live = _fundamental_score_adjustment(
        pe   = _parse_fund(_pe),
        pb   = None,   # not in pe_data dict — would need separate fetch
        div  = _parse_fund(_div) / 100 if _parse_fund(_div) else None,
        eps  = None,
        mc   = _parse_fund(_mc),
        beta = _parse_fund(_beta),
        asset_type = "ETF" if is_etf else "Stock",
    )

    # ── Write fresh signal back to in-memory signals_df ──────────────
    # This keeps scanner results consistent with Deep Dive live data
    global signals_df
    try:
        from datetime import date
        # Mirror the fixed scoring from build_signals.py
        # Skip structural collapses (same -55% hard cap as build_signals)
        if dist_ma < -55:
            action2, score2 = "AVOID", -2.0
        else:
            pass  # classification follows
        knife_thr2 = -25
        is_knife2  = (dist_ma < knife_thr2) and (15 < rsi_val < 35) and raw["trend_down_strong"]
        reversal2  = (is_knife2 and macd_bull and rsi_rising and rsi_val > 25 and not raw["trend_down_strong"])
        if is_knife2 and not reversal2:
            action2 = "AVOID"
        elif (-40 < dist_ma < -10) and (30 <= rsi_val < 48) and macd_bull and macd_accel:
            action2 = "BUY"
        elif (-40 < dist_ma < -5) and rsi_val < 48 and (macd_bull or rsi_rising):
            action2 = "WATCH"
        elif dist_ma < -40 and rsi_val > 28 and macd_bull and rsi_rising:
            action2 = "WATCH"
        elif dist_ma > 10 and rsi_val > 70:
            action2 = "SELL"
        else:
            action2 = "WAIT"
        sweet2       = -20
        dist_pen2    = max(0, (-dist_ma - 40) / 30)
        dist_s2      = max(0, 1 - abs(dist_ma - sweet2) / 25) * (1 - min(dist_pen2, 1))
        rsi_s2_live  = max((50 - rsi_val) / 50, 0) if rsi_val >= 25 else 0
        score2       = (dist_s2*0.35 + rsi_s2_live*0.25 +
                        (0.20 if macd_bull else 0) + (0.10 if macd_accel else 0) + conf*0.10)
        if action2 == "AVOID": score2 -= 2
        if is_knife2:          score2 -= 0.3
        score2 += fund_delta_live
        # Fundamental-driven action adjustments
        if action2 == "WATCH" and fund_delta_live >= 0.08:  action2 = "BUY"
        if action2 == "BUY"   and fund_delta_live <= -0.08: action2 = "WATCH"

        # MA200 slope adjustments — is the long-term trend itself healthy?
        if ma200_steep_down:
            score2 -= 0.10
            if action2 == "BUY":   action2 = "WATCH"   # don't BUY into falling MA200
            if action2 == "WATCH": action2 = "WAIT"    # steep decline → wait for floor
        elif ma200_trend == "falling":
            score2 -= 0.05
            if action2 == "BUY":   action2 = "WATCH"   # gently falling → downgrade
        elif ma200_trend == "rising":
            score2 += 0.05                              # dip into rising trend = ideal

        # Category / strategy hard flag — leveraged/inverse products
        if hard_flagged:
            score2 -= 0.20
            if action2 in ("BUY", "WATCH"): action2 = "AVOID"
        elif soft_flagged:
            score2 -= 0.08                              # carry/VIX/futures — soft penalty

        new_row = {
            "ticker": ticker, "yf_symbol": resolved_yf or ticker,
            "data_source": "live_deepdive", "price": round(cur_p, 4),
            "ma200": round(ma200_now, 4), "dist_ma200": round(dist_ma, 2),
            "rsi": round(rsi_val, 2), "rsi_rising": int(rsi_rising),
            "macd_bull": int(macd_bull), "macd_accel": int(macd_accel),
            "vol_pct": round(vol, 4), "conf": round(conf, 4),
            "action": action2, "score": round(score2, 4),
            "is_knife": int(is_knife2), "reversal": int(reversal2),
            "computed_at": date.today().isoformat(),
        }
        if not signals_df.empty and "ticker" in signals_df.columns:
            signals_df = signals_df[signals_df["ticker"] != ticker]
            signals_df = pd.concat(
                [pd.DataFrame([new_row]), signals_df], ignore_index=True)
            print(f"[deepdive] Updated {ticker} in signals_df: {action2} score={score2:.3f}", flush=True)
    except Exception as e:
        print(f"[deepdive] Signal writeback failed: {e}", flush=True)
    # ── End writeback ─────────────────────────────────────────────────

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

    # ── Category / strategy flag for ETFs (needs name + jetf_meta) ───
    HARD_FLAG_KW = ["leveraged","2x ","3x ","4x ","-2x","-3x",
                    " short ","inverse","daily short","daily long"]
    SOFT_FLAG_KW = ["carry","volatility","vix","futures","roll",
                    "enhanced","momentum tilt","swap-based carry"]
    etf_name_lower  = name.lower() if name else ""
    etf_strat_lower = str(jetf_meta.get("strategy","")).lower()
    combined_text   = etf_name_lower + " " + etf_strat_lower

    hard_flagged = is_etf and any(kw in combined_text for kw in HARD_FLAG_KW)
    soft_flagged = is_etf and (not hard_flagged) and any(
        kw in combined_text for kw in SOFT_FLAG_KW)
    flag_reason  = next(
        (kw for kw in HARD_FLAG_KW + SOFT_FLAG_KW if kw.strip() in combined_text),
        ""
    ).strip()

    # PE data (yfinance primary)
    pe_data = pe_data_early  # already fetched above for score writeback

    # FMP second opinion — async-style: fetch in background, show if available
    fmp_data    = fetch_fmp_fundamentals(resolved_yf or ticker) if _get_fmp_key() else {}

    # For value score: use yfinance fundamentals if FMP has no coverage
    _yf_fund = {}
    if not is_etf:
        _cached_yfund = cache_get(f"yfund_{resolved_yf or ticker}")
        if _cached_yfund:
            _yf_fund = _cached_yfund
        elif not fmp_data:
            _yf_fund = fetch_yf_fundamentals(resolved_yf or ticker, timeout=6)

    # Merge: FMP takes priority where available, yfinance fills gaps
    _combined_fund = {**_yf_fund, **fmp_data}
    value_score, value_grade, value_bdown, value_coverage = compute_value_score(_combined_fund)
    value_available = value_score > 0 and not is_etf

    # Conviction signals (cached 24h — analyst, short interest, EPS beats, insider)
    conviction = {} if is_etf else fetch_conviction_signals(resolved_yf or ticker, timeout=6)
    conv_grade  = conviction.get("conviction_grade", "—")
    conv_labels = conviction.get("conviction_labels", [])
    conv_score  = conviction.get("conviction_score", 50)

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

    # ── Buy decision (enhanced with value score) ─────────────────────
    def _val_suffix():
        if not value_available: return ""
        if value_grade == "A":  return f" · 🏆 Value A ({value_score}/100)"
        if value_grade == "B":  return f" · ✅ Value B ({value_score}/100)"
        if value_grade == "D":  return f" · ⚠️ Expensive ({value_score}/100)"
        return ""

    if hard_flagged:
        buy_el = dbc.Alert("⛔ AVOID — Leveraged/inverse strategy. Not suitable for dip buying.", color="danger")
    elif is_knife and not reversal:
        buy_el = dbc.Alert("⛔ AVOID — Falling knife. Wait for reversal.", color="danger")
    elif entry=="WAIT":
        buy_el = dbc.Alert("⏳ WAIT — Still falling. Let it stabilise.", color="warning")
    elif entry=="WATCH":
        if buy_score>=70:
            buy_el = dbc.Alert(f"⚖️ PREPARE — Good setup (~€{budget:,.0f}){_val_suffix()}", color="info")
        else:
            buy_el = dbc.Alert(f"🔍 WATCH — No strong entry yet.{_val_suffix()}", color="secondary")
    else:
        if value_available and value_grade == "A" and buy_score >= 60:
            buy_el = dbc.Alert(f"🏆 CONVICTION BUY — Technically ready + fundamentally cheap · €{budget*2:,.0f}", color="success")
        elif buy_score>=70:
            buy_el = dbc.Alert(f"🔥 BUY — €{budget*2:,.0f}{_val_suffix()}", color="success")
        elif buy_score>=40:
            buy_el = dbc.Alert(f"⚖️ BUY — €{budget:,.0f}{_val_suffix()}", color="success")
        else:
            buy_el = dbc.Alert(f"⚠️ LIGHT — €{budget*0.5:,.0f}{_val_suffix()}", color="warning")
    if reversal:
        buy_el = html.Div([buy_el, dbc.Alert("✅ Reversal confirmed (MACD + RSI turned bullish)", color="success")])

    if sell_score>=70:   sell_el = dbc.Alert("🚨 STRONG SELL — Reduce significantly.", color="danger")
    elif sell_score>=40: sell_el = dbc.Alert("⚠️ TRIM — Partial reduction.", color="warning")
    else:                sell_el = dbc.Alert("🟢 HOLD — No sell signal.", color="success")

    # MA200 trend label for KPI card
    ma200_trend_label = (
        "↗ Rising"  if ma200_trend == "rising"  else
        "↘ Falling" if ma200_trend == "falling" else
        "→ Flat"
    )
    ma200_trend_color = (
        "#00e676" if ma200_trend == "rising"      else
        "#ff6d00" if ma200_steep_down             else
        "#ffd600" if ma200_trend == "falling"     else
        "#e8edf5"
    )

    # Value score KPI colour
    value_color = (
        "#00e676" if value_grade == "A" else
        "#00bcd4" if value_grade == "B" else
        "#ffd600" if value_grade == "C" else
        "#ff6d00" if value_grade == "D" else
        "#e8edf5"
    )
    value_label = f"{value_score}/100 ({value_grade})" if value_available else "N/A"

    # ── Metrics
    metrics = dbc.Row([
        dbc.Col(kpi_card("Price",       f"{curr_sym}{cur_p:.2f}"), width=2),
        dbc.Col(kpi_card("vs MA200",    f"{dist_ma:+.1f}%",  "#00e676" if dist_ma<0 else "#ff6d00"), width=2),
        dbc.Col(kpi_card("MA200 Trend", ma200_trend_label, ma200_trend_color), width=2),
        dbc.Col(kpi_card("RSI",         f"{rsi_val:.1f} {'↗' if rsi_rising else '↘'}"), width=2),
        dbc.Col(kpi_card("MACD",        "▲ Bull" if macd_bull else "▼ Bear", "#00e676" if macd_bull else "#ff6d00"), width=2),
        dbc.Col(kpi_card("Value Score", value_label, value_color), width=2),
    ], className="g-2 mb-3")

    # ── Warning banners for category flags and MA200 slope ────────────
    flag_banners = []
    if hard_flagged:
        flag_banners.append(dbc.Alert(
            [html.Strong("⚠️ Strategy warning: "),
             f"This ETF uses a '{flag_reason}' strategy — leveraged or inverse products "
             f"can structurally decay over time. Technical dip signals are unreliable here."],
            color="danger", className="py-2 mb-2"))
    elif soft_flagged:
        flag_banners.append(dbc.Alert(
            [html.Strong("⚠️ Strategy note: "),
             f"This ETF involves '{flag_reason}' — these strategies can experience "
             f"structural decay unrelated to broader market cycles. Verify the thesis before acting."],
            color="warning", className="py-2 mb-2"))
    if ma200_steep_down:
        flag_banners.append(dbc.Alert(
            [html.Strong("📉 MA200 in steep decline — "),
             f"The 200-day moving average has been falling sharply "
             f"(slope {ma200_slope:+.4f}). This is a structural downtrend, not a cyclical dip. "
             f"Wait for MA200 to flatten before considering entry."],
            color="warning", className="py-2 mb-2"))
    elif ma200_trend == "falling":
        flag_banners.append(dbc.Alert(
            [html.Strong("⚠️ MA200 declining — "),
             f"Long-term trend is weakening (slope {ma200_slope:+.4f}). "
             f"Signal downgraded. Look for MA200 to stabilise first."],
            color="secondary", className="py-2 mb-2"))
    flag_banner_el = html.Div(flag_banners) if flag_banners else None

    # justETF strip
    meta_parts = []
    if jetf_meta.get("domicile"):     meta_parts.append(f"🏳️ {jetf_meta['domicile']}")
    if jetf_meta.get("ter"):          meta_parts.append(f"💸 TER {jetf_meta['ter']:.2f}%")
    if jetf_meta.get("fund_size_eur"):meta_parts.append(f"📦 €{jetf_meta['fund_size_eur']:,.0f}m")
    if jetf_meta.get("dist_policy"):  meta_parts.append(f"💰 {jetf_meta['dist_policy']}")
    if jetf_meta.get("replication"):  meta_parts.append(f"🔄 {jetf_meta['replication']}")
    meta_strip = html.P("  ·  ".join(meta_parts), className="text-muted small") if meta_parts else None

    # ── Fundamentals: yfinance primary + FMP second opinion ──────────
    def _fmt_pct(v, mult=100):
        try: return f"{float(v)*mult:.1f}%"
        except: return "—"
    def _fmt_x(v):
        try: return f"{float(v):.1f}x"
        except: return "—"
    def _fmt_money(v):
        try:
            f = float(v)
            return f"${f/1e9:.1f}B" if f>=1e9 else f"${f/1e6:.0f}M"
        except: return "—"
    def _fmt_num(v, dec=1):
        try: return f"{float(v):.{dec}f}"
        except: return "—"

    # Primary fundamentals row — ETFs and stocks show different fields
    def _fund_kpi(label, value):
        return dbc.Col([
            html.Div(label, style={"fontSize":"9px","color":"#94a3b8","fontWeight":"600",
                                   "textTransform":"uppercase","letterSpacing":"0.08em"}),
            html.Div(str(value), style={"fontSize":"14px","fontWeight":"700","color":"#0f172a",
                                        "fontFamily":"'DM Mono',monospace"}),
        ], width="auto", className="me-3")

    # Fundamental score delta badge
    def _fund_badge(delta):
        if delta == 0:  return None
        color = "#0d9488" if delta > 0 else "#dc2626"
        label = f"{'▲' if delta > 0 else '▼'} Fund adj {delta:+.2f}"
        return html.Span(label, style={
            "fontSize":"10px","fontWeight":"700","color":color,
            "background": "#f0fdf4" if delta > 0 else "#fff1f2",
            "border": f"1px solid {'#86efac' if delta > 0 else '#fecaca'}",
            "borderRadius":"4px","padding":"2px 7px","marginLeft":"10px",
            "fontFamily":"'DM Mono',monospace",
        })

    if is_etf:
        ter_val  = (f"{jetf_meta['ter']:.2f}%" if jetf_meta.get("ter") else
                    pe_data.get("Div%","—"))
        size_val = (f"€{jetf_meta['fund_size_eur']:,.0f}m"
                    if jetf_meta.get("fund_size_eur") else "—")
        dist_val = jetf_meta.get("dist_policy","—") or "—"
        repl_val = jetf_meta.get("replication","—") or "—"
        fund_row = dbc.Row([
            _fund_kpi("TER",          ter_val),
            _fund_kpi("Fund Size",    size_val),
            _fund_kpi("Distribution", dist_val),
            _fund_kpi("Replication",  repl_val),
        ], className="mb-2 align-items-end")
    else:
        pe_val   = pe_data.get("PE","—")
        beta_val = pe_data.get("Beta","—")
        div_val  = pe_data.get("Div%","—")
        mcap_val = pe_data.get("MCap","—")
        badge    = _fund_badge(fund_delta_live)
        fund_row = dbc.Row([
            _fund_kpi("PE",   pe_val),
            _fund_kpi("Beta", beta_val),
            _fund_kpi("Div%", div_val),
            _fund_kpi("MCap", mcap_val),
            dbc.Col(badge, width="auto", className="align-self-center") if badge else None,
        ], className="mb-2 align-items-end")

    # FMP second opinion section
    fmp_section = None
    if fmp_data:
        fmp_items = []

        # Valuation block
        val_items = []
        if fmp_data.get("fmp_pe_ttm"):
            val_items.append(("Trailing PE", _fmt_num(fmp_data["fmp_pe_ttm"])))
        if fmp_data.get("fmp_pb"):
            val_items.append(("P/B", _fmt_num(fmp_data["fmp_pb"])))
        if fmp_data.get("fmp_ev_ebitda"):
            val_items.append(("EV/EBITDA", _fmt_num(fmp_data["fmp_ev_ebitda"])))
        if fmp_data.get("fmp_div_yield"):
            val_items.append(("Div Yield", _fmt_pct(fmp_data["fmp_div_yield"])))
        if fmp_data.get("fmp_debt_eq"):
            val_items.append(("Debt/Eq", _fmt_num(fmp_data["fmp_debt_eq"])))
        if fmp_data.get("fmp_roe"):
            val_items.append(("ROE", _fmt_pct(fmp_data["fmp_roe"])))
        if fmp_data.get("fmp_fcf_yield"):
            val_items.append(("FCF Yield", _fmt_pct(fmp_data["fmp_fcf_yield"])))

        # Analyst target block
        target_items = []
        target_mean = fmp_data.get("fmp_target_mean") or fmp_data.get("fmp_target_med")
        if target_mean:
            upside = ((float(target_mean) - cur_p) / cur_p * 100) if cur_p else 0
            color  = "#0d9488" if upside > 0 else "#dc2626"
            target_items.append(html.Span([
                html.Span("Analyst Target  ", style={"fontSize":"11px","color":"#64748b"}),
                html.Span(f"{curr_sym}{float(target_mean):.2f}", style={"fontWeight":"700","color":"#0f172a",
                                                                 "fontFamily":"'DM Mono',monospace"}),
                html.Span(f"  {upside:+.1f}%", style={"color":color,"fontWeight":"600",
                                                        "marginLeft":"4px","fontFamily":"'DM Mono',monospace"}),
            ]))
            if fmp_data.get("fmp_target_low") and fmp_data.get("fmp_target_high"):
                target_items.append(html.Span(
                    f"  Range {curr_sym}{float(fmp_data['fmp_target_low']):.2f}–{curr_sym}{float(fmp_data['fmp_target_high']):.2f}",
                    style={"fontSize":"11px","color":"#94a3b8","fontFamily":"'DM Mono',monospace"}
                ))

        # Earnings date
        if fmp_data.get("fmp_next_earnings"):
            target_items.append(html.Span([
                html.Span("  ·  Next Earnings  ", style={"fontSize":"11px","color":"#64748b"}),
                html.Span(fmp_data["fmp_next_earnings"],
                         style={"fontWeight":"600","color":"#0284c7",
                                "fontFamily":"'DM Mono',monospace","fontSize":"12px"}),
            ]))

        # Build the section
        if val_items or target_items:
            val_cols = [
                dbc.Col([
                    html.Div(label, style={"fontSize":"9px","color":"#94a3b8","fontWeight":"600",
                                          "textTransform":"uppercase","letterSpacing":"0.07em"}),
                    html.Div(val, style={"fontSize":"13px","fontWeight":"700","color":"#0f172a",
                                         "fontFamily":"'DM Mono',monospace"}),
                ], width="auto", className="me-3")
                for label, val in val_items
            ]

            fmp_section = html.Div([
                html.Div([
                    html.Span("📊 Second Opinion  ", style={"fontSize":"10px","fontWeight":"700",
                                                             "color":"#1a56db","letterSpacing":"0.06em",
                                                             "textTransform":"uppercase"}),
                    html.Span("via Financial Modeling Prep", style={"fontSize":"10px","color":"#94a3b8"}),
                ], className="mb-2"),
                dbc.Row(val_cols, className="mb-1 align-items-end") if val_cols else None,
                html.Div(target_items, className="mb-1") if target_items else None,
            ], style={"background":"#f0f9ff","border":"1px solid #bae6fd",
                      "borderRadius":"6px","padding":"10px 14px","marginBottom":"10px"})

    fund_strip = None  # replaced by fund_row above

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
        *([ html.Code(isin, style={"marginLeft":"8px","color":"#94a3b8","fontSize":"11px",
            "background":"#ffffff","padding":"2px 6px","borderRadius":"4px"})
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
        style_cell={"background":"#1a1a2e","color":"#0f172a","border":"1px solid #2a2a3e","fontSize":"12px","padding":"6px"},
        style_header={"background":"#f8f9fc","color":"#00bcd4","fontWeight":"bold","border":"1px solid #2a2a3e"},
    )

    # ── Value breakdown panel ─────────────────────────────────────────
    value_panel = None
    if value_available and value_bdown:
        grade_colors = {"A":"#0d9488","B":"#0284c7","C":"#d97706","D":"#dc2626","N/A":"#94a3b8"}
        gc = grade_colors.get(value_grade, "#94a3b8")
        metric_cols = []
        label_map = {
            "PE":"P/E","PEG":"PEG","P/B":"P/B",
            "FCF Yield":"FCF Yield","ROE":"ROE",
            "D/E":"Debt/Eq","Rev Growth":"Rev Growth",
        }
        for m, pts in value_bdown.items():
            if pts is None: continue
            bar_color = "#0d9488" if pts>=70 else "#0284c7" if pts>=50 else "#d97706" if pts>=30 else "#dc2626"
            metric_cols.append(
                dbc.Col([
                    html.Div(label_map.get(m,m),
                             style={"fontSize":"9px","color":"#64748b","fontWeight":"600",
                                    "textTransform":"uppercase","letterSpacing":"0.06em"}),
                    html.Div(f"{pts}/100",
                             style={"fontSize":"13px","fontWeight":"700","color":bar_color,
                                    "fontFamily":"'DM Mono',monospace"}),
                ], width="auto", className="me-3")
            )
        # Growth context row
        growth_parts = []
        if fmp_data.get("fmp_rev_growth") is not None:
            rg = fmp_data["fmp_rev_growth"]*100
            growth_parts.append(f"Rev {rg:+.1f}%")
        if fmp_data.get("fmp_eps_growth") is not None:
            eg = fmp_data["fmp_eps_growth"]*100
            growth_parts.append(f"EPS {eg:+.1f}%")
        if fmp_data.get("fmp_peg") is not None:
            growth_parts.append(f"PEG {fmp_data['fmp_peg']:.2f}")
        growth_str = "  ·  ".join(growth_parts)

        value_panel = html.Div([
            html.Div([
                html.Span("📊 Value Score  ", style={"fontSize":"10px","fontWeight":"700",
                                                      "color":gc,"letterSpacing":"0.06em",
                                                      "textTransform":"uppercase"}),
                html.Span(f"{value_score}/100", style={"fontSize":"16px","fontWeight":"800",
                                                         "color":gc,"fontFamily":"'DM Mono',monospace"}),
                html.Span(f"  Grade {value_grade}", style={"fontSize":"12px","fontWeight":"700",
                                                             "color":gc,"marginLeft":"6px"}),
                html.Span(f"  {value_coverage}/7 metrics",
                          style={"fontSize":"10px","color":"#94a3b8","marginLeft":"8px",
                                 "fontFamily":"'DM Mono',monospace"}),
                html.Span(f"  {growth_str}" if growth_str else "",
                          style={"fontSize":"11px","color":"#64748b","marginLeft":"10px",
                                 "fontFamily":"'DM Mono',monospace"}),
            ], className="mb-2"),
            dbc.Row(metric_cols, className="align-items-end"),
        ], style={"background":"#f8fafc","border":"1px solid #e2e8f0",
                  "borderRadius":"6px","padding":"10px 14px","marginBottom":"10px"})

    # ── Conviction panel ──────────────────────────────────────────────
    conviction_panel = None
    if not is_etf and conviction:
        c_color = "#0d9488" if "HIGH" in conv_grade else "#d97706" if "MED" in conv_grade else "#dc2626" if "LOW" in conv_grade else "#94a3b8"
        c_items = []

        # Analyst consensus
        am = conviction.get("analyst_mean")
        ac = conviction.get("analyst_count", 0)
        if am and ac:
            am_label = ("Strong Buy" if am<=1.5 else "Buy" if am<=2.2
                        else "Hold" if am<=3.2 else "Underperform" if am<=4 else "Sell")
            am_color = "#0d9488" if am<=2.2 else "#d97706" if am<=3.2 else "#dc2626"
            c_items.append(html.Div([
                html.Span("Analyst  ", style={"fontSize":"9px","color":"#94a3b8","fontWeight":"600","textTransform":"uppercase"}),
                html.Span(f"{am_label} ({am:.1f})", style={"fontSize":"13px","fontWeight":"700","color":am_color,"fontFamily":"'DM Mono',monospace"}),
                html.Span(f"  {ac} analysts", style={"fontSize":"10px","color":"#94a3b8"}),
            ], className="me-4"))

        # Short interest
        si = conviction.get("short_pct")
        if si is not None:
            si_color = "#dc2626" if si>25 else "#d97706" if si>12 else "#64748b"
            c_items.append(html.Div([
                html.Span("Short Interest  ", style={"fontSize":"9px","color":"#94a3b8","fontWeight":"600","textTransform":"uppercase"}),
                html.Span(f"{si:.1f}%", style={"fontSize":"13px","fontWeight":"700","color":si_color,"fontFamily":"'DM Mono',monospace"}),
            ], className="me-4"))

        # EPS beats
        beats = conviction.get("earnings_beat_count")
        avg_beat = conviction.get("earnings_beat_avg_pct")
        if beats is not None:
            b_color = "#0d9488" if beats>=3 else "#d97706" if beats==2 else "#dc2626"
            c_items.append(html.Div([
                html.Span("EPS Beats  ", style={"fontSize":"9px","color":"#94a3b8","fontWeight":"600","textTransform":"uppercase"}),
                html.Span(f"{beats}/4 qtrs", style={"fontSize":"13px","fontWeight":"700","color":b_color,"fontFamily":"'DM Mono',monospace"}),
                html.Span(f"  avg {avg_beat:+.1f}%" if avg_beat else "",
                          style={"fontSize":"10px","color":"#94a3b8"}),
            ], className="me-4"))

        # Insider buying
        ins_buy = conviction.get("insider_buying")
        if ins_buy is not None:
            ins_color = "#0d9488" if ins_buy else "#dc2626"
            ins_label = "Buying" if ins_buy else "Selling"
            c_items.append(html.Div([
                html.Span("Insiders  ", style={"fontSize":"9px","color":"#94a3b8","fontWeight":"600","textTransform":"uppercase"}),
                html.Span(ins_label, style={"fontSize":"13px","fontWeight":"700","color":ins_color,"fontFamily":"'DM Mono',monospace"}),
            ], className="me-4"))

        if c_items:
            conviction_panel = html.Div([
                html.Div([
                    html.Span("🎯 Conviction  ", style={"fontSize":"10px","fontWeight":"700",
                                                         "color":c_color,"letterSpacing":"0.06em",
                                                         "textTransform":"uppercase"}),
                    html.Span(conv_grade, style={"fontSize":"14px","fontWeight":"800","color":c_color,
                                                  "fontFamily":"'DM Mono',monospace","marginRight":"10px"}),
                    html.Span(" · ".join(conv_labels) if conv_labels else "",
                              style={"fontSize":"10px","color":"#94a3b8"}),
                ], className="mb-2"),
                html.Div(c_items, style={"display":"flex","flexWrap":"wrap","alignItems":"flex-end"}),
            ], style={"background":"#fafafa","border":"1px solid #e2e8f0",
                      "borderRadius":"6px","padding":"10px 14px","marginBottom":"10px"})

    return html.Div([
        html.Hr(style={"borderColor":"#333"}),
        metrics,
        flag_banner_el,
        links,
        meta_strip,
        conviction_panel,
        value_panel,
        fund_row,
        fmp_section,
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

# ═══════════════════════════════════════════════════════════════════
# CALLBACKS — Value Screener
# ═══════════════════════════════════════════════════════════════════

SP500_TOP50 = [
    "AAPL","MSFT","NVDA","GOOGL","AMZN","META","BRK-B","LLY","AVGO","TSLA",
    "WMT","JPM","V","UNH","XOM","ORCL","MA","HD","PG","COST","JNJ","ABBV",
    "BAC","NFLX","KO","CRM","CVX","MRK","AMD","PEP","ADBE","TMO","ACN","LIN",
    "MCD","ABT","CSCO","TXN","DIS","DHR","NEE","INTC","PM","IBM","RTX","INTU",
    "NOW","QCOM","GE","HON",
]

@app.callback(
    Output("vs-tickers","value"),
    Input("vs-preset-btn","n_clicks"),
    prevent_initial_call=True,
)
def load_sp500_preset(n):
    return ", ".join(SP500_TOP50)


@app.callback(
    Output("vs-results","children"),
    Output("vs-status","children"),
    Input("vs-btn","n_clicks"),
    State("vs-tickers","value"),
    State("vs-min-score","value"),
    State("vs-tech-filter","value"),
    prevent_initial_call=True,
)
def run_value_screen(n_clicks, tickers_raw, min_score, tech_filter):
    if not n_clicks or not tickers_raw:
        return "", ""

    tickers = [t.strip().upper() for t in tickers_raw.replace("\n"," ").split(",")
               if t.strip()]
    tickers = list(dict.fromkeys(tickers))[:100]

    # ── Data source: FMP if key set, yfinance fallback ──────────────
    # Even with a key, FMP free tier blocks key-metrics-ttm (403).
    # Test with a sample ticker — if coverage=0, fall back to yfinance.
    timed_out  = []
    fmp_works  = False

    if _get_fmp_key():
        test_sample = fetch_fmp_value_batch(tickers[:2], max_workers=2)
        test_scores = [compute_value_score(v)[3]   # coverage count
                       for v in test_sample.values()]
        fmp_works = any(c > 0 for c in test_scores)

    if fmp_works:
        # Fetch remaining tickers (first 2 already done)
        rest = [t for t in tickers if t not in test_sample]
        if rest:
            rest_batch = fetch_fmp_value_batch(rest, max_workers=10)
            fund_batch = {**test_sample, **rest_batch}
        else:
            fund_batch = test_sample
        data_src_label = "FMP"
    else:
        fund_batch, timed_out = fetch_yf_fundamentals_batch(
            tickers, max_workers=8, timeout=5)
        data_src_label = "yfinance" + (" (FMP key set but plan restricts fundamentals)" if _get_fmp_key() else "")

    # Tech signal from scanner cache
    tech_map = {}
    if not signals_df.empty and "ticker" in signals_df.columns:
        for _, row in signals_df.iterrows():
            tech_map[str(row.get("ticker","")).upper()] = str(row.get("action","WAIT"))

    def _fmt(v, pct=False, mult=1):
        if v is None: return "—"
        try:
            f = float(v) * mult
            return f"{f:.1f}%" if pct else f"{f:.2f}"
        except: return "—"

    # ── First pass: value score + filters ────────────────────────────
    scored = []
    for t in tickers:
        fund = fund_batch.get(t, {})
        score, grade, bdown, coverage = compute_value_score(fund)
        if score < (min_score or 0):
            continue
        tech_signal = tech_map.get(t, "—")
        if tech_filter == "BUY"       and tech_signal != "BUY":               continue
        if tech_filter == "BUY_WATCH" and tech_signal not in ("BUY","WATCH"): continue
        scored.append((t, fund, score, grade, bdown, coverage, tech_signal))

    if not scored:
        return dbc.Alert(
            "No tickers passed the filters. Try lowering Min Value Score or "
            "relaxing the tech filter.", color="warning"), ""

    # ── Second pass: conviction layer (only for passing tickers) ─────
    passing    = [t for t,*_ in scored]
    conv_batch = fetch_conviction_batch(passing, max_workers=6, timeout=5)

    # ── Build rows ────────────────────────────────────────────────────
    rows = []
    for t, fund, score, grade, bdown, coverage, tech_signal in scored:
        conv    = conv_batch.get(t, {})
        cgrade  = conv.get("conviction_grade", "—")
        clabels = conv.get("conviction_labels", [])
        am      = conv.get("analyst_mean")
        ac      = conv.get("analyst_count", 0)
        mcap    = fund.get("fmp_mcap")
        mcap_str = (f"${mcap/1e9:.1f}B" if mcap and mcap>=1e9
                    else f"${mcap/1e6:.0f}M" if mcap else "—")
        rows.append({
            "Ticker":     t,
            "Score":      score,
            "Grade":      grade,
            "Coverage":   f"{coverage}/7",
            "Conviction": cgrade,
            "Tech":       tech_signal,
            "PE":         _fmt(fund.get("fmp_pe_ttm") or fund.get("fmp_pe")),
            "PEG":        _fmt(fund.get("fmp_peg")),
            "P/B":        _fmt(fund.get("fmp_pb")),
            "FCF Yield":  _fmt(fund.get("fmp_fcf_yield"), pct=True, mult=100),
            "ROE":        _fmt(fund.get("fmp_roe"),        pct=True, mult=100),
            "D/E":        _fmt(fund.get("fmp_debt_eq")),
            "Rev Growth": _fmt(fund.get("fmp_rev_growth"), pct=True, mult=100),
            "MCap":       mcap_str,
            "Analyst":    f"{am:.1f} ({ac})" if am and ac else "—",
            "Short%":     _fmt(conv.get("short_pct"), pct=True, mult=1),
            "EPS Beats":  (f"{conv.get('earnings_beat_count')}/4"
                           if conv.get("earnings_beat_count") is not None else "—"),
            "_labels":    " · ".join(clabels),
        })

    df_out = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    df_out.insert(0, "Rank", range(1, len(df_out)+1))
    display_cols = [c for c in df_out.columns if not c.startswith("_")]

    table = dash_table.DataTable(
        id="vs-table",
        data=df_out.to_dict("records"),
        columns=[{"name":c,"id":c} for c in display_cols],
        sort_action="native",
        filter_action="native",
        page_size=25,
        style_table={"overflowX":"auto"},
        style_cell={
            "background":"#ffffff","color":"#0f172a",
            "border":"1px solid #e2e8f0","fontSize":"12px",
            "padding":"8px 12px","fontFamily":"'DM Mono',monospace",
            "textAlign":"center",
        },
        style_header={
            "background":"#f1f5f9","color":"#475569","fontWeight":"700",
            "border":"1px solid #e2e8f0","fontSize":"11px",
            "textTransform":"uppercase","letterSpacing":"0.05em",
        },
        style_data_conditional=[
            {"if":{"filter_query":'{Grade} = "A"',"column_id":"Grade"},  "color":"#0d9488","fontWeight":"700"},
            {"if":{"filter_query":'{Grade} = "B"',"column_id":"Grade"},  "color":"#0284c7","fontWeight":"700"},
            {"if":{"filter_query":'{Grade} = "C"',"column_id":"Grade"},  "color":"#d97706","fontWeight":"600"},
            {"if":{"filter_query":'{Grade} = "D"',"column_id":"Grade"},  "color":"#dc2626","fontWeight":"600"},
            {"if":{"filter_query":'{Tech} = "BUY"',"column_id":"Tech"},  "color":"#00c853","fontWeight":"700"},
            {"if":{"filter_query":'{Tech} = "WATCH"',"column_id":"Tech"},"color":"#0284c7","fontWeight":"600"},
            {"if":{"filter_query":'{Tech} = "SELL"',"column_id":"Tech"}, "color":"#dc2626","fontWeight":"700"},
            {"if":{"filter_query":"{Score} >= 75","column_id":"Score"},
             "background":"#f0fdf4","color":"#0d9488","fontWeight":"700"},
            {"if":{"filter_query":"{Score} >= 55 && {Score} < 75","column_id":"Score"},
             "background":"#eff6ff","color":"#0284c7","fontWeight":"600"},
            {"if":{"row_index":"odd"},"background":"#f8fafc"},
            {"if":{"filter_query":'{Coverage} = "1/7" || {Coverage} = "2/7"',"column_id":"Coverage"},
             "color":"#dc2626","fontSize":"11px"},
            {"if":{"filter_query":'{Coverage} = "3/7"',"column_id":"Coverage"},
             "color":"#d97706","fontSize":"11px"},
            {"if":{"filter_query":'{Coverage} = "4/7"',"column_id":"Coverage"},
             "color":"#0284c7","fontSize":"11px"},
            {"if":{"filter_query":'{Coverage} = "5/7" || {Coverage} = "6/7" || {Coverage} = "7/7"',"column_id":"Coverage"},
             "color":"#0d9488","fontSize":"11px"},
        ],
        style_cell_conditional=[
            {"if":{"column_id":"Ticker"},     "fontWeight":"700","textAlign":"left"},
            {"if":{"column_id":"Rank"},       "color":"#94a3b8","width":"40px"},
            {"if":{"column_id":"Coverage"},   "fontSize":"11px"},
            {"if":{"column_id":"Conviction"}, "fontSize":"12px"},
        ],
        tooltip_data=[{
            "Score": {
                "value": "\n".join([f"{k}: {v}/100" for k,v in
                    (compute_value_score(fund_batch.get(r["Ticker"],{}))[2] or {}).items()
                    if v is not None]),
                "type":"markdown"},
            "Conviction": {
                "value": r.get("_labels","—") or "No signals",
                "type":"markdown"},
        } for r in df_out.to_dict("records")],
        tooltip_duration=None,
    )

    n_a  = sum(1 for r in rows if r["Grade"]=="A")
    n_b  = sum(1 for r in rows if r["Grade"]=="B")
    n_hi = sum(1 for r in rows if "HIGH" in str(r.get("Conviction","")))
    cached_count = sum(1 for k in _cache if str(k).startswith(("fmp_","yfund_","conv_")))
    to_note = f"  ·  ⏱ {len(timed_out)} timed out" if timed_out else ""

    status_out = dbc.Alert([
        html.Strong(f"✅ {len(rows)} tickers via {data_src_label}  "),
        html.Span(
            f"Grade A: {n_a}  ·  Grade B: {n_b}  ·  🟢 High conviction: {n_hi}"
            f"  ·  Hover Score/Conviction for details{to_note}",
            style={"fontSize":"12px","color":"#64748b"}),
        html.Span(f"  ·  💾 {cached_count} cached (24h)",
                  style={"fontSize":"11px","color":"#94a3b8",
                         "fontFamily":"'DM Mono',monospace"}),
    ], color="success", className="py-2")

    return html.Div([table]), status_out

# CALLBACK — Value Screen candidate loader
# ───────────────────────────────────────────────────────────────────

@app.callback(
    Output("vs-tickers", "value"),
    Output("vs-candidate-info", "children"),
    Input("vs-candidates-btn", "n_clicks"),
    Input("vs-preset-btn", "n_clicks"),
    prevent_initial_call=True,
)
def load_vs_candidates(cand_clicks, preset_clicks):
    from dash import ctx
    trigger = ctx.triggered_id if ctx.triggered_id else ""

    if trigger == "vs-preset-btn":
        return ", ".join(SP500_TOP50), f"{len(SP500_TOP50)} tickers loaded"

    # ── Load candidates from signals_df cache ─────────────────────
    if signals_df.empty or "ticker" not in signals_df.columns:
        return "", "⚠️ No scanner cache yet — run a scan first"

    df = signals_df.copy()

    # Stocks only — ETFs have no FMP fundamentals
    if "type" in df.columns:
        df = df[df["type"] == "Stock"]
    else:
        # Fallback: exclude known ETF tickers by checking universe
        if not universe.empty and "type" in universe.columns:
            etf_tickers = set(universe[universe["type"] == "ETF"]["ticker"].str.upper())
            df = df[~df["ticker"].str.upper().isin(etf_tickers)]

    # Filter: BUY or WATCH action
    if "action" in df.columns:
        df = df[df["action"].isin(["BUY", "WATCH"])]

    # Filter: meaningful dip range — not a collapse, not barely moved
    if "dist_ma200" in df.columns:
        dm = pd.to_numeric(df["dist_ma200"], errors="coerce")
        df = df[(dm >= -40) & (dm <= -5)]

    # Filter: RSI in oversold-to-neutral range
    if "rsi" in df.columns:
        rsi = pd.to_numeric(df["rsi"], errors="coerce")
        df = df[(rsi >= 25) & (rsi <= 55)]

    # Filter: exclude penny stocks (price < $1)
    if "price" in df.columns:
        price = pd.to_numeric(df["price"], errors="coerce")
        df = df[price >= 1.0]

    # Sort by score descending, take top 60
    if "score" in df.columns:
        df = df.sort_values("score", ascending=False)
    df = df.head(60)

    if df.empty:
        return "", "No stock candidates found in cache — run Market Scanner first"

    # Use yf_symbol if available (FMP needs clean base ticker)
    if "yf_symbol" in df.columns:
        tickers = df["yf_symbol"].fillna(df["ticker"]).str.split(".").str[0].str.upper()
    else:
        tickers = df["ticker"].str.upper()

    tickers = tickers.drop_duplicates().tolist()

    # Annotate with action counts for info label
    n_buy   = int((df["action"] == "BUY").sum())  if "action" in df.columns else 0
    n_watch = int((df["action"] == "WATCH").sum()) if "action" in df.columns else 0

    info = (f"{len(tickers)} candidates loaded  ·  "
            f"🟢 {n_buy} BUY  👀 {n_watch} WATCH  ·  "
            f"dist MA200 -5% to -40%  ·  RSI 25–55")

    return ", ".join(tickers), info

# ───────────────────────────────────────────────────────────────────
# CALLBACK — FMP key test
# ───────────────────────────────────────────────────────────────────

@app.callback(
    Output("vs-fmp-status", "children"),
    Input("vs-test-fmp-btn", "n_clicks"),
    prevent_initial_call=True,
)
def test_fmp_key(n):
    key = _get_fmp_key()
    if not key:
        return html.Span("❌ FMP_API_KEY not found in environment",
                         style={"color":"#dc2626"})
    # Try a single lightweight call
    try:
        import requests as _req
        r = _req.get(
            "https://financialmodelingprep.com/api/v3/quote/AAPL",
            params={"apikey": key}, timeout=6
        )
        if r.status_code == 200:
            data = r.json()
            if data and isinstance(data, list) and data[0].get("price"):
                price = data[0]["price"]
                return html.Span(
                    f"✅ FMP key works — AAPL ${price:.2f}",
                    style={"color":"#0d9488","fontWeight":"600"})
            else:
                return html.Span(
                    f"⚠️ Key accepted but empty response — check plan limits",
                    style={"color":"#d97706"})
        elif r.status_code == 401:
            return html.Span("❌ Invalid API key (401)",
                             style={"color":"#dc2626"})
        elif r.status_code == 403:
            return html.Span("❌ Key rejected or plan limit reached (403)",
                             style={"color":"#dc2626"})
        else:
            return html.Span(f"⚠️ HTTP {r.status_code}",
                             style={"color":"#d97706"})
    except Exception as e:
        return html.Span(f"❌ Request failed: {str(e)[:60]}",
                         style={"color":"#dc2626"})

# ───────────────────────────────────────────────────────────────────
# CALLBACK — Value Screen: Refresh Live Data button
# Force-refreshes price + technical cache for all tickers in the
# textarea, then re-runs the screen so BUY/WATCH signals are current.
# ───────────────────────────────────────────────────────────────────

@app.callback(
    Output("vs-refresh-status", "children"),
    Output("vs-results",        "children", allow_duplicate=True),
    Output("vs-status",         "children", allow_duplicate=True),
    Input("vs-refresh-btn", "n_clicks"),
    State("vs-tickers",    "value"),
    State("vs-min-score",  "value"),
    State("vs-tech-filter","value"),
    prevent_initial_call=True,
)
def refresh_value_screen(n_clicks, tickers_raw, min_score, tech_filter):
    """Bust tick_/sfx_/yfund_/conv_ caches for every listed ticker,
    re-fetch live data via yfinance/Stooq, then re-run the screen."""
    if not n_clicks or not tickers_raw:
        return "", no_update, no_update

    tickers = [t.strip().upper() for t in tickers_raw.replace("\n"," ").split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))[:100]

    refreshed, failed = [], []
    for t in tickers:
        try:
            result = fetch_ticker_data(t, force_refresh=True)
            if result:
                refreshed.append(t)
            else:
                failed.append(t)
        except Exception as exc:
            print(f"[refresh] {t} error: {exc}", flush=True)
            failed.append(t)

    # Also bust yfund_ and conv_ keys so Value Score is fresh
    with _cache_lock:
        for t in tickers:
            base = t.split(".")[0]
            for k in list(_cache.keys()):
                if k.startswith(f"yfund_{t}") or k.startswith(f"conv_{t}") \
                        or (k.startswith("fmp_") and f"/{base}" in k):
                    _cache.pop(k, None)

    status_msg = html.Div([
        dbc.Alert([
            html.Strong(f"🔄 Refreshed {len(refreshed)}/{len(tickers)} tickers live  "),
            html.Span(
                (f"· ✅ {', '.join(refreshed[:8])}{'…' if len(refreshed)>8 else ''}  " if refreshed else "") +
                (f"· ❌ no data: {', '.join(failed[:5])}" if failed else ""),
                style={"fontSize":"12px","color":"#64748b"}
            ),
        ], color="info", className="py-2 mb-1"),
    ])

    # Re-run the full screen with fresh data
    results, screen_status = run_value_screen(1, tickers_raw, min_score, tech_filter)
    return status_msg, results, screen_status
