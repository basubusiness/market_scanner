import streamlit as st
st.set_page_config(page_title="Market Decision Engine", layout="wide")

APP_VERSION = "v9.0"

import yfinance as yf
import pandas as pd
import numpy as np
import financedatabase as fd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import justetf_scraping
    JUSTETF_AVAILABLE = True
except ImportError:
    JUSTETF_AVAILABLE = False

# ═══════════════════════════════════════════════════════════
# UNIVERSE LOADERS
# ═══════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading asset universe…")
def load_base_universe():
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
    return df

@st.cache_data(show_spinner="Loading justETF data…", ttl=86400*7)
def load_justetf_universe():
    if not JUSTETF_AVAILABLE:
        return pd.DataFrame()
    try:
        df = justetf_scraping.load_overview()
        df = df.reset_index()
        rename = {
            "ticker": "ticker", "isin": "isin", "name": "jname",
            "domicile_country": "domicile", "ter": "ter",
            "distribution_policy": "dist_policy",
            "fund_size_eur": "fund_size_eur",
            "replication": "replication", "currency": "jcurrency",
        }
        keep = {k: v for k, v in rename.items() if k in df.columns}
        df = df[list(keep.keys())].rename(columns=keep)
        df["ticker"] = df["ticker"].fillna("").astype(str).str.strip().str.upper()
        df = df[df["ticker"].str.match(r"^[A-Z]{1,5}$")]
        df = df.drop_duplicates(subset=["ticker"], keep="first")
        return df
    except Exception:
        return pd.DataFrame()

universe = load_base_universe()
jetf_df  = load_justetf_universe() if JUSTETF_AVAILABLE else pd.DataFrame()

# ═══════════════════════════════════════════════════════════
# TECHNICAL HELPERS
# ═══════════════════════════════════════════════════════════

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
    return float(macd.iloc[-1]), float(signal.iloc[-1]), float((macd-signal).iloc[-1])

def linear_slope(series, window=10):
    y = series.tail(window).values
    if len(y) < window:
        return 0.0
    x = np.arange(len(y))
    slope = np.polyfit(x, y, 1)[0]
    return float(slope / (y.mean() + 1e-9))

# ═══════════════════════════════════════════════════════════
# LIVE DATA HELPERS
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=1800, show_spinner=False)
def get_fg_index():
    import re, datetime
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json, */*"}

    def rating(s):
        if s <= 25: return "Extreme Fear"
        if s <= 44: return "Fear"
        if s <= 55: return "Neutral"
        if s <= 75: return "Greed"
        return "Extreme Greed"

    today = datetime.date.today().strftime("%Y-%m-%d")
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
                    return s, f"{rating(s)} (CNN)"
        except Exception:
            continue
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", headers=headers, timeout=6)
        if r.status_code == 200:
            e = r.json().get("data", [{}])[0]
            s = round(float(e.get("value", 50)))
            return s, f"{e.get('value_classification','').replace('_',' ').title()} (alt.me)"
    except Exception:
        pass
    return None, None

@st.cache_data(ttl=300, show_spinner=False)
def get_live_vix():
    try:
        d = flatten_df(yf.Ticker("^VIX").history(period="5d", auto_adjust=True))
        if not d.empty and "Close" in d.columns:
            return float(d["Close"].dropna().iloc[-1])
    except Exception:
        pass
    return 20.0

# ═══════════════════════════════════════════════════════════
# TICKER ANALYSIS — cached per ticker, 1h TTL
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_data(ticker):
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
        rsi    = float(calculate_rsi(close).iloc[-1])
        if not (1 < rsi < 99):
            return None

        macd_val, macd_sig, macd_hist = calculate_macd(close)
        rsi_slope   = linear_slope(calculate_rsi(close).dropna(), window=5)
        dist_ma     = ((price - ma200) / ma200) * 100
        dist_52h    = ((price - float(close.tail(252).max())) / float(close.tail(252).max())) * 100
        vol_pct     = float(close.pct_change().rolling(20).std().iloc[-1] * 100)
        price_slope = linear_slope(close, window=10)

        conf  = min(abs(dist_ma)/20, 1)*0.5 + min(abs(50-rsi)/50, 1)*0.5
        conf *= 1 - min(vol_pct/5, 0.5)

        return dict(
            price=price, ma50=ma50, ma200=ma200,
            rsi=rsi, rsi_slope=rsi_slope,
            macd=macd_val, macd_signal=macd_sig, macd_hist=macd_hist,
            dist_ma=dist_ma, dist_52h=dist_52h,
            vol=vol_pct, price_slope=price_slope,
            trend_down_strong=(price_slope < -0.003),
            confidence=conf,
        )
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

    knife_thr = -15 * (1 / risk_mult)
    is_knife  = (dm < knife_thr) and (rsi < 35) and raw["trend_down_strong"]
    reversal  = is_knife and macd_bull and rsi_rising

    if is_knife and not reversal:
        action = "AVOID"
    elif dm < -10 and rsi < 40 and macd_bull and macd_accel:
        action = "BUY"
    elif dm < -10 and rsi < 40 and (macd_bull or rsi_rising):
        action = "WATCH"
    elif dm > 10 and rsi > 70:
        action = "SELL"
    else:
        action = "WAIT"

    strength = "Strong" if conf > 0.7 else "Medium" if conf > 0.4 else "Weak"

    # Score for ranking
    dist_s  = min(-dm/30, 1) if dm < 0 else 0
    rsi_s   = max((50-rsi)/50, 0)
    score   = dist_s*0.30 + rsi_s*0.25 + (0.20 if macd_bull else 0) + (0.10 if macd_accel else 0) + conf*0.15
    if action == "AVOID":
        score -= 2

    return {
        "ticker":    ticker,
        "price":     round(raw["price"], 2),
        "ma200":     round(raw["ma200"], 2),
        "dist_pct":  round(dm, 1),
        "dist_52h":  round(raw["dist_52h"], 1),
        "rsi":       round(rsi, 1),
        "rsi_dir":   "↗" if rsi_rising else "↘",
        "macd":      "▲ Bull" if macd_bull else "▼ Bear",
        "macd_acc":  "⚡" if macd_accel else "—",
        "vol_pct":   round(raw["vol"], 2),
        "confidence":round(conf, 2),
        "action":    action,
        "signal":    f"{action} ({strength})",
        "knife":     "⚠️ Knife" if (is_knife and not reversal) else ("✅ Reversal" if reversal else ""),
        "score":     round(score, 4),
    }

def parallel_scan(tickers, risk_mult, max_workers=8):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyse_ticker, t, risk_mult): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                results[t] = fut.result()
            except Exception:
                results[t] = None
    return [results[t] for t in tickers if results.get(t) is not None]

# ═══════════════════════════════════════════════════════════
# NAME + ISIN LOOKUP  (cached per ticker, 24h)
# ═══════════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def get_name_isin(ticker):
    # 1. Try justETF data first (already loaded)
    if not jetf_df.empty and ticker in jetf_df["ticker"].values:
        row  = jetf_df[jetf_df["ticker"] == ticker].iloc[0]
        name = row.get("jname", "")
        isin = row.get("isin", "")
        if name:
            return str(name)[:35], str(isin) if isin else ""
    # 2. yfinance info fallback
    try:
        info = yf.Ticker(ticker).fast_info
        name = getattr(info, "name", "") or ""
        if not name:
            full = yf.Ticker(ticker).info
            name = full.get("longName") or full.get("shortName") or ""
        isin = ""
        return str(name)[:35], isin
    except Exception:
        return ticker, ""

# ═══════════════════════════════════════════════════════════
# UNIVERSE BUILDER — simple presets + optional advanced
# ═══════════════════════════════════════════════════════════

PRESETS = {
    "🌍 All ETFs (global)":              {"type": "ETF"},
    "🇪🇺 UCITS ETFs (EUR domicile)":     {"type": "ETF", "domicile": ["Ireland","Luxembourg"]},
    "🇺🇸 US ETFs":                        {"type": "ETF", "domicile": ["United States"]},
    "📈 US Stocks":                       {"type": "Stock", "country": ["United States"]},
    "🇬🇧 UK Stocks":                      {"type": "Stock", "country": ["United Kingdom"]},
    "🇩🇪 German Stocks":                  {"type": "Stock", "country": ["Germany"]},
    "🌐 Global Stocks (top markets)":    {"type": "Stock", "country": ["United States","United Kingdom","Germany","France","Japan","Canada"]},
    "💰 ETFs — Equities only":            {"type": "ETF", "category_group": ["Equities"]},
    "🏦 ETFs — Fixed Income":             {"type": "ETF", "category_group": ["Fixed Income"]},
    "🥇 ETFs — Commodities":              {"type": "ETF", "category_group": ["Commodities"]},
    "🔧 Custom (advanced)":              {"type": "custom"},
}

def build_ticker_list(preset_key, adv_type, adv_sector, adv_country,
                      adv_domicile, adv_dist, adv_min_size,
                      adv_repl=None, adv_strategy=None,
                      adv_category=None, adv_min_ter=0.0, adv_max_ter=2.0):
    preset = PRESETS[preset_key]
    ptype  = preset.get("type")

    if ptype == "custom":
        atype = adv_type
        parts = []
        if "ETF" in atype:
            e = universe[universe["type"] == "ETF"].copy()
            if adv_category:
                e = e[e["category_group"].isin(adv_category)]
            if not jetf_df.empty:
                jcols = [c for c in ["ticker","domicile","dist_policy","fund_size_eur",
                                      "replication","strategy","ter"] if c in jetf_df.columns]
                e = e.merge(jetf_df[jcols], on="ticker", how="left")
                if adv_domicile and "domicile" in e.columns:
                    e = e[e["domicile"].isin(adv_domicile)]
                if adv_dist and "dist_policy" in e.columns:
                    e = e[e["dist_policy"].isin(adv_dist)]
                if adv_repl and "replication" in e.columns:
                    e = e[e["replication"].isin(adv_repl)]
                if adv_strategy and "strategy" in e.columns:
                    e = e[e["strategy"].isin(adv_strategy)]
                if adv_min_size > 0 and "fund_size_eur" in e.columns:
                    e = e[e["fund_size_eur"].fillna(0) >= adv_min_size]
                if "ter" in e.columns:
                    if adv_min_ter > 0:
                        e = e[e["ter"].fillna(0) >= adv_min_ter]
                    if adv_max_ter < 2.0:
                        e = e[e["ter"].fillna(999) <= adv_max_ter]
            parts.append(e)
        if "Stock" in atype:
            s = universe[universe["type"] == "Stock"].copy()
            if adv_country:
                s = s[s["country"].isin(adv_country)]
            if adv_sector:
                s = s[s["sector"].isin(adv_sector)]
            parts.append(s)
        filtered = pd.concat(parts) if parts else pd.DataFrame(columns=["ticker"])

    elif ptype == "ETF":
        e = universe[universe["type"] == "ETF"].copy()
        if "category_group" in preset:
            e = e[e["category_group"].isin(preset["category_group"])]
        if adv_category:
            e = e[e["category_group"].isin(adv_category)]
        # Merge justETF for domicile/ter/replication filters
        need_jetf = (
            ("domicile" in preset) or adv_domicile or adv_dist or
            adv_repl or adv_strategy or adv_min_size > 0 or
            adv_min_ter > 0 or adv_max_ter < 2.0
        )
        if need_jetf and not jetf_df.empty:
            jcols = [c for c in ["ticker","domicile","dist_policy","fund_size_eur",
                                  "replication","strategy","ter"] if c in jetf_df.columns]
            e = e.merge(jetf_df[jcols], on="ticker", how="left")
            if "domicile" in preset:
                e = e[e["domicile"].isin(preset["domicile"])]
            if adv_domicile:
                e = e[e["domicile"].isin(adv_domicile)]
            if adv_dist and "dist_policy" in e.columns:
                e = e[e["dist_policy"].isin(adv_dist)]
            if adv_repl and "replication" in e.columns:
                e = e[e["replication"].isin(adv_repl)]
            if adv_strategy and "strategy" in e.columns:
                e = e[e["strategy"].isin(adv_strategy)]
            if adv_min_size > 0 and "fund_size_eur" in e.columns:
                e = e[e["fund_size_eur"].fillna(0) >= adv_min_size]
            if "ter" in e.columns:
                if adv_min_ter > 0:
                    e = e[e["ter"].fillna(0) >= adv_min_ter]
                if adv_max_ter < 2.0:
                    e = e[e["ter"].fillna(999) <= adv_max_ter]
        filtered = e

    else:  # Stock
        s = universe[universe["type"] == "Stock"].copy()
        if "country" in preset:
            s = s[s["country"].isin(preset["country"])]
        filtered = s

    tickers = list(dict.fromkeys(
        filtered["ticker"].str.upper().str.strip().tolist()
    ))
    return tickers

# ═══════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════

def signal_colour(val):
    v = str(val)
    if "BUY"   in v: return "color:#00e676;font-weight:bold"
    if "WATCH" in v: return "color:#00bcd4;font-weight:bold"
    if "AVOID" in v: return "color:#ff6d00;font-weight:bold"
    if "SELL"  in v: return "color:#ff1744;font-weight:bold"
    if "WAIT"  in v: return "color:#ffd600"
    return ""

def build_display_df(raw_results, name_map):
    """Convert raw scan results list to clean display DataFrame."""
    rows = []
    for r in raw_results:
        t    = r["ticker"]
        name, isin = name_map.get(t, (t, ""))
        rows.append({
            "Ticker":    t,
            "Name":      name,
            "ISIN":      isin,
            "Price":     r["price"],
            "MA200":     r["ma200"],
            "Dist%":     r["dist_pct"],
            "52W%":      r["dist_52h"],
            "RSI":       r["rsi"],
            "RSI↗":      r["rsi_dir"],
            "MACD":      r["macd"],
            "⚡":         r["macd_acc"],
            "Vol%":      r["vol_pct"],
            "Conf":      r["confidence"],
            "Signal":    r["signal"],
            "Knife":     r["knife"],
            "Action":    r["action"],
            "Score":     r["score"],
            "Yahoo":     f"https://finance.yahoo.com/quote/{t}",
            "justETF":   f"https://www.justetf.com/en/search.html?query={t}",
        })
    df = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", df.index + 1)
    return df

def render_table(df, key_suffix=""):
    if df.empty:
        st.info("No results in this category.")
        return

    DISP = ["Rank","Ticker","Name","ISIN","Price","MA200","Dist%","52W%",
            "RSI","RSI↗","MACD","⚡","Vol%","Conf","Signal","Knife",
            "Yahoo","justETF"]
    DISP = [c for c in DISP if c in df.columns]
    fmt  = {"Price":"{:.2f}","MA200":"{:.2f}",
            "Dist%":"{:+.1f}%","52W%":"{:+.1f}%",
            "RSI":"{:.1f}","Vol%":"{:.2f}%","Conf":"{:.2f}"}
    fmt  = {k:v for k,v in fmt.items() if k in DISP}

    styled = (
        df[DISP].style
        .applymap(signal_colour, subset=["Signal"])
        .format(fmt)
        .background_gradient(subset=["Dist%"], cmap="RdYlGn")
        .background_gradient(subset=["RSI"],   cmap="RdYlGn_r")
        .background_gradient(subset=["Conf"],  cmap="Blues")
    )
    st.dataframe(
        styled,
        use_container_width=True,
        height=min(80 + len(df) * 35, 600),
        column_config={
            "Yahoo":   st.column_config.LinkColumn("Yahoo",   display_text="📈 YF"),
            "justETF": st.column_config.LinkColumn("justETF", display_text="🔍 jETF"),
            "ISIN":    st.column_config.TextColumn("ISIN", width="small"),
            "Name":    st.column_config.TextColumn("Name", width="medium"),
        }
    )

    # ── Row selector → Deep Dive
    tickers = df["Ticker"].tolist()
    sc1, sc2 = st.columns([4, 1])
    with sc1:
        chosen = st.selectbox(
            "🔍 Select ticker to deep dive",
            ["— pick a ticker —"] + tickers,
            key=f"dive_select_{key_suffix}",
        )
    with sc2:
        st.markdown("<br>", unsafe_allow_html=True)
        go = st.button("🔬 Analyse", key=f"dive_go_{key_suffix}", use_container_width=True)
    if go and chosen != "— pick a ticker —":
        st.session_state["dive_ticker"] = chosen
        st.session_state["dive_open"]   = True
        st.rerun()

# ═══════════════════════════════════════════════════════════
# ALLOCATION SIZING
# ═══════════════════════════════════════════════════════════

def suggest_allocation(action, conf, vol_pct, dist_pct, rsi, fg, risk_mult, base):
    if action == "AVOID":
        return "⛔ Skip"
    if action == "WAIT":
        return "—"
    if action == "WATCH":
        amt = base * 0.25 * (0.5 + conf)
        return f"👀 €{amt:,.0f} starter"
    # BUY or SELL
    ctx   = (40 if fg < 35 else 20 if fg < 50 else 0) / 100
    sig   = (min(-dist_pct/20, 1)*0.5 + min((50-rsi)/50, 1)*0.5)
    conf_m= 0.5 + conf
    vol_m = max(0.5, 1 - vol_pct/20)
    amt   = base * (ctx + sig) * conf_m * vol_m * risk_mult
    amt   = max(base*0.25, min(amt, base*3))
    tier  = "🔥" if amt >= base*1.5 else "⚖️" if amt >= base*0.8 else "🔍"
    label = "BUY" if action == "BUY" else "SELL"
    return f"{tier} {label} €{amt:,.0f}"

# ═══════════════════════════════════════════════════════════
# DEEP DIVE
# ═══════════════════════════════════════════════════════════

def run_deep_dive(preloaded_ticker=None):
    st.header("🔬 Deep Dive — Individual Asset")

    col_in, col_base = st.columns([3, 1])
    with col_in:
        default = preloaded_ticker or st.session_state.get("dive_ticker", "")
        user_input = st.text_input(
            "Ticker or ISIN",
            value=default,
            placeholder="e.g. VWRA or IE00B3RBWM25",
            key="dd_input"
        ).strip().upper()
    with col_base:
        dd_base = st.number_input("Budget (EUR)", value=1000, min_value=100, step=100, key="dd_base")

    if not user_input:
        st.info("Enter a ticker or ISIN above, or select a row in the Scanner tab.")
        return

    def is_isin(x):
        return len(x) == 12 and x[:2].isalpha() and x[2:].isalnum()

    ticker = None
    isin   = user_input if is_isin(user_input) else None

    try:
        search = yf.Search(user_input, max_results=15)
        if hasattr(search, "quotes") and search.quotes:
            options = {
                f"{r['symbol']}  —  {r.get('longname', r.get('shortname',''))}  [{r.get('exchDisp','')}]": r["symbol"]
                for r in search.quotes if "symbol" in r
            }
            if options:
                chosen = st.selectbox("Select asset", list(options.keys()), key="dd_sel")
                ticker = options[chosen]
                raw_r  = [r for r in search.quotes if r.get("symbol") == ticker]
                if raw_r:
                    isin = isin or raw_r[0].get("isin")
            else:
                ticker = user_input
        else:
            ticker = user_input
    except Exception:
        ticker = user_input

    if not ticker:
        st.error("Could not resolve ticker.")
        return

    from_scanner = bool(preloaded_ticker and user_input == preloaded_ticker.upper())
    btn_label = f"🔍 Analyse {user_input}" if user_input else "🔍 Analyse"
    if not st.button(btn_label, type="primary", key="dd_run", use_container_width=True):
        if not from_scanner:
            return
        # If loaded from scanner, show the button prominently but don't auto-run
        # (user still needs to click to trigger the API call)
        return

    with st.spinner(f"Fetching {ticker}…"):
        try:
            raw_df = flatten_df(yf.Ticker(ticker).history(period="2y", auto_adjust=True))
        except Exception as e:
            st.error(f"Download failed: {e}")
            return

    if raw_df is None or raw_df.empty or "Close" not in raw_df.columns:
        st.error("No price data found.")
        return

    close = raw_df["Close"].dropna()
    if close.empty or close.nunique() < 5:
        st.warning("⚠️ Insufficient or flat price data.")
        return

    # ── Indicators
    cur_p       = float(close.iloc[-1])
    ma50        = close.rolling(50).mean()
    ma200       = close.rolling(200).mean()
    delta_p     = close.diff()
    gain        = delta_p.where(delta_p > 0, 0.0).rolling(14).mean()
    loss        = (-delta_p.where(delta_p < 0, 0.0)).rolling(14).mean()
    rs          = gain / loss
    rs[loss==0] = np.inf
    rsi_s       = 100 - (100 / (1 + rs))
    rsi_val     = float(rsi_s.iloc[-1])
    rsi_rising  = rsi_val > float(rsi_s.iloc[-2])
    ema12       = close.ewm(span=12, adjust=False).mean()
    ema26       = close.ewm(span=26, adjust=False).mean()
    macd_l      = ema12 - ema26
    macd_sig    = macd_l.ewm(span=9, adjust=False).mean()
    macd_hist   = macd_l - macd_sig
    macd_bull   = float(macd_l.iloc[-1]) > float(macd_sig.iloc[-1])
    macd_accel  = float(macd_hist.iloc[-1]) > 0
    ma200_slope = ((float(ma200.iloc[-1]) - float(ma200.iloc[-20])) / float(ma200.iloc[-20])) * 100
    dist_ma200  = ((cur_p - float(ma200.iloc[-1])) / float(ma200.iloc[-1])) * 100
    volatility  = float(close.pct_change().rolling(20).std().iloc[-1] * 100)
    price_chg   = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)
    thr         = max(1.0, volatility * 1.5)
    week52h     = float(close.tail(252).max())
    dist_52h    = ((cur_p - week52h) / week52h) * 100
    p_slope     = linear_slope(close, window=10)
    is_knife    = (dist_ma200 < -15) and (rsi_val < 35) and (p_slope < -0.003)
    reversal    = is_knife and macd_bull and rsi_rising
    fg_val, fg_lbl = get_fg_index()
    fg_val      = fg_val or 50
    buy_score   = (40 if fg_val<35 else 0)+(30 if rsi_val<40 else 0)+(30 if dist_ma200<0 else 0)
    sell_score  = (40 if fg_val>65 else 0)+(30 if rsi_val>65 else 0)+(30 if dist_ma200>0 else 0)
    if rsi_val < 35 and ma200_slope <= 0:
        entry = "WAIT"
    elif abs(price_chg) > thr and rsi_rising:
        entry = "TRIGGER"
    else:
        entry = "WATCH"
    exit_s = "WAIT" if (rsi_val>65 and ma200_slope>0) else ("TRIGGER" if (-price_chg>thr and not rsi_rising) else "WATCH")

    # ── Lookup name/ISIN from justETF
    jetf_info = {}
    if not jetf_df.empty and ticker in jetf_df["ticker"].values:
        row = jetf_df[jetf_df["ticker"]==ticker].iloc[0]
        jetf_info = row.to_dict()
        isin = isin or str(jetf_info.get("isin",""))

    # ── Header metrics
    st.markdown("---")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Price",     f"{cur_p:.2f}")
    c2.metric("vs MA200",  f"{dist_ma200:+.1f}%",    delta_color="normal" if dist_ma200>0 else "inverse")
    c3.metric("RSI",       f"{rsi_val:.1f}",          delta="↗" if rsi_rising else "↘", delta_color="off")
    c4.metric("MACD",      "▲ Bull" if macd_bull else "▼ Bear")
    c5.metric("52W High",  f"{dist_52h:+.1f}%")
    c6.metric("Vol (20d)", f"{volatility:.1f}%")

    # Links row
    lc1, lc2, lc3 = st.columns(3)
    lc1.markdown(f"[📈 Yahoo Finance](https://finance.yahoo.com/quote/{ticker})")
    lc2.markdown(f"[📊 ETF.com](https://www.etf.com/{ticker})")
    if isin:
        lc3.markdown(f"[🔍 justETF](https://www.justetf.com/en/etf-profile.html?isin={isin})  `{isin}`")
    else:
        lc3.markdown(f"[🔍 justETF search](https://www.justetf.com/en/search.html?query={ticker})")

    # justETF metadata strip
    if jetf_info:
        meta = []
        if jetf_info.get("domicile"): meta.append(f"🏳️ {jetf_info['domicile']}")
        if jetf_info.get("ter"):      meta.append(f"💸 TER {jetf_info['ter']:.2f}%")
        if jetf_info.get("fund_size_eur"): meta.append(f"📦 €{jetf_info['fund_size_eur']:,.0f}m")
        if jetf_info.get("dist_policy"):   meta.append(f"💰 {jetf_info['dist_policy']}")
        if jetf_info.get("replication"):   meta.append(f"🔄 {jetf_info['replication']}")
        if meta:
            st.caption("  ·  ".join(meta))

    # ── Decision
    st.markdown("### 🎯 Decision")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("#### 📥 Buy Signal")
        if is_knife and not reversal:
            st.error("⛔ AVOID — Falling knife. Wait for trend reversal.")
        elif entry == "WAIT":
            st.warning("⏳ WAIT — Still falling. Let it stabilise.")
        elif entry == "WATCH":
            if buy_score >= 70:
                st.info(f"⚖️ PREPARE — Good setup (~€{dd_base:,.0f})")
            else:
                st.info("🔍 WATCH — No strong entry yet.")
        else:
            if buy_score >= 70:
                st.success(f"🔥 BUY — €{dd_base*2:,.0f}")
            elif buy_score >= 40:
                st.success(f"⚖️ BUY — €{dd_base:,.0f}")
            else:
                st.warning(f"⚠️ LIGHT — €{dd_base*0.5:,.0f}")
        if reversal:
            st.success("✅ Reversal confirmed (knife + MACD + RSI turned up)")

    with d2:
        st.markdown("#### 📤 Sell Signal")
        if exit_s == "WAIT":
            st.info("🟡 HOLD — Momentum intact.")
        elif exit_s == "WATCH":
            st.warning("🔵 WATCH — No confirmed downtrend yet.")
        else:
            st.error("🔴 SELL — Downtrend accelerating.")
        if sell_score >= 70:
            st.error("🚨 STRONG SELL — Reduce significantly.")
        elif sell_score >= 40:
            st.warning("⚠️ TRIM — Partial reduction.")
        else:
            st.success("🟢 HOLD — No sell signal.")

    # ── Timing table
    with st.expander("⏱ Full Signal Detail"):
        st.markdown(f"""
| Indicator | Value | Status |
|-----------|-------|--------|
| Price vs MA200 | `{dist_ma200:+.1f}%` | {'🟢 Below (opportunity)' if dist_ma200<0 else '🔴 Above'} |
| RSI | `{rsi_val:.1f}` | {'↗ Rising' if rsi_rising else '↘ Falling'} · {'Oversold' if rsi_val<30 else 'Neutral' if rsi_val<70 else 'Overbought'} |
| MACD | {'▲ Bullish' if macd_bull else '▼ Bearish'} | {'⚡ Accelerating' if macd_accel else 'Decelerating'} |
| MA200 slope | `{ma200_slope:+.2f}%` | {'Uptrend' if ma200_slope>0 else 'Downtrend'} |
| 1-day price change | `{price_chg:+.2f}%` | vs trigger `{thr:.2f}%` |
| 52W High distance | `{dist_52h:+.1f}%` | — |
| Volatility (20d) | `{volatility:.2f}%` | — |
| Falling knife | {'⚠️ YES' if is_knife else '✅ No'} | {'Reversal confirmed ✅' if reversal else ''} |
        """)

    # ── Charts
    st.markdown("### 📊 Price + MAs (2Y)")
    st.line_chart(pd.DataFrame({"Price":close,"MA50":ma50,"MA200":ma200}).dropna(subset=["Price"]))
    c1c, c2c = st.columns(2)
    with c1c:
        st.markdown("**MACD**")
        st.line_chart(pd.DataFrame({"MACD":macd_l,"Signal":macd_sig,"Hist":macd_hist}).tail(120).dropna())
    with c2c:
        st.markdown("**RSI (14d)**")
        st.line_chart(rsi_s.tail(120).to_frame("RSI").dropna())
        st.caption("< 30 oversold · > 70 overbought")

# ═══════════════════════════════════════════════════════════
# SIDEBAR — simplified
# ═══════════════════════════════════════════════════════════

st.sidebar.title(f"📡 Market Scanner {APP_VERSION}")

# Market context
live_vix         = get_live_vix()
fg_auto, fg_lbl  = get_fg_index()
fg_index         = fg_auto or 50
risk_mult        = 1.0 + ((live_vix / 20) * ((100 - fg_index) / 50))

vix_col, fg_col = st.sidebar.columns(2)
vix_col.metric("VIX", f"{live_vix:.1f}")
if fg_auto:
    gauge = "🔴" if fg_auto<=25 else "🟠" if fg_auto<=44 else "🟡" if fg_auto<=55 else "🟢" if fg_auto<=75 else "💚"
    fg_col.metric(f"{gauge} F&G", str(fg_auto))
    st.sidebar.caption(fg_lbl)
else:
    fg_index = st.sidebar.slider("Fear & Greed (manual)", 0, 100, 50)

if   risk_mult > 2.5: st.sidebar.error(f"🔥 Extreme Fear — {risk_mult:.1f}x")
elif risk_mult > 2.0: st.sidebar.error(f"😰 High Fear — {risk_mult:.1f}x")
elif risk_mult < 1.2: st.sidebar.info(f"😌 Calm — {risk_mult:.1f}x")
else:                 st.sidebar.warning(f"⚖️ Normal — {risk_mult:.1f}x")

st.sidebar.divider()

# ── Preset picker
st.sidebar.subheader("🎯 Universe")
preset = st.sidebar.selectbox("Quick preset", list(PRESETS.keys()), index=0)

# ── Determine what asset types the preset implies
_ptype = PRESETS[preset]["type"]
_etfs_in_preset   = _ptype in ("ETF", "custom")
_stocks_in_preset = _ptype in ("Stock", "custom")

# ── Optional filters — always visible, cascade on preset
adv_type     = (["ETF"] if _ptype == "ETF" else
                ["Stock"] if _ptype == "Stock" else ["ETF"])
adv_sector   = []
adv_country  = []
adv_domicile = []
adv_dist     = []
adv_repl     = []
adv_strategy = []
adv_min_size = 0
adv_category = []
adv_min_ter  = 0.0
adv_max_ter  = 2.0

with st.sidebar.expander("🔧 Optional filters", expanded=False):

    if _ptype == "custom":
        adv_type = st.multiselect("Asset type", ["ETF","Stock"], default=["ETF"])
    else:
        st.caption(f"Preset: **{preset}** — asset type locked")

    etfs_active   = "ETF"   in adv_type or _ptype == "ETF"
    stocks_active = "Stock" in adv_type or _ptype == "Stock"

    # ── ETF filters (justETF-powered)
    if etfs_active:
        st.markdown("**📦 ETF Filters**")
        if not jetf_df.empty:
            doms = sorted([d for d in jetf_df["domicile"].dropna().unique() if str(d).strip()])
            adv_domicile = st.multiselect("Domicile", doms,
                help="Ireland/Luxembourg = UCITS · United States = US-domiciled")

            dists = sorted([d for d in jetf_df["dist_policy"].dropna().unique() if str(d).strip()])
            adv_dist = st.multiselect("Distribution policy", dists,
                help="Accumulating (growth) or Distributing (income)")

            if "replication" in jetf_df.columns:
                repls = sorted([r for r in jetf_df["replication"].dropna().unique() if str(r).strip()])
                adv_repl = st.multiselect("Replication method", repls,
                    help="Physical Full · Physical Sampling · Swap-based")

            if "strategy" in jetf_df.columns:
                strats = sorted([s for s in jetf_df["strategy"].dropna().unique() if str(s).strip()])
                adv_strategy = st.multiselect("Strategy", strats,
                    help="Long-only · Short & Leveraged · Active")

            adv_min_size = st.number_input("Min fund size (€m, 0=all)", 0, step=50,
                help="100m+ = reasonable liquidity · 500m+ = large & liquid")

            if "ter" in jetf_df.columns:
                ter_vals = jetf_df["ter"].dropna()
                if not ter_vals.empty:
                    tc1, tc2 = st.columns(2)
                    adv_min_ter = tc1.number_input("Min TER %", 0.0, 5.0, 0.0, step=0.05)
                    adv_max_ter = tc2.number_input("Max TER %", 0.0, 5.0, 2.0, step=0.05)
        else:
            st.info("💡 Add `justetf-scraping` to requirements.txt for domicile, TER, replication filters.")

        # financedatabase ETF category filter (works without justETF)
        if "category_group" in universe.columns:
            cats = sorted([c for c in universe[universe["type"]=="ETF"]["category_group"].unique() if c])
            if cats:
                adv_category = st.multiselect("Asset class (ETF)", cats,
                    help="Equities · Fixed Income · Commodities · Real Estate etc.")

    # ── Stock filters
    if stocks_active:
        st.markdown("**📈 Stock Filters**")
        ctries = sorted([c for c in universe[universe["type"]=="Stock"]["country"].unique() if c])
        adv_country = st.multiselect("Country", ctries,
            help="Filter stocks by listed country")

        sects = sorted([s for s in universe[universe["type"]=="Stock"]["sector"].unique() if s])
        # Cascade: if country selected, narrow sectors
        if adv_country:
            sects = sorted([s for s in universe[
                (universe["type"]=="Stock") & (universe["country"].isin(adv_country))
            ]["sector"].unique() if s])
        adv_sector = st.multiselect("Sector", sects)

st.sidebar.divider()

# ── Scan settings
baseline  = st.sidebar.number_input("💰 Monthly budget (EUR)", value=1000, min_value=100, step=100)
n_workers = st.sidebar.slider("⚡ Parallel workers", 2, 12, 6,
    help="More = faster but may cause data collisions. 6 is the safe default.")

# ── Links
st.sidebar.divider()
with st.sidebar.expander("📖 VIX & Fear & Greed guide"):
    st.markdown("""
**VIX (Volatility Index)**
Measures expected S&P 500 volatility over 30 days.
- < 15 → Complacency / low fear
- 15–25 → Normal market
- 25–30 → Elevated tension
- > 30 → Fear / sell-off
- > 40 → Crisis / extreme fear

[📈 Live VIX on Yahoo](https://finance.yahoo.com/quote/%5EVIX/)

---

**CNN Fear & Greed Index**
Composite of 7 market indicators: momentum, breadth, put/call ratio, junk bond demand, safe haven demand, market volatility, and stock price strength.
- 0–25 → Extreme Fear 🔴 (historically: best entry)
- 25–45 → Fear 🟠
- 45–55 → Neutral 🟡
- 55–75 → Greed 🟢
- 75–100 → Extreme Greed 💚 (historically: caution)

[🧠 CNN Fear & Greed](https://edition.cnn.com/markets/fear-and-greed)

---
*This tool auto-fetches both. If CNN API fails, alternative.me is used as fallback.*
    """)

# ═══════════════════════════════════════════════════════════
# BUILD TICKER LIST
# ═══════════════════════════════════════════════════════════

all_tickers = build_ticker_list(
    preset, adv_type, adv_sector, adv_country,
    adv_domicile, adv_dist, adv_min_size,
    adv_repl=adv_repl, adv_strategy=adv_strategy,
    adv_category=adv_category,
    adv_min_ter=adv_min_ter, adv_max_ter=adv_max_ter,
)

MAX_SCAN = 10_000
if len(all_tickers) > MAX_SCAN:
    all_tickers = all_tickers[:MAX_SCAN]

# Active filter summary
active_filters = []
if adv_domicile:    active_filters.append(f"Domicile: {', '.join(adv_domicile)}")
if adv_dist:        active_filters.append(f"Dist: {', '.join(adv_dist)}")
if adv_repl:        active_filters.append(f"Replication: {', '.join(adv_repl)}")
if adv_strategy:    active_filters.append(f"Strategy: {', '.join(adv_strategy)}")
if adv_category:    active_filters.append(f"Class: {', '.join(adv_category)}")
if adv_country:     active_filters.append(f"Country: {', '.join(adv_country[:3])}")
if adv_sector:      active_filters.append(f"Sector: {', '.join(adv_sector[:2])}")
if adv_min_size:    active_filters.append(f"Size ≥ €{adv_min_size}m")
if adv_min_ter > 0 or adv_max_ter < 2.0:
    active_filters.append(f"TER {adv_min_ter:.2f}–{adv_max_ter:.2f}%")

st.sidebar.caption(f"**{len(all_tickers):,} tickers in scope**")
if active_filters:
    st.sidebar.caption("🔧 " + "  ·  ".join(active_filters))

# ═══════════════════════════════════════════════════════════
# MAIN — TABS
# ═══════════════════════════════════════════════════════════

st.title(f"Market Decision Engine {APP_VERSION}")

# Auto-switch to Deep Dive tab when user clicked Analyse
_default_tab = 1 if st.session_state.get("dive_open") else 0
if st.session_state.get("dive_open"):
    st.session_state.pop("dive_open", None)

tab_scan, tab_dive = st.tabs(["🔭 Market Scanner", "🔬 Deep Dive"])

# ─────────────────────────────────────────────
# SCANNER TAB
# ─────────────────────────────────────────────

with tab_scan:

    if not all_tickers:
        st.warning("No tickers match the selected preset. Try a different one.")
        st.stop()

    # ── VIX regime banner
    if live_vix > 40:
        st.warning("⚡ VIX > 40 — Extreme fear. Signals are noisier. Size conservatively.")

    # ── Scan controls
    col_run, col_clear = st.columns([3, 1])
    with col_run:
        run = st.button("🔄 Run Scan", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear", use_container_width=True):
            for k in ["scan_df","scan_meta"]:
                st.session_state.pop(k, None)
            st.rerun()

    if run:
        n = len(all_tickers)
        with st.spinner(f"Scanning {n:,} tickers ({n_workers} workers)…"):
            raw = parallel_scan(all_tickers, risk_mult, max_workers=n_workers)

        if not raw:
            st.error("No data returned. Check your internet connection.")
        else:
            # Fetch names/ISINs in parallel for valid tickers only
            valid_tickers = [r["ticker"] for r in raw]
            name_map = {}
            with st.spinner(f"Fetching names for {len(valid_tickers)} tickers…"):
                with ThreadPoolExecutor(max_workers=16) as ex:
                    futs = {ex.submit(get_name_isin, t): t for t in valid_tickers}
                    for f in as_completed(futs):
                        t = futs[f]
                        try:
                            name_map[t] = f.result()
                        except Exception:
                            name_map[t] = (t, "")

            df_scan = build_display_df(raw, name_map)

            # Add allocation column
            df_scan["Allocation"] = df_scan.apply(
                lambda r: suggest_allocation(
                    r["Action"], r["Conf"], r["Vol%"],
                    r["Dist%"], r["RSI"], fg_index, risk_mult, baseline
                ), axis=1
            )

            st.session_state["scan_df"]   = df_scan
            st.session_state["scan_meta"] = {
                "attempted": n, "valid": len(df_scan), "preset": preset
            }

    # ── Results
    if "scan_df" in st.session_state:
        df    = st.session_state["scan_df"]
        meta  = st.session_state["scan_meta"]

        st.markdown("---")

        # KPI row
        k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
        k1.metric("Attempted",  meta["attempted"])
        k2.metric("Valid",      meta["valid"])
        k3.metric("🟢 BUY",    int((df["Action"]=="BUY").sum()))
        k4.metric("👀 WATCH",  int((df["Action"]=="WATCH").sum()))
        k5.metric("⛔ AVOID",  int((df["Action"]=="AVOID").sum()))
        k6.metric("🔴 SELL",   int((df["Action"]=="SELL").sum()))
        k7.metric("🟡 WAIT",   int((df["Action"]=="WAIT").sum()))

        # ── Top lists — the clean summary view
        st.subheader("🏆 Top Signals")
        tl1, tl2, tl3 = st.columns(3)

        with tl1:
            st.markdown("#### 🟢 Top BUY")
            buys = df[df["Action"]=="BUY"][["Rank","Ticker","Name","Dist%","RSI","Conf","Signal"]].head(10)
            if buys.empty:
                st.info("No BUY signals.")
            else:
                st.dataframe(buys, use_container_width=True, hide_index=True,
                    column_config={"Conf": st.column_config.ProgressColumn("Conf", min_value=0, max_value=1)})

        with tl2:
            st.markdown("#### 👀 WATCH (emerging)")
            watches = df[df["Action"]=="WATCH"][["Rank","Ticker","Name","Dist%","RSI","Conf","Signal"]].head(10)
            if watches.empty:
                st.info("No WATCH signals.")
            else:
                st.dataframe(watches, use_container_width=True, hide_index=True,
                    column_config={"Conf": st.column_config.ProgressColumn("Conf", min_value=0, max_value=1)})

        with tl3:
            st.markdown("#### 🔴 Top SELL")
            sells = df[df["Action"]=="SELL"][["Rank","Ticker","Name","Dist%","RSI","Conf","Signal"]].head(10)
            if sells.empty:
                st.info("No SELL signals.")
            else:
                st.dataframe(sells, use_container_width=True, hide_index=True,
                    column_config={"Conf": st.column_config.ProgressColumn("Conf", min_value=0, max_value=1)})

        # ── Full table with tabs
        st.markdown("---")
        st.subheader("📋 Full Results")

        DISP_COLS = ["Rank","Ticker","Name","ISIN","Price","MA200","Dist%","52W%",
                     "RSI","RSI↗","MACD","⚡","Vol%","Conf","Signal","Knife","Allocation",
                     "Yahoo","justETF"]

        tabs_map = {
            "All": None, "🟢 BUY": "BUY", "👀 WATCH": "WATCH",
            "⛔ AVOID": "AVOID", "🔴 SELL": "SELL", "🟡 WAIT": "WAIT"
        }
        tab_objs = st.tabs(list(tabs_map.keys()))

        for tab_obj, (label, action_filter) in zip(tab_objs, tabs_map.items()):
            with tab_obj:
                sub = df if action_filter is None else df[df["Action"]==action_filter].reset_index(drop=True)
                # Add Allocation to render_table DISP
                cols = [c for c in DISP_COLS if c in sub.columns]
                render_table(sub[cols] if cols else sub, key_suffix=label)

        # ── Download
        st.markdown("---")
        csv_cols = [c for c in DISP_COLS if c in df.columns and c not in ["Yahoo","justETF"]]
        st.download_button(
            "⬇️ Download CSV",
            data=df[csv_cols].to_csv(index=False).encode("utf-8"),
            file_name="scan_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ─────────────────────────────────────────────
# DEEP DIVE TAB
# ─────────────────────────────────────────────

with tab_dive:
    preloaded = st.session_state.get("dive_ticker", "")
    if preloaded:
        st.info(f"**{preloaded}** loaded from Scanner — click **🔍 Analyse** below to run analysis.")
    run_deep_dive(preloaded_ticker=preloaded)
