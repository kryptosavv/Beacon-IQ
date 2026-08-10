import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone, date
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. CONFIGURATION & BRANDING ---
st.set_page_config(
    page_title="Beacon IQ",
    page_icon="🗼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS FOR ROUNDED CORNERS & CLEAN UI ---
st.markdown("""
    <style>
    :root {
        --beacon-primary: #F5EB16;
        --beacon-primary-hover: #d6cd13;
        --beacon-text-on-primary: #000000;
    }
    
    .main-title {font-size: 3em; font-weight: bold; color: var(--beacon-primary);}
    .sub-title {font-size: 1.2em; color: #555;}
    .date-banner {
        background-color: #000000; 
        color: #ffffff;
        padding: 10px; 
        border-radius: 5px; 
        border-left: 5px solid var(--beacon-primary);
        font-weight: bold;
        margin-bottom: 20px;
    }
    .metric-box {
        padding: 10px;
        background-color: #d1ecf1; 
        color: #0c5460; 
        border-radius: 5px;
        margin-bottom: 10px;
        font-weight: bold;
        text-align: center;
        border: 1px solid #bee5eb;
    }
    /* ROUNDED CORNERS FOR PLOTLY TILES */
    .stPlotlyChart {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* SYNCED PRIMARY BUTTON STYLING */
    div.stButton > button[kind="primary"] {
        background-color: var(--beacon-primary) !important;
        border-color: var(--beacon-primary) !important;
        color: var(--beacon-text-on-primary) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: var(--beacon-primary-hover) !important;
        border-color: var(--beacon-primary-hover) !important;
        color: var(--beacon-text-on-primary) !important;
    }
    div.stButton > button[kind="primary"] * {
        color: var(--beacon-text-on-primary) !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. DATA LOADING LAYER ---
@st.cache_data(show_spinner=False, ttl=3600)
def download_data(tickers):
    if not tickers:
        return pd.DataFrame()
    
    # Using ^CRSLDX as the standard Yahoo Finance ticker for NIFTY 500
    fixed_tickers = [t if t.endswith('.NS') or t == "^CRSLDX" else f"{t}.NS" for t in tickers]
    download_list = list(set(fixed_tickers + ["^CRSLDX"]))
    
    try:
        data = yf.download(
            download_list,
            period="2y", 
            group_by='ticker',
            threads=True,
            progress=False
        )
        return data
    except Exception as e:
        st.error(f"Download API failed: {e}")
        return pd.DataFrame()

# --- 3. BREADTH HELPERS & ENGINE ---
def render_regime_tile(title, value, series, threshold, positive=True, suffix=""):
    if positive:
        regime_color = "#00FF88" if value >= threshold else "#FF4B4B"
    else:
        regime_color = "#00FF88" if value <= threshold else "#FF4B4B"

    series_5d = series.tail(5)
    dates_5d = series_5d.index.strftime('%Y-%m-%d').tolist() if hasattr(series_5d.index, 'strftime') else list(range(len(series_5d)))
    
    prev_val = series.iloc[-2] if len(series) >= 2 else value

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.65, 0.35], 
        vertical_spacing=0.15,    
        specs=[[{"type": "indicator"}], [{"type": "xy"}]]
    )
    
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=value,
        number={"suffix": suffix, "font": {"size": 36, "color": regime_color, "family": "Arial Black"}},
        delta={
            'reference': prev_val, 
            'relative': False, 
            'position': "right",
            'valueformat': '.2f',
            'font': {'size': 16}
        },
        title={"text": title.upper(), "font": {"size": 13, "color": "gray", "family": "Arial"}},
        align="left"
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=dates_5d,
        y=series_5d,
        mode='lines+markers',
        line=dict(width=3, color=regime_color),
        marker=dict(size=5, color=regime_color),
        fill='tozeroy',
        fillcolor=f"rgba{tuple(int(regime_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.1,)}",
        hovertemplate='<b>%{x}</b><br>Val: %{y:.2f}<extra></extra>' 
    ), row=2, col=1)

    fig.update_layout(
        height=125, 
        margin=dict(l=20, r=20, t=25, b=15),
        template="plotly_dark",
        paper_bgcolor='#111827', 
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def calculate_market_breadth(raw_data):
    if isinstance(raw_data.columns, pd.MultiIndex):
        stock_data = raw_data.drop(columns=["^CRSLDX"], level=0, errors='ignore')
    else: return []

    try:
        closes = stock_data.xs('Close', level=1, axis=1)
        highs = stock_data.xs('High', level=1, axis=1)
        lows = stock_data.xs('Low', level=1, axis=1)
        
        sma20 = closes.rolling(20).mean()
        sma200 = closes.rolling(200).mean()
        above_20dma = (closes > sma20)
        above_200dma = (closes > sma200)
        
        roll_high_252 = highs.rolling(252).max()
        roll_low_252 = lows.rolling(252).min() 
        is_new_high = (highs >= roll_high_252)
        is_new_low = (lows <= roll_low_252)
        
        daily_diff = closes.diff()
        advances = (daily_diff > 0).sum(axis=1)
        declines = (daily_diff < 0).sum(axis=1)
        net_ad = advances - declines
        ad_line = net_ad.cumsum() 
        
        pivot_20 = highs.rolling(20).max().shift(1)
        is_breakout = (closes > pivot_20)
        
        breakout_10d_ago = is_breakout.shift(10)
        ret_10d = closes.pct_change(10)
        successful_breakout = breakout_10d_ago & (ret_10d > 0)
        
        daily_bo_attempts = breakout_10d_ago.sum(axis=1)
        daily_bo_successes = successful_breakout.sum(axis=1)
        
        rolling_attempts = daily_bo_attempts.rolling(10, min_periods=3).sum()
        rolling_successes = daily_bo_successes.rolling(10, min_periods=3).sum()
        
        rolling_bo_success_series = np.where(
            rolling_attempts > 0,
            (rolling_successes / rolling_attempts) * 100,
            np.nan
        )
        rolling_bo_success_series = pd.Series(
            rolling_bo_success_series,
            index=closes.index
        )

        break_below_20dma = (closes < sma20) & (closes.shift(1) > sma20.shift(1))
        
        # Calculate breadth across all valid dates starting from day 200 (to ensure SMA200 is warm)
        valid_dates = closes.index[200:]
        
        breadth_records = []
        
        for d in valid_dates:
            total_valid_stocks = closes.loc[d].count()
            if total_valid_stocks == 0: continue
            
            idx_loc = closes.index.get_loc(d)

            slope_val = np.nan
            if idx_loc >= 20:
                y = ad_line.iloc[idx_loc-19 : idx_loc+1].values
                x = np.arange(len(y))
                if len(y) == 20:
                    slope = np.polyfit(x, y, 1)[0]
                    denom = abs(ad_line.iloc[idx_loc-20])
                    slope_val = (slope / denom) * 100 if denom > 0 else 0

            if idx_loc >= 20:
                ad_change_20d = ad_line.iloc[idx_loc] - ad_line.iloc[idx_loc - 20]
            else:
                ad_change_20d = np.nan

            nh = is_new_high.loc[d].sum()
            nl = is_new_low.loc[d].sum()
            
            bo_val = rolling_bo_success_series.loc[d]
            bo_val_safe = round(bo_val, 2) if not np.isnan(bo_val) else np.nan

            breadth_records.append({
                "Date": d.date(),
                "% Above 20 DMA": round((above_20dma.loc[d].sum() / total_valid_stocks) * 100, 2),
                "% Above 200 DMA": round((above_200dma.loc[d].sum() / total_valid_stocks) * 100, 2),
                "New Highs": int(nh), "New Lows": int(nl), "Net New Highs": int(nh - nl),
                "AD Line": int(ad_line.loc[d]), "AD Slope 20D": round(slope_val, 2) if not np.isnan(slope_val) else np.nan,
                "AD Change 20D": round(ad_change_20d, 2) if not np.isnan(ad_change_20d) else np.nan,
                "Rolling BO Success 10D": bo_val_safe,
                "% Breaking < 20 DMA": round((break_below_20dma.loc[d].sum() / total_valid_stocks) * 100, 1)
            })
            
        return breadth_records
    except Exception as e:
        return []

# --- 4. SCANNER ORCHESTRATOR ---
def scan_stocks(tickers, progress_bar, status_text):
    status_text.text("🔌 Downloading Data...")
    raw_data = download_data(tickers)
    
    if raw_data.empty:
        st.error("⚠️ Data download failed. Please check your internet or ticker list.")
        return [], [], [], []
    
    status_text.text("📊 Calculating Breadth...")
    breadth_data = calculate_market_breadth(raw_data)

    try:
        if isinstance(raw_data.columns, pd.MultiIndex):
            if "^CRSLDX" in raw_data.columns.levels[0]:
                bench_data = raw_data["^CRSLDX"]['Close']
            else: bench_data = pd.Series()
        else: bench_data = pd.Series()
    except: bench_data = pd.Series()

    if isinstance(raw_data.columns, pd.MultiIndex):
        downloaded_tickers = list(set([col[0] for col in raw_data.columns]))
    else: downloaded_tickers = [] 
        
    if "^CRSLDX" in downloaded_tickers: downloaded_tickers.remove("^CRSLDX")
    
    if not downloaded_tickers:
        st.warning("⚠️ No stock data found. Check if tickers have '.NS' suffix.")
        return [], [], [], breadth_data

    # ==========================================
    # IBD-STYLE RS ENGINE (Cross-Sectional Vectorized)
    # ==========================================
    rs_score_matrix = pd.DataFrame()
    try:
        stock_data = raw_data.drop(columns=["^CRSLDX"], level=0, errors='ignore') if isinstance(raw_data.columns, pd.MultiIndex) else raw_data
        closes = stock_data.xs('Close', level=1, axis=1) if isinstance(raw_data.columns, pd.MultiIndex) else pd.DataFrame(stock_data['Close'])
        
        bench_aligned = bench_data.reindex(closes.index).ffill().replace(0, np.nan)
        
        # Calculate Returns
        ret_3m = (closes / closes.shift(63)) - 1
        ret_6m = (closes / closes.shift(126)) - 1
        ret_9m = (closes / closes.shift(189)) - 1
        ret_12m = (closes / closes.shift(252)) - 1
        
        bench_ret_3m = (bench_aligned / bench_aligned.shift(63)) - 1
        bench_ret_6m = (bench_aligned / bench_aligned.shift(126)) - 1
        bench_ret_9m = (bench_aligned / bench_aligned.shift(189)) - 1
        bench_ret_12m = (bench_aligned / bench_aligned.shift(252)) - 1
        
        # Relative Performance vs Benchmark (Excess Return)
        rel_3m = ret_3m.sub(bench_ret_3m, axis=0)
        rel_6m = ret_6m.sub(bench_ret_6m, axis=0)
        rel_9m = ret_9m.sub(bench_ret_9m, axis=0)
        rel_12m = ret_12m.sub(bench_ret_12m, axis=0)
        
        # Apply composite weighting
        rs_comp = (0.40 * rel_3m) + (0.20 * rel_6m) + (0.20 * rel_9m) + (0.20 * rel_12m)
        
        # Rank cross-sectionally per row (date) and format as 1-99 Score
        rs_score_matrix = (rs_comp.rank(axis=1, pct=True) * 99).apply(np.floor).clip(1, 99)
    except Exception:
        pass

    # Result Arrays
    dryup_results, near_52w_results, breakout_results = [], [], []
    total = len(downloaded_tickers)
    
    for idx, ticker in enumerate(downloaded_tickers):
        progress_bar.progress((idx + 1) / total)
        status_text.text(f"Analyzing {ticker}...")
        
        try:
            df = raw_data[ticker].copy()
            # STRICT REQUIREMENT: No blanket OHLCV ffill(). Must rely on valid data only.
            df = df.dropna(subset=["Open", "High", "Low", "Close"]) 
            if df.empty or len(df) < 260: continue

            # Apply mapped cross-sectional RS Score for this ticker
            if not rs_score_matrix.empty and ticker in rs_score_matrix.columns:
                df['RS_Score'] = rs_score_matrix[ticker]
            else: df['RS_Score'] = 0

            # ==========================================
            # SHARED INDICATOR ENGINE
            # ==========================================
            df['SMA20'] = df['Close'].rolling(20).mean()
            df['SMA50'] = df['Close'].rolling(50).mean()
            df['SMA150'] = df['Close'].rolling(150).mean()
            df['SMA200'] = df['Close'].rolling(200).mean()
            df['SMA200_20D'] = df['SMA200'].shift(20)
            df['Ext_SMA200'] = (df['Close'] - df['SMA200']) / df['SMA200'].replace(0, np.nan)
            
            df['High_52'] = df['High'].rolling(252).max()
            df['Low_52'] = df['Low'].rolling(252).min()
            df['Prev_High_52'] = df['High_52'].shift(1)
            
            df['High_20'] = df['High'].rolling(20).max().shift(1)
            df['High_50'] = df['High'].rolling(50).max().shift(1)
            
            df['Vol_MA50'] = df['Volume'].rolling(50).mean()
            df['Vol_Exp'] = df['Volume'] / df['Vol_MA50'].replace(0, np.nan)

            # Base Metrics
            df['Base_High'] = df['High'].rolling(45).max()
            df['Base_Low'] = df['Low'].rolling(45).min()
            df['Base_Depth'] = ((df['Base_High'] - df['Base_Low']) / df['Base_High'].replace(0, np.nan)) * 100

            # Persistent Volume Dry-Up (requires 3 of 5 days AND current day to be dry)
            df['Volume_Dry'] = df['Volume'] < df['Vol_MA50']
            df['Persistent_Dry_Up'] = (df['Volume_Dry'].rolling(5).sum() >= 3) & df['Volume_Dry']

            # Stage 2 Attribute
            df['Stage2'] = (df['Close'] > df['SMA200']) & (df['SMA50'] > df['SMA200']) & (df['SMA200'] > df['SMA200'].shift(20))
            
            # Distance from 52W High
            df['Dist_From_52W_High'] = ((df['High_52'] - df['Close']) / df['High_52'].replace(0, np.nan)) * 100

            # Breakout Attributes
            df['Breakout_52W'] = df['Close'] > df['Prev_High_52']
            df['Breakout_20D'] = df['Close'] > df['High_20']
            df['Breakout_50D'] = df['Close'] > df['High_50']
            df['Base_Breakout'] = df['Close'] > df['Base_High'].shift(1)

            # ==========================================
            # DATE SLICING & EVENT DETECTION
            # ==========================================
            # Process all fully warmed-up dates for event generation (last ~1 year of 2y data)
            range_df = df.iloc[252:]
            
            if range_df.empty: continue
            
            ltp = df.iloc[-1]['Close']
            clean_ticker = ticker.replace(".NS", "")

            for event_date, row in range_df.iterrows():
                
                ret_pct = ((ltp / row['Close']) - 1) * 100 if row['Close'] > 0 else 0
                base_event = {
                    "Ticker": clean_ticker,
                    "Event Date": event_date.date(),
                    "Event Price": row['Close'],
                    "LTP": ltp,
                    "Return": ret_pct,
                    "RS Score": int(row['RS_Score']) if not np.isnan(row['RS_Score']) else 0,
                    "Vol Ratio": row['Vol_Exp'] if not np.isnan(row['Vol_Exp']) else 0.0
                }

                # 1. SCREEN: DRY-UP
                is_dry_up = (
                    (row['Stage2'] == True) & 
                    (row['Dist_From_52W_High'] <= 25) & 
                    (row['Base_Depth'] <= 25) & 
                    (row['Persistent_Dry_Up'] == True) &
                    (row['RS_Score'] >= 75)
                )
                if is_dry_up:
                    dryup_results.append(base_event.copy())

                # 2. SCREEN: NEAR 52W HIGH
                if row['Dist_From_52W_High'] <= 5:
                    near_52w_results.append(base_event.copy())

                # 3. SCREEN: BREAKOUTS
                stage2_trend = (
                    (row["Close"] > row["SMA200"]) and
                    (row["SMA50"] > row["SMA200"]) and
                    (row["SMA200"] > row["SMA200_20D"])
                )
                not_extended = (row["Ext_SMA200"] > 0.02) and (row["Ext_SMA200"] < 0.20)
                stage2_breakout = (
                    stage2_trend and 
                    (row["Breakout_20D"] or row["Breakout_50D"]) and 
                    not_extended and 
                    (row["Vol_Exp"] >= 1.3)
                )

                breakout_types = []
                if stage2_breakout:
                    breakout_types.append("Stage 2 Breakout")
                if row["Breakout_52W"]:
                    breakout_types.append("52W High Breakout")
                if row["Base_Breakout"]:
                    breakout_types.append("Base Breakout")
                if row["Breakout_50D"]:
                    breakout_types.append("50D Breakout")
                if row["Breakout_20D"]:
                    breakout_types.append("20D Breakout")

                if len(breakout_types) > 0 and (row["RS_Score"] >= 60) and (row["Vol_Exp"] >= 1.3):
                    breakout_event = base_event.copy()
                    breakout_event["Breakout Type"] = " + ".join(breakout_types)
                    breakout_results.append(breakout_event)

        except Exception: 
            continue

    return dryup_results, near_52w_results, breakout_results, breadth_data

# --- 5. UTILS: STYLING & TV WATCHLIST ---
def apply_text_styling(val, mode='standard'):
    if not isinstance(val, (int, float)): return ''
    
    green = 'color: #008000; font-weight: bold;' 
    amber = 'color: #DAA520; font-weight: bold;'
    red = 'color: #FF0000; font-weight: bold;' 
    
    if mode == 'standard':
        if val >= 70: return green
        elif val >= 40: return amber
        else: return red
    elif mode == 'return': 
        if val > 0: return green
        else: return red
    elif mode == 'bo_success':
        if val >= 60: return green
        elif val >= 40: return amber
        else: return red
    return ''

def render_screener_tab(data_list, tab_name, sort_cols=None, asc_flags=None, display_cols=None):
    if not data_list:
        st.info(f"No {tab_name} setups found in selected date range.")
        return

    df = pd.DataFrame(data_list)
    
    if sort_cols and asc_flags:
        df = df.sort_values(by=sort_cols, ascending=asc_flags)

    entries_count = len(df)
    unique_count = df["Ticker"].nunique()
    st.markdown(f"<div class='metric-box'>Total Entries: {entries_count} | Unique Stocks: {unique_count}</div>", unsafe_allow_html=True)
    
    if display_cols is None:
        display_cols = ["Ticker", "Event Date", "Event Price", "LTP", "Return", "RS Score", "Vol Ratio"]
    df_display = df[display_cols]

    styled = df_display.style.format({
        "Event Price": "₹ {:.2f}", "LTP": "₹ {:.2f}", "Return": "{:.2f}%", "Vol Ratio": "{:.2f}x"
    })\
    .map(lambda v: apply_text_styling(v, 'return'), subset=["Return"])\
    .map(lambda v: apply_text_styling(v, 'standard'), subset=["RS Score"])
    
    st.dataframe(styled, use_container_width=True, hide_index=True)

    unique_tickers = df["Ticker"].drop_duplicates().tolist()
    tv_formatted = [f"NSE:{t}" for t in unique_tickers]
    
    if tv_formatted:
        st.markdown("### 📋 TradingView Watchlist")
        batches = [", ".join(tv_formatted[i:i+30]) for i in range(0, len(tv_formatted), 30)]
        for b in batches: 
            st.code(b, language="text")

# --- 6. MAIN UI ---
with st.sidebar:
    st.title("⚙️ Configuration")
    file_path = "universe.txt" 
    tickers = []
    
    if os.path.exists(file_path):
        with open(file_path, "r") as f: 
            tickers = [line.strip() for line in f.readlines() if line.strip()]
    
    if not tickers:
        st.sidebar.warning(f"'{file_path}' not found or empty. Please upload a ticker list.")
        uploaded = st.sidebar.file_uploader("Upload Ticker List", type=["txt"])
        if uploaded: 
            tickers = [line.strip() for line in uploaded.read().decode("utf-8").splitlines() if line.strip()]

    st.divider()
    
    # NEW DISPLAY FILTERING LOGIC
    preset = st.radio(
        "Analysis Date:", 
        ["Today", "Yesterday", "Last 7 Days", "Last 30 Days", "This Year", "Custom"],
        index=0
    )
    
    if preset == "Custom":
        c1, c2 = st.columns(2)
        with c1: custom_start = st.date_input("Start", value=date.today()-timedelta(7))
        with c2: custom_end = st.date_input("End", value=date.today())
    else:
        custom_start, custom_end = None, None
    
    st.divider()
    run_btn = st.button("🧮 Run Beacon Engine", type="primary", use_container_width=True)

utc_now = datetime.now(timezone.utc)
ist_offset = timedelta(hours=5, minutes=30)
ist_time = utc_now + ist_offset
last_refreshed = ist_time.strftime("%Y-%m-%d | %H:%M:%S")

col_header, col_refresh = st.columns([3, 1])

with col_header:
    st.markdown('<div class="main-title">Beacon IQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Swing-Trading Screeners & Market Breadth</div>', unsafe_allow_html=True)

with col_refresh:
    st.markdown(f"""
    <div style="text-align: right; color: gray; font-size: 0.9em; padding-top: 20px;">
        Last Refreshed:<br><b>{last_refreshed}</b>
    </div>
    """, unsafe_allow_html=True)

if 'beacon_results' not in st.session_state:
    st.session_state.beacon_results = None

if tickers:
    should_run = run_btn or (st.session_state.beacon_results is None)
    
    if should_run:
        msg = "Downloading data and calculating indicators... this might take 30 seconds..."
        with st.spinner(msg):
            bar = st.progress(0)
            status = st.empty()
            # Scan all available dates without date filter args
            st.session_state.beacon_results = scan_stocks(tickers, bar, status)
            bar.empty()
            status.empty()

if st.session_state.beacon_results:
    # 1. Fetch unfiltered data from cache
    raw_dryup, raw_near_52w, raw_breakouts, breadth = st.session_state.beacon_results
    
    # 2. Determine Filter Boundaries Post-Generation
    if breadth:
        all_dates = sorted(list(set([b["Date"] for b in breadth])))
        latest_avail = all_dates[-1] if all_dates else date.today()
        prev_avail = all_dates[-2] if len(all_dates) > 1 else latest_avail
    else:
        latest_avail = date.today()
        prev_avail = date.today() - timedelta(days=1)

    if preset == "Today":
        f_start, f_end = latest_avail, latest_avail
    elif preset == "Yesterday":
        f_start, f_end = prev_avail, prev_avail
    elif preset == "Last 7 Days":
        f_start, f_end = latest_avail - timedelta(days=7), latest_avail
    elif preset == "Last 30 Days":
        f_start, f_end = latest_avail - timedelta(days=30), latest_avail
    elif preset == "This Year":
        f_start, f_end = date(latest_avail.year, 1, 1), date(latest_avail.year, 12, 31)
    elif preset == "Custom":
        f_start, f_end = custom_start, custom_end

    st.markdown(f'<div class="date-banner">📅 Period: {f_start} — {f_end}</div>', unsafe_allow_html=True)

    # 3. Apply Filters strictly for presentation display
    def filter_events(events):
        return [e for e in events if f_start <= e["Event Date"] <= f_end]

    f_dryup = filter_events(raw_dryup)
    f_near_52w = filter_events(raw_near_52w)
    f_breakouts = filter_events(raw_breakouts)
    
    if breadth:
        df_breadth = pd.DataFrame(breadth)
        df_breadth["Net New Lows"] = df_breadth["New Lows"] - df_breadth["New Highs"]
        
        # Display the regime tiles using unfiltered latest data to maintain valid moving charts
        last10 = df_breadth.tail(10)
        
        pct20_series = last10.get("% Above 20 DMA", pd.Series(dtype=float))
        pct200_series = last10.get("% Above 200 DMA", pd.Series(dtype=float))
        adslope_series = last10.get("AD Slope 20D", pd.Series(dtype=float))
        bo_series = last10.get("Rolling BO Success 10D", pd.Series(dtype=float))
        break20_series = last10.get("% Breaking < 20 DMA", pd.Series(dtype=float))
        net_highs_series = last10.get("Net New Highs", pd.Series(dtype=float))

        if not pct200_series.empty:
            c1, c2, c3 = st.columns(3)
            with c1: render_regime_tile("% Above 20 DMA", pct20_series.iloc[-1], pct20_series, 50, True, "%")
            with c2: render_regime_tile("% Above 200 DMA", pct200_series.iloc[-1], pct200_series, 50, True, "%")
            with c3: render_regime_tile("% Breaking < 20 DMA", break20_series.iloc[-1], break20_series, 4, False, "%")
            
            c4, c5, c6 = st.columns(3)
            with c4: render_regime_tile("AD Slope 20D", adslope_series.iloc[-1], adslope_series.fillna(0), 0, True, "%")
            with c5: render_regime_tile("10D BO Success", bo_series.iloc[-1], bo_series.fillna(0), 50, True, "%")
            with c6: render_regime_tile("Net New Highs", net_highs_series.iloc[-1], net_highs_series.fillna(0), 0, True, "")

        if not df_breadth.empty:
            latest = df_breadth.iloc[-1]

            cond_expansion = (
                latest["% Above 20 DMA"] > 60 and
                latest["AD Slope 20D"] > 0 and
                latest["Rolling BO Success 10D"] > 50
            )

            cond_chop = (
                latest["% Above 20 DMA"] < 45 and
                latest["AD Slope 20D"] < 0 and
                latest["Rolling BO Success 10D"] < 40
            )

            if cond_expansion:
                regime_label = "🟢 EXPANSION REGIME"
            elif cond_chop:
                regime_label = "🔴 CHOP / RISK-OFF"
            else:
                regime_label = "🟡 TRANSITIONAL"

            st.markdown(f"<div class='metric-box'>{regime_label}</div>", unsafe_allow_html=True)
            
            with st.expander("📊 View Market Internals Data"):
                # Filter breadth display dataframe for the selected date range
                df_breadth_filtered = df_breadth[(df_breadth["Date"] >= f_start) & (df_breadth["Date"] <= f_end)]
                
                cols = ["Date", "New Highs", "New Lows", "Net New Highs", "AD Slope 20D", "AD Change 20D", "Rolling BO Success 10D", "% Above 20 DMA", "% Breaking < 20 DMA", "% Above 200 DMA"]
                
                def color_breadth(val):
                    if isinstance(val, (int, float)):
                        if val > 70: return 'color: #008000; font-weight: bold;'
                        elif val > 40: return 'color: #DAA520; font-weight: bold;'
                        else: return 'color: #FF0000; font-weight: bold;'
                    return ''
                    
                def color_slope(val): 
                    return 'color: #008000; font-weight: bold;' if val > 0 else 'color: #FF0000; font-weight: bold;'

                styled = df_breadth_filtered[cols].style.format({
                    "AD Slope 20D": "{:.2f}%", "AD Change 20D": "{:.2f}", "Rolling BO Success 10D": "{:.1f}%",
                    "% Above 20 DMA": "{:.2f}%", "% Above 200 DMA": "{:.2f}%", "% Breaking < 20 DMA": "{:.2f}%"
                })\
                .map(color_breadth, subset=["% Above 20 DMA", "% Above 200 DMA"])\
                .map(lambda v: apply_text_styling(v, 'bo_success'), subset=["Rolling BO Success 10D"])\
                .map(color_slope, subset=["AD Slope 20D", "Net New Highs", "AD Change 20D"])
                
                st.dataframe(styled, use_container_width=True, hide_index=True)
            st.write("") 

    tab_dry, tab_ath, tab_bo = st.tabs([
        "ꔛ Dry-Up", 
        "🎯 Near 52W High", 
        "🚀 Breakouts"
    ])

    with tab_dry:
        render_screener_tab(
            f_dryup, 
            "Dry-Up", 
            sort_cols=["RS Score"], 
            asc_flags=[False]
        )

    with tab_ath:
        render_screener_tab(
            f_near_52w, 
            "Near 52W High", 
            sort_cols=["RS Score", "Event Date"], 
            asc_flags=[False, False]
        )

    with tab_bo:
        # Pass updated column list for Breakouts to include Breakout Type
        render_screener_tab(
            f_breakouts, 
            "Breakouts", 
            sort_cols=["RS Score", "Event Date"], 
            asc_flags=[False, False],
            display_cols=["Ticker", "Event Date", "Breakout Type", "Event Price", "LTP", "Return", "RS Score", "Vol Ratio"]
        )