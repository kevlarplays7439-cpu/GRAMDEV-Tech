import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Gramdev AI Dashboard", layout="wide")

# --- 1. MAPPING DICTIONARY (Updated for 3B) ---
YAHOO_MAP = {
    "Action": "ACE", "Bharat": "BEL", "Blue_Star": "BLUESTARCO", "Caplin": "CAPLIPOINT",
    "C_D_S_L": "CDSL", "Dr_Lal": "LALPATHLAB", "Dynacons": "DYNPRO", "Dynamic": "DYCL",
    "Frontier": "FRONTIER", "Ganesh": "GANESHHOU", "HDFC": "HDFCAMC",
    "I_R_C_T_C": "IRCTC", "Indiamart": "INDIAMART", "Indo_Tech": "INDOTECH",
    "J_B_Chem": "JBCHEPHARM", "Jai_Balaji": "JAIBALAJI", "Jyoti": "JYOTIRES",
    "KNR": "KNRCON", "Kingfa": "KINGFA", "Kirl": "KIRLPNU", "Macpower": "MACPOWER",
    "Master": "MASTERTR", "Mazagon": "MAZDOCK", "Monarch": "MONARCH", "Newgen": "NEWGEN",
    "Polycab": "POLYCAB", "Prec": "PRECWIRE", "RRP": "RRP", "Radhika": "RADHIKAJWE",
    "Schaeffler": "SCHAEFFLER", "Shakti": "SHAKTIPUMP", "Shanthi": "SHANTIGEAR",
    "Sharda": "SHARDAMOTR", "Shilchar": "SHILCHAR", "Sika": "SIKA", "Solar": "SOLARINDS",
    "Stylam": "STYLAMIND", "Swaraj": "SWARAJENG", "Tanfac": "TANFACIND", "Tata": "TATAELXSI",
    "Timex": "TIMEX", "Voltamp": "VOLTAMP", 
    "BLS": "BLS", "Apar": "APARINDS", "Ashoka": "ASHOKA", "Astrazeneca": "ASTRAZEN", 
    "BSE": "BSE", "Cams": "CAMS", 
    # SPECIFIC FIX FOR 3B BLACKBIO
    "3B": "3BBLACKBIO", "3B_Blackbio": "3BBLACKBIO" 
}

def normalize_ticker(name):
    # Try direct lookup
    if name in YAHOO_MAP: return YAHOO_MAP[name]
    # Try fuzzy lookup
    for key, value in YAHOO_MAP.items():
        if key.upper() in name.upper(): return value
    return name

# --- 2. ROBUST DATA FETCHER ---
def fetch_live_data_smart(ticker_base):
    """
    Tries .NS, .BO, and raw. Returns data + status.
    """
    # Clean base (remove old suffixes if any)
    clean = ticker_base.replace(".NS", "").replace(".BO", "")
    
    # Priority: NSE -> BSE -> Raw
    attempts = [f"{clean}.NS", f"{clean}.BO", clean]
    
    for sym in attempts:
        try:
            df = yf.download(sym, period="3mo", progress=False)
            if not df.empty:
                # Fix Columns
                df = df.reset_index()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                # Standardize
                cols_map = {c: c.capitalize() for c in df.columns}
                df.rename(columns=cols_map, inplace=True)
                
                if 'Date' in df.columns and 'Close' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
                    return df, f"Success ({sym})"
        except:
            continue
            
    return None, f"Failed for {clean}"

# --- 3. LOAD DATA ---
@st.cache_data
def load_csvs():
    try:
        scores = pd.read_csv("scores.csv")
        fund = pd.read_csv("fundamentals.csv")
        price = pd.read_csv("price_data.csv")
        
        scores['Ticker'] = scores['Ticker'].apply(normalize_ticker)
        fund['Ticker'] = fund['Ticker'].apply(normalize_ticker)
        price['Ticker'] = price['Ticker'].apply(normalize_ticker)
        
        fund['Date'] = pd.to_datetime(fund['Date'])
        price['Date'] = pd.to_datetime(price['Date'])
        
        if 'NetProfit' in fund.columns: fund.rename(columns={'NetProfit': 'Net profit'}, inplace=True)
        if 'Equity' in fund.columns: fund.rename(columns={'Equity': 'Equity Share Capital'}, inplace=True)
        return scores, fund, price
    except:
        return None, None, None

scores_df, fund_df, price_df = load_csvs()
if scores_df is None: st.stop()

# --- INIT SESSION ---
if 'live_data_cache' not in st.session_state:
    st.session_state['live_data_cache'] = {}

# ==========================================
# 🚀 SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("⚡ Gramdev Controls")
page = st.sidebar.radio("Navigate", ["📊 Dashboard", "🔮 Forecasting", "⚖️ Portfolio"])

st.sidebar.markdown("---")
st.sidebar.header("🔴 Live Data Manager")

# 1. Global Stock Selector
valid_tickers = sorted(scores_df['Ticker'].unique())
selected_ticker = st.sidebar.selectbox("Select Active Stock", valid_tickers)

# 2. Manual Override (The Fail-Safe)
# If auto-detection fails, user can type "3BBLACKBIO.BO" here manually
default_yahoo = YAHOO_MAP.get(selected_ticker, selected_ticker)
manual_ticker = st.sidebar.text_input("Yahoo Symbol (Edit if failed)", default_yahoo)

# 3. Fetch Button
if st.sidebar.button("Fetch Live Data 🔄"):
    with st.spinner(f"Connecting to {manual_ticker}..."):
        # Use the manual input directly
        live_data, status = fetch_live_data_smart(manual_ticker)
        
        if live_data is not None:
            st.session_state['live_data_cache'][selected_ticker] = live_data
            st.sidebar.success(f"✅ {status}")
        else:
            st.sidebar.error(f"❌ {status} - Try typing symbol manually (e.g. 'RELIANCE.NS')")

# ==========================================
# 📄 MAIN APP LOGIC
# ==========================================

# Prepare Active Data
static_data = price_df[price_df['Ticker'] == selected_ticker].sort_values('Date')

if selected_ticker in st.session_state['live_data_cache']:
    live_part = st.session_state['live_data_cache'][selected_ticker]
    live_part = live_part[['Date', 'Close', 'Open', 'High', 'Low', 'Volume']].copy()
    live_part['Ticker'] = selected_ticker
    
    active_df = pd.concat([static_data, live_part], ignore_index=True)
    active_df = active_df.drop_duplicates(subset=['Date'], keep='last').sort_values('Date')
    source_msg = "🟢 LIVE DATA ACTIVE"
else:
    active_df = static_data
    source_msg = "⚠️ USING OLD CSV DATA"

# --- PAGE 1: DASHBOARD ---
if page == "📊 Dashboard":
    st.title(f"📊 {selected_ticker}")
    st.caption(source_msg)
    
    latest_price = active_df['Close'].iloc[-1]
    last_date = active_df['Date'].iloc[-1].strftime('%d-%b-%Y')
    
    score = scores_df[scores_df['Ticker'] == selected_ticker]['Moat_Score'].values[0]
    
    c1, c2 = st.columns(2)
    c1.metric("Latest Close", f"₹{latest_price:,.2f}", f"Date: {last_date}")
    c2.metric("Moat Score", f"{score}/100")
    
    fig = px.line(active_df, x='Date', y='Close', title="Price Trend")
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: FORECASTING ---
elif page == "🔮 Forecasting":
    st.title(f"🔮 AI Forecast: {selected_ticker}")
    st.caption(source_msg)
    
    if st.button("Run LSTM Model"):
        if len(active_df) < 60:
            st.error("Insufficient Data.")
        else:
            with st.spinner("Training..."):
                data = active_df['Close'].values.reshape(-1, 1)
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled = scaler.fit_transform(data)
                
                X, y = [], []
                lookback = 60
                for i in range(lookback, len(scaled)):
                    X.append(scaled[i-lookback:i, 0])
                    y.append(scaled[i, 0])
                X, y = np.array(X), np.array(y)
                X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                
                model = Sequential()
                model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
                model.add(LSTM(50))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mse')
                model.fit(X, y, epochs=5, batch_size=32, verbose=0)
                
                last_60 = scaled[-lookback:].reshape(1, lookback, 1)
                pred = scaler.inverse_transform(model.predict(last_60))[0][0]
