import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
import warnings

warnings.filterwarnings("ignore")

# --- PAGE CONFIG ---
st.set_page_config(page_title="Gramdev AI Dashboard", layout="wide")

# --- 1. MAPPING DICTIONARY ---
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
    "BSE": "BSE", "Cams": "CAMS", "3B": "3BBLACKBIO"
}

def normalize_ticker(name):
    if name in YAHOO_MAP: return YAHOO_MAP[name]
    for key, value in YAHOO_MAP.items():
        if key.upper() in name.upper(): return value
    return name

# --- 2. ROBUST DATA FETCHER ---
def fetch_live_data_smart(ticker_base):
    clean = ticker_base.replace(".NS", "").replace(".BO", "")
    attempts = [f"{clean}.NS", f"{clean}.BO", clean]
    
    for sym in attempts:
        try:
            df = yf.download(sym, period="3mo", progress=False)
            if not df.empty:
                df = df.reset_index()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
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

valid_tickers = sorted(scores_df['Ticker'].unique())
selected_ticker = st.sidebar.selectbox("Select Active Stock", valid_tickers)

default_yahoo = YAHOO_MAP.get(selected_ticker, selected_ticker)
manual_ticker = st.sidebar.text_input("Yahoo Symbol (Edit if needed)", default_yahoo)

if st.sidebar.button("Fetch Live Data 🔄"):
    with st.spinner(f"Connecting to {manual_ticker}..."):
        live_data, status = fetch_live_data_smart(manual_ticker)
        if live_data is not None:
            st.session_state['live_data_cache'][selected_ticker] = live_data
            st.sidebar.success(f"✅ {status}")
        else:
            st.sidebar.error(f"❌ {status}")

# ==========================================
# 📄 MAIN APP LOGIC
# ==========================================
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
