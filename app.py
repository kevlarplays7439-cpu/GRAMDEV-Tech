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
    
    fig = px.line(active_df, x='Date', y='Close', title="Price Trend")
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: FORECASTING ---
elif page == "🔮 Forecasting":
    st.title(f"🔮 AI Forecast: {selected_ticker}")
    st.caption(source_msg)
    
    # --- RESTORED: 3 OPTIONS ---
    analysis_type = st.radio("Select Analysis Module", ["LSTM Price Forecast", "GARCH Volatility Risk", "ARIMA Trend"])
    
    if len(active_df) < 60:
        st.error("Insufficient Data for Analysis.")
    else:
        # Date Logic
        last_dt = active_df['Date'].iloc[-1]
        next_dt = last_dt + pd.Timedelta(days=1)
        if next_dt.weekday() == 5: next_dt += pd.Timedelta(days=2)
        elif next_dt.weekday() == 6: next_dt += pd.Timedelta(days=1)
        date_str = next_dt.strftime("%d %b %Y")

        # --- OPTION 1: LSTM ---
        if analysis_type == "LSTM Price Forecast":
            st.subheader("🧠 Deep Learning (LSTM)")
            if st.button("Run LSTM Model"):
                with st.spinner("Training Brain..."):
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
                    st.success(f"🧠 LSTM Prediction for {date_str}: ₹{pred:.2f}")

        # --- OPTION 2: GARCH ---
        elif analysis_type == "GARCH Volatility Risk":
            st.subheader("⚠️ GARCH Volatility Model")
            if st.button("Analyze Risk"):
                returns = active_df['Close'].pct_change().dropna() * 100
                am = arch_model(returns, vol='Garch', p=1, q=1)
                res = am.fit(disp='off')
                st.write(res.summary())
                st.line_chart(res.conditional_volatility)
                st.info("Higher spikes mean higher risk of crash/fluctuation.")

        # --- OPTION 3: ARIMA ---
        elif analysis_type == "ARIMA Trend":
            st.subheader("📈 ARIMA Trend Model")
            if st.button("Run ARIMA"):
                model = ARIMA(active_df['Close'], order=(5,1,0))
                fit = model.fit()
                forecast = fit.forecast(steps=1).iloc[0]
                st.info(f"📈 ARIMA Forecast for {date_str}: ₹{forecast:.2f}")

# --- PAGE 3: PORTFOLIO ---
elif page == "⚖️ Portfolio":
    st.title("⚖️ Portfolio Optimization")
    st.info("Uses CSV data for speed.")
    
    selection = st.multiselect("Select Stocks", valid_tickers, default=valid_tickers[:3])
    
    if len(selection) >= 3:
        if st.button("Optimize"):
            pivot = price_df.pivot(index='Date', columns='Ticker', values='Close')[selection].dropna()
            returns = pivot.pct_change().dropna()
            
            mu = returns.mean() * 252
            cov = returns.cov() * 252
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
            bounds = tuple((0, 1) for _ in range(len(selection)))
            init = [1/len(selection)]*len(selection)
            
            res = minimize(lambda w: -(np.sum(mu*w)/np.sqrt(np.dot(w.T,np.dot(cov,w)))), init, bounds=bounds, constraints=cons)
            
            df_res = pd.DataFrame({'Stock': selection, 'Weight': res.x})
            df_res['Weight'] = df_res['Weight'].apply(lambda x: f"{x*100:.1f}%")
            
            c1, c2 = st.columns(2)
            c1.table(df_res)
            c2.plotly_chart(px.pie(values=res.x, names=selection, title="Allocation"))
