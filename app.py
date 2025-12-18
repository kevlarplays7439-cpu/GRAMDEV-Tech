import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA

# --- PAGE CONFIG ---
st.set_page_config(page_title="Wide Moat AI Dashboard", layout="wide")

# --- 1. MAPPING DICTIONARY ( The Fix for Ticker Mismatch ) ---
TICKER_MAP = {
    "Action": "ACE", "Bharat": "BEL", "Blue_Star": "BLUESTARCO", "Caplin": "CAPLIPOINT",
    "C_D_S_L": "CDSL", "Dr_Lal": "LALPATHLAB", "Dynacons": "DYNPRO", "Dynamic": "DYCL",
    "Frontier": "Frontier_Springs", "Ganesh": "GANESHHOU", "HDFC": "HDFCAMC",
    "I_R_C_T_C": "IRCTC", "Indiamart": "INDIAMART", "Indo_Tech": "INDOTECH",
    "J_B_Chem": "JBCHEPHARM", "Jai_Balaji": "JAIBALAJI", "Jyoti": "JYOTIRES",
    "KNR": "KNRCON", "Kingfa": "KINGFA", "Kirl": "KIRLPNU", "Macpower": "MACPOWER",
    "Master": "MASTERTR", "Mazagon": "MAZDOCK", "Monarch": "MONARCH", "Newgen": "NEWGEN",
    "Polycab": "POLYCAB", "Prec": "PRECWIRE", "RRP": "RRP_Defense", "Radhika": "RADHIKAJWE",
    "Schaeffler": "SCHAEFFLER", "Shakti": "SHAKTIPUMP", "Shanthi": "SHANTIGEAR",
    "Sharda": "SHARDAMOTR", "Shilchar": "SHILCHAR", "Sika": "SIKA", "Solar": "SOLARINDS",
    "Stylam": "STYLAMIND", "Swaraj": "SWARAJENG", "Tanfac": "Tanfac_Inds", "Tata": "TATAELXSI",
    "Timex": "TIMEX", "Voltamp": "VOLTAMP"
}

def normalize_ticker(name):
    if name in TICKER_MAP.values(): return name
    name_upper = name.upper()
    for key, value in TICKER_MAP.items():
        if key.upper() in name_upper:
            return value
    return name

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        scores = pd.read_csv("scores.csv")
        fund = pd.read_csv("fundamentals.csv")
        price = pd.read_csv("price_data.csv")

        # FIX: Apply the name correction
        scores['Ticker'] = scores['Ticker'].apply(normalize_ticker)
        fund['Ticker'] = fund['Ticker'].apply(normalize_ticker)

        fund['Date'] = pd.to_datetime(fund['Date'])
        price['Date'] = pd.to_datetime(price['Date'])
        
        if 'NetProfit' in fund.columns: fund.rename(columns={'NetProfit': 'Net profit'}, inplace=True)
        if 'Equity' in fund.columns: fund.rename(columns={'Equity': 'Equity Share Capital'}, inplace=True)

        return scores, fund, price
    except FileNotFoundError:
        return None, None, None

scores_df, fund_df, price_df = load_data()

if scores_df is None:
    st.error("❌ Critical Error: Data files not found.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio("Go to", ["📊 Executive Dashboard", "🔮 Phase A: AI Forecasting", "⚖️ Phase B: Portfolio Opt."])

# --- PAGE 1: EXECUTIVE DASHBOARD ---
if page == "📊 Executive Dashboard":
    st.title("📊 Wide Moat Executive Summary")
    
    tickers = scores_df['Ticker'].unique()
    selected_ticker = st.selectbox("Select Company", tickers)
    
    subset_price = price_df[price_df['Ticker'] == selected_ticker].sort_values('Date')
    subset_fund = fund_df[fund_df['Ticker'] == selected_ticker].sort_values('Date')

    c1, c2, c3 = st.columns(3)
    
    # Score
    score_rows = scores_df[scores_df['Ticker'] == selected_ticker]
    score = score_rows['Moat_Score'].values[0] if not score_rows.empty else "N/A"
    c1.metric("Moat Score", f"{score}/100" if score != "N/A" else "N/A")
    
    # Price
    if not subset_price.empty:
        c2.metric("Latest Price", f"₹{subset_price.iloc[-1]['Close']:,.2f}")
    else:
        c2.metric("Latest Price", "No Data")

    # Sales
    if not subset_fund.empty:
        col = 'Sales' if 'Sales' in subset_fund.columns else subset_fund.columns[2]
        c3.metric("Latest Sales", f"₹{subset_fund.iloc[-1][col]:,.2f} Cr")
    else:
        c3.metric("Latest Sales", "No Data")

    st.subheader("📈 Price History")
    if not subset_price.empty:
        fig = px.line(subset_price, x='Date', y='Close')
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: PHASE A (FORECASTING) ---
elif page == "🔮 Phase A: AI Forecasting":
    st.title("🔮 Phase A: Deep Learning Forecast")
    
    valid_tickers = sorted(price_df['Ticker'].unique())
    ticker = st.selectbox("Select Stock", valid_tickers)
    
    model_type = st.radio("Select Model", ["LSTM (Deep Learning)", "ARIMA (Statistical)"])
    days_lookback = st.slider("Training Lookback Days", 30, 365, 60)
    
    if st.button("Run Forecast"):
        subset = price_df[price_df['Ticker'] == ticker].sort_values('Date')
        
        if len(subset) < days_lookback + 10:
            st.error("Not enough data to train.")
        else:
            # Calculate Next Date
            last_date = subset['Date'].iloc[-1]
            next_date = last_date + pd.Timedelta(days=1)
            if next_date.weekday() == 5: next_date += pd.Timedelta(days=2)
            elif next_date.weekday() == 6: next_date += pd.Timedelta(days=1)
            formatted_date = next_date.strftime("%d %b %Y")

            if model_type == "LSTM (Deep Learning)":
                with st.spinner("Training Neural Network..."):
                    data = subset['Close'].values.reshape(-1, 1)
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_data = scaler.fit_transform(data)
                    
                    x_train, y_train = [], []
                    for i in range(days_lookback, len(scaled_data)):
                        x_train.append(scaled_data[i-days_lookback:i, 0])
                        y_train.append(scaled_data[i, 0])
                    
                    x_train, y_train = np.array(x_train), np.array(y_train)
                    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                    
                    model = Sequential()
                    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
                    model.add(LSTM(units=50))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=0)
                    
                    last_days = scaled_data[-days_lookback:]
                    x_test = np.reshape(last_days, (1, days_lookback, 1))
                    predicted_price = scaler.inverse_transform(model.predict(x_test))
                    
                    st.success(f"🧠 LSTM Prediction for {formatted_date}: ₹{predicted_price[0][0]:.2f}")
            
            else: # ARIMA
                with st.spinner("Running ARIMA..."):
                    model = ARIMA(subset['Close'], order=(5,1,0))
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=1)
                    st.info(f"📈 ARIMA Forecast for {formatted_date}: ₹{forecast.iloc[0]:.2f}")

# --- PAGE 3: PHASE B (PORTFOLIO) ---
elif page == "⚖️ Phase B: Portfolio Opt.":
    st.title("⚖️ Phase B: Portfolio Optimization")
    
    valid_tickers = sorted(price_df['Ticker'].unique())
    selected_tickers = st.multiselect("Select Stocks (Min 3)", valid_tickers)
    
    if len(selected_tickers) >= 3:
        if st.button("Optimize Portfolio"):
            df_pivot = price_df.pivot(index='Date', columns='Ticker', values='Close')[selected_tickers].dropna()
            
            if df_pivot.empty:
                st.error("No overlapping data found.")
            else:
                returns = df_pivot.pct_change().mean() * 252
                cov = df_pivot.pct_change().cov() * 252
                
                def neg_sharpe(weights):
                    p_ret = np.sum(returns * weights)
                    p_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
                    return -(p_ret / p_vol)
                
                bounds = tuple((0, 1) for _ in range(len(selected_tickers)))
                constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                init_guess = [1/len(selected_tickers)] * len(selected_tickers)
                
                result = minimize(neg_sharpe, init_guess, bounds=bounds, constraints=constraints)
                
                st.subheader("🏆 Optimal Allocation (Max Sharpe)")
                alloc = pd.DataFrame({'Stock': selected_tickers, 'Weight': result.x})
                alloc['Weight'] = alloc['Weight'].apply(lambda x: f"{x*100:.1f}%")
                
                c1, c2 = st.columns(2)
                with c1: st.table(alloc)
                with c2: 
                    fig = px.pie(values=result.x, names=selected_tickers, title="Portfolio Allocation")
                    st.plotly_chart(fig, use_container_width=True)
