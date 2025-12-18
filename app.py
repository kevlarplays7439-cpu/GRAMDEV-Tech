import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.optimize import minimize

# --- PAGE CONFIG ---
st.set_page_config(page_title="Wide Moat AI Dashboard", layout="wide")

# --- 1. LOAD DATA ---
@st.cache_data
def load_data():
    scores = pd.read_csv("scores.csv")
    fund = pd.read_csv("fundamentals.csv")
    price = pd.read_csv("price_data.csv")

    fund['Date'] = pd.to_datetime(fund['Date'])
    price['Date'] = pd.to_datetime(price['Date'])
    
    # Fix Column Names
    if 'NetProfit' in fund.columns: fund.rename(columns={'NetProfit': 'Net profit'}, inplace=True)
    if 'Equity' in fund.columns: fund.rename(columns={'Equity': 'Equity Share Capital'}, inplace=True)

    return scores, fund, price

try:
    scores_df, fund_df, price_df = load_data()
except FileNotFoundError:
    st.error("❌ Data files not found!")
    st.stop()

# --- SIDEBAR ---
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio("Go to", ["📊 Executive Dashboard", "🔮 AI Forecasting (LSTM)", "⚖️ Portfolio Opt."])

# --- PAGE 1: EXECUTIVE DASHBOARD ---
if page == "📊 Executive Dashboard":
    st.title("📊 Wide Moat Executive Summary")
    
    # Select Company
    tickers = scores_df['Ticker'].unique()
    selected_ticker = st.selectbox("Select Company", tickers)
    
    # Get Data
    subset_price = price_df[price_df['Ticker'] == selected_ticker].sort_values('Date')
    subset_fund = fund_df[fund_df['Ticker'] == selected_ticker].sort_values('Date')

    # Metrics
    c1, c2, c3 = st.columns(3)
    score = scores_df[scores_df['Ticker'] == selected_ticker]['Moat_Score'].values[0]
    c1.metric("Moat Score", f"{score}/100")
    c2.metric("Latest Price", f"₹{subset_price.iloc[-1]['Close']:.2f}")
    c3.metric("Sales Growth", "Analyzing...")

    # Chart
    st.subheader("📈 Price History")
    fig = px.line(subset_price, x='Date', y='Close')
    st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: AI FORECASTING (LSTM) ---
elif page == "🔮 AI Forecasting (LSTM)":
    st.title("🔮 Deep Learning Price Forecast")
    st.markdown("Using **TensorFlow & Keras (LSTM)** to predict future trends.")
    
    ticker = st.selectbox("Select Stock", price_df['Ticker'].unique())
    days_lookback = st.slider("Training Lookback Days", 30, 365, 60)
    
    if st.button("Train LSTM Model"):
        with st.spinner("Training Neural Network... (This may take a moment)"):
            # Prepare Data
            data = price_df[price_df['Ticker'] == ticker].sort_values('Date')['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)
            
            x_train, y_train = [], []
            for i in range(days_lookback, len(scaled_data)):
                x_train.append(scaled_data[i-days_lookback:i, 0])
                y_train.append(scaled_data[i, 0])
            
            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            
            # Build LSTM Model (Keras/TensorFlow)
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(units=50))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train
            model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=0)
            
            # Predict Next Day
            last_days = scaled_data[-days_lookback:]
            x_test = np.reshape(last_days, (1, days_lookback, 1))
            predicted_price = scaler.inverse_transform(model.predict(x_test))
            
            st.success(f"🧠 LSTM Prediction for Tomorrow: ₹{predicted_price[0][0]:.2f}")
            st.info("Note: This model was trained on-the-fly. For higher accuracy, increase epochs in code.")

# --- PAGE 3: PORTFOLIO ---
elif page == "⚖️ Portfolio Opt.":
    st.title("⚖️ Portfolio Optimization")
    st.write("Using **SciPy & Markowitz Theory** to find optimal weights.")
    
    selected_tickers = st.multiselect("Select Stocks (Min 3)", price_df['Ticker'].unique())
    
    if len(selected_tickers) >= 3:
        if st.button("Optimize"):
            df_pivot = price_df.pivot(index='Date', columns='Ticker', values='Close')[selected_tickers].dropna()
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
            
            st.subheader("Recommended Allocation")
            alloc = pd.DataFrame({'Stock': selected_tickers, 'Weight': result.x})
            st.bar_chart(alloc.set_index('Stock'))
