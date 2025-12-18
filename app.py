import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.optimize import minimize
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# --- PAGE CONFIG ---
st.set_page_config(page_title="Gramdev AI Dashboard", layout="wide")

# --- 1. EXPANDED MAPPING DICTIONARY (Fixes BLS, Apar, Ashoka, etc.) ---
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
    "Timex": "TIMEX", "Voltamp": "VOLTAMP", 
    # NEW FIXES ADDED HERE:
    "BLS": "BLS", "Apar": "APARINDS", "Ashoka": "ASHOKA", "Astrazeneca": "ASTRAZEN", 
    "BSE": "BSE", "Cams": "CAMS", "3B": "3B_Blackbio"
}

def normalize_ticker(name):
    # 1. Check if name is already a valid value (e.g. "BLS")
    if name in TICKER_MAP.values(): return name
    
    # 2. Check for substring matches from the Map
    name_upper = name.upper()
    for key, value in TICKER_MAP.items():
        if key.upper() in name_upper:
            return value
            
    # 3. Fallback: Return original
    return name

# --- 2. LOAD DATA ---
@st.cache_data
def load_data():
    try:
        scores = pd.read_csv("scores.csv")
        fund = pd.read_csv("fundamentals.csv")
        price = pd.read_csv("price_data.csv")
        
        # Apply normalization to fix names like 'BLS_Internat' -> 'BLS'
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
if scores_df is None: st.stop()

# --- SIDEBAR ---
st.sidebar.title("🚀 Gramdev Analysis")
page = st.sidebar.radio("Go to", ["📊 Executive Dashboard", "🔮 Phase A: AI Forecasting", "⚖️ Phase B: Portfolio Mgmt"])

# --- PAGE 1: DASHBOARD ---
if page == "📊 Executive Dashboard":
    st.title("📊 Executive Summary")
    # Filter tickers to only show ones we have price data for
    valid_tickers = sorted(price_df['Ticker'].unique())
    # Try to find intersection between Scores and Price
    common_tickers = [t for t in scores_df['Ticker'].unique() if t in valid_tickers]
    
    if not common_tickers:
        st.error("No matching tickers found between Scores and Price data. Check TICKER_MAP.")
        st.stop()
        
    ticker = st.selectbox("Select Company", common_tickers)
    
    sub_p = price_df[price_df['Ticker'] == ticker].sort_values('Date')
    sub_f = fund_df[fund_df['Ticker'] == ticker].sort_values('Date')
    
    score_rows = scores_df[scores_df['Ticker'] == ticker]
    score = score_rows['Moat_Score'].values[0] if not score_rows.empty else "N/A"
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Moat Score", f"{score}/100")
    c2.metric("Latest Price", f"₹{sub_p.iloc[-1]['Close']:,.2f}" if not sub_p.empty else "N/A")
    col = 'Sales' if 'Sales' in sub_f.columns else sub_f.columns[2]
    c3.metric("Latest Sales", f"₹{sub_f.iloc[-1][col]:,.2f} Cr" if not sub_f.empty else "N/A")
    
    if not sub_p.empty:
        st.plotly_chart(px.line(sub_p, x='Date', y='Close', title=f"{ticker} Price History"), use_container_width=True)
    else:
        st.warning(f"No price data for {ticker}. Check spelling in TICKER_MAP.")

# --- PAGE 2: FORECASTING (PHASE A) ---
elif page == "🔮 Phase A: AI Forecasting":
    st.title("🔮 Phase A: Advanced Forecasting")
    
    ticker = st.selectbox("Select Stock", sorted(price_df['Ticker'].unique()))
    analysis_type = st.radio("Select Analysis Module", ["LSTM Price Forecast", "GARCH Volatility Risk", "ARIMA Trend"])
    
    subset = price_df[price_df['Ticker'] == ticker].sort_values('Date')
    
    if len(subset) < 60:
        st.error("Insufficient data for AI analysis.")
    else:
        # Date Logic
        last_date = subset['Date'].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        if next_date.weekday() == 5: next_date += pd.Timedelta(days=2)
        elif next_date.weekday() == 6: next_date += pd.Timedelta(days=1)
        date_str = next_date.strftime("%d %b %Y")

        if analysis_type == "LSTM Price Forecast":
            st.subheader("🧠 Deep Learning (LSTM)")
            if st.button("Run Neural Network"):
                with st.spinner("Training Brain..."):
                    data = subset['Close'].values.reshape(-1, 1)
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
                    model.fit(X, y, epochs=1, batch_size=1, verbose=0)
                    
                    last_60 = scaled[-lookback:].reshape(1, lookback, 1)
                    pred = scaler.inverse_transform(model.predict(last_60))[0][0]
                    
                    st.success(f"🤖 LSTM Prediction for {date_str}: ₹{pred:.2f}")

        elif analysis_type == "GARCH Volatility Risk":
            st.subheader("⚠️ GARCH Volatility Model")
            if st.button("Analyze Risk"):
                returns = subset['Close'].pct_change().dropna() * 100
                am = arch_model(returns, vol='Garch', p=1, q=1)
                res = am.fit(disp='off')
                st.write(res.summary())
                vol = res.conditional_volatility.iloc[-1]
                st.metric("Predicted Volatility (Risk)", f"{vol:.2f}%")
                st.line_chart(res.conditional_volatility)

        elif analysis_type == "ARIMA Trend":
            st.subheader("📈 ARIMA Trend Model")
            if st.button("Run ARIMA"):
                model = ARIMA(subset['Close'], order=(5,1,0))
                fit = model.fit()
                forecast = fit.forecast(steps=1).iloc[0]
                st.info(f"ARIMA Forecast for {date_str}: ₹{forecast:.2f}")

# --- PAGE 3: PORTFOLIO (PHASE B) ---
elif page == "⚖️ Phase B: Portfolio Mgmt":
    st.title("⚖️ Phase B: Portfolio Construction")
    
    tickers = sorted(price_df['Ticker'].unique())
    selection = st.multiselect("Select Stocks (Min 5 for Clustering)", tickers, default=tickers[:5])
    
    if len(selection) < 3:
        st.warning("Select at least 3 stocks.")
    else:
        pivot = price_df.pivot(index='Date', columns='Ticker', values='Close')[selection].dropna()
        if pivot.empty:
            st.error("No overlapping data found. Try different stocks.")
        else:
            returns = pivot.pct_change().dropna()
            
            tab1, tab2, tab3 = st.tabs(["Clustering (K-Means)", "PCA Factors", "Optimization"])
            
            with tab1:
                st.subheader("🧬 Stock Clustering")
                st.write("Grouping stocks based on movement similarity.")
                k = st.slider("Number of Clusters", 2, 5, 3)
                
                # Cluster based on correlation
                corr = returns.corr()
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(corr)
                
                cluster_df = pd.DataFrame({'Ticker': selection, 'Cluster': kmeans.labels_})
                st.table(cluster_df.sort_values('Cluster'))
                
            with tab2:
                st.subheader("🧩 PCA Analysis")
                if len(selection) < 3:
                    st.warning("Need at least 3 stocks for PCA.")
                else:
                    pca = PCA(n_components=3)
                    pca.fit(returns)
                    expl = pca.explained_variance_ratio_
                    st.write(f"Factor 1 explains {expl[0]*100:.1f}% of variance (Market Risk).")
                    st.bar_chart(expl)
                
            with tab3:
                st.subheader("🏆 Markowitz Optimization")
                if st.button("Optimize Weights"):
                    mu = returns.mean() * 252
                    cov = returns.cov() * 252
                    
                    def neg_sharpe(w):
                        ret = np.sum(mu * w)
                        vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
                        return -(ret/vol)
                    
                    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                    bounds = tuple((0, 1) for _ in range(len(selection)))
                    init = [1/len(selection)]*len(selection)
                    
                    res = minimize(neg_sharpe, init, bounds=bounds, constraints=cons)
                    
                    res_df = pd.DataFrame({'Stock': selection, 'Weight': res.x})
                    res_df['Weight'] = res_df['Weight'].apply(lambda x: f"{x*100:.1f}%")
                    
                    c1, c2 = st.columns(2)
                    with c1: st.table(res_df)
                    with c2: st.plotly_chart(px.pie(values=res.x, names=selection, title="Optimal Allocation"))
