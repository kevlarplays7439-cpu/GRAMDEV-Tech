import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Wide Moat Dashboard", layout="wide")

# --- 1. LOAD DATA ---
@st.cache_data
def load_data():
    # Load the 3 files
    scores_df = pd.read_csv("scores.csv")
    fund_df = pd.read_csv("fundamentals.csv")
    price_df = pd.read_csv("price_data.csv")

    # Fix Dates
    fund_df['Date'] = pd.to_datetime(fund_df['Date'])
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    
    # Handle the specific column names from your file
    # If your file has 'NetProfit', we standardise it for the app
    if 'NetProfit' in fund_df.columns:
        fund_df.rename(columns={'NetProfit': 'Net profit'}, inplace=True)
    if 'Equity' in fund_df.columns:
        fund_df.rename(columns={'Equity': 'Equity Share Capital'}, inplace=True)

    return scores_df, fund_df, price_df

try:
    scores, fundamentals, prices = load_data()
except FileNotFoundError:
    st.error("❌ Files not found! Please make sure 'scores.csv', 'fundamentals.csv', and 'price_data.csv' are in the same folder as this script.")
    st.stop()

# --- 2. SIDEBAR (User Input) ---
st.sidebar.title("🔍 Stock Screener")

# Filter: Show only stocks with a specific minimum score?
min_score = st.sidebar.slider("Minimum Moat Score", 0, 100, 50)
filtered_stocks = scores[scores['Moat_Score'] >= min_score]['Ticker'].unique()

# Dropdown to select company
selected_ticker = st.sidebar.selectbox("Select a Company", filtered_stocks)

# Get data for selected company
company_score = scores[scores['Ticker'] == selected_ticker]['Moat_Score'].values[0]
company_fund = fundamentals[fundamentals['Ticker'] == selected_ticker].sort_values('Date')
company_price = prices[prices['Ticker'] == selected_ticker].sort_values('Date')

# --- 3. MAIN DASHBOARD ---
st.title(f"📊 {selected_ticker} Analysis")

# Top Row: Score & Stats
col1, col2, col3 = st.columns(3)
with col1:
    color = "green" if company_score >= 75 else "orange" if company_score >= 50 else "red"
    st.markdown(f"### Moat Score: :{color}[{company_score}/100]")
with col2:
    if not company_price.empty:
        latest_price = company_price.iloc[-1]['Close']
        st.metric("Latest Price", f"₹{latest_price:,.2f}")
with col3:
    if not company_fund.empty:
        latest_sales = company_fund.iloc[-1]['Sales']
        st.metric("Latest Annual Sales", f"₹{latest_sales:,.2f} Cr")

st.markdown("---")

# Row 2: Stock Price Chart
st.subheader("📈 10-Year Price Trend")
if not company_price.empty:
    fig_price = px.line(company_price, x='Date', y='Close', title=f"{selected_ticker} Share Price")
    st.plotly_chart(fig_price, use_container_width=True)
else:
    st.warning("No price data available for this company.")

# Row 3: Fundamental Charts (Sales vs Profit)
st.subheader("🏢 Fundamental Growth")
if not company_fund.empty:
    tab1, tab2 = st.tabs(["Sales vs Profit", "Debt Profile"])
    
    with tab1:
        # Dual Axis Chart for Sales & Profit
        fig_fund = go.Figure()
        fig_fund.add_trace(go.Bar(x=company_fund['Date'], y=company_fund['Sales'], name='Sales', marker_color='blue'))
        fig_fund.add_trace(go.Scatter(x=company_fund['Date'], y=company_fund['Net profit'], name='Net Profit', yaxis='y2', line=dict(color='green', width=3)))
        
        fig_fund.update_layout(
            title="Sales (Bar) vs Net Profit (Line)",
            yaxis=dict(title="Sales (Cr)"),
            yaxis2=dict(title="Net Profit (Cr)", overlaying='y', side='right')
        )
        st.plotly_chart(fig_fund, use_container_width=True)
        
    with tab2:
        # Debt vs Equity
        fig_debt = px.bar(company_fund, x='Date', y=['Borrowings', 'Equity Share Capital', 'Reserves'], 
                          title="Capital Structure (Debt vs Equity)", barmode='stack')
        st.plotly_chart(fig_debt, use_container_width=True)
else:
    st.warning("No fundamental data available.")

# --- 4. SHOW RAW DATA ---
with st.expander("View Raw Data"):
    st.write("Fundamental Data:", company_fund)
