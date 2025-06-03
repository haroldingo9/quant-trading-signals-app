import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from ta.momentum import RSIIndicator

st.title("📊 Multi-Strategy Quant Trading Signals")

st.sidebar.header("🔧 Choose Your Strategy")
strategy = st.sidebar.selectbox("Strategy", ["Momentum", "Mean Reversion (RSI)", "MA Crossover"])

st.markdown("""
Upload your stock price CSV (with **Date, Open, High, Low, Close, Volume**)  
**OR** enter a ticker symbol to fetch data from Yahoo Finance.
""")

# Sample CSV download button
sample_data = pd.DataFrame({
    'Date': pd.date_range(end=pd.Timestamp.today(), periods=100),
    'Open': np.random.uniform(100, 200, 100).round(2),
    'High': np.random.uniform(200, 250, 100).round(2),
    'Low': np.random.uniform(90, 150, 100).round(2),
    'Close': np.random.uniform(100, 200, 100).round(2),
    'Volume': np.random.randint(1000000, 5000000, 100)
})
sample_csv = sample_data.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Sample CSV", data=sample_csv, file_name='sample_stock_data.csv', mime='text/csv')

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=['csv'])
ticker = st.text_input("🔍 Or enter ticker symbol (e.g., AAPL)", "AAPL")

# Strategy functions
def momentum_strategy(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['Momentum'] = df['Close'].pct_change(5)
    df['Signal'] = np.where((df['Close'] > df['SMA20']) & (df['Momentum'] > 0), 'Buy', 'Hold')
    return df

def mean_reversion_strategy(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    rsi_period = 14
    rsi = RSIIndicator(df['Close'], window=rsi_period)
    df['RSI'] = rsi.rsi()
    df['Signal'] = np.where(df['RSI'] < 30, 'Buy',
                            np.where(df['RSI'] > 70, 'Sell', 'Hold'))
    return df

def ma_crossover_strategy(df):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df['SMA10'] = df['Close'].rolling(window=10).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['Signal'] = 'Hold'
    df.loc[(df['SMA10'] > df['SMA50']) & (df['SMA10'].shift(1) <= df['SMA50'].shift(1)), 'Signal'] = 'Buy'
    df.loc[(df['SMA10'] < df['SMA50']) & (df['SMA10'].shift(1) >= df['SMA50'].shift(1)), 'Signal'] = 'Sell'
    return df

# Load data
df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}.issubset(df.columns):
            st.success("✅ CSV loaded successfully!")
        else:
            st.error("❌ CSV missing required columns!")
            df = None
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        df = None
else:
    if ticker:
        try:
            df = yf.download(ticker, period="6mo", interval="1d").reset_index()
            st.success(f"📈 Data fetched for {ticker}!")
        except Exception as e:
            st.error(f"❌ Error fetching data: {e}")
            df = None

# Apply strategy and show results
if df is not None:
    if strategy == "Momentum":
        df = momentum_strategy(df)
    elif strategy == "Mean Reversion (RSI)":
        df = mean_reversion_strategy(df)
    elif strategy == "MA Crossover":
        df = ma_crossover_strategy(df)

    st.subheader(f"📌 {strategy} Strategy Results")

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price'))

    if strategy == "Momentum":
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA20'], mode='lines', name='SMA 20'))
        buys = df[df['Signal'] == 'Buy']
        fig.add_trace(go.Scatter(x=buys['Date'], y=buys['Close'], mode='markers',
                                 name='Buy Signal', marker=dict(color='green', size=10)))

    elif strategy == "Mean Reversion (RSI)":
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI'))
        buy = df[df['Signal'] == 'Buy']
        sell = df[df['Signal'] == 'Sell']
        fig.add_trace(go.Scatter(x=buy['Date'], y=buy['Close'], mode='markers',
                                 name='Buy Signal', marker=dict(color='green', size=10)))
        fig.add_trace(go.Scatter(x=sell['Date'], y=sell['Close'], mode='markers',
                                 name='Sell Signal', marker=dict(color='red', size=10)))

    elif strategy == "MA Crossover":
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA10'], mode='lines', name='SMA 10'))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['SMA50'], mode='lines', name='SMA 50'))
        buy = df[df['Signal'] == 'Buy']
        sell = df[df['Signal'] == 'Sell']
        fig.add_trace(go.Scatter(x=buy['Date'], y=buy['Close'], mode='markers',
                                 name='Buy Signal', marker=dict(color='green', size=10)))
        fig.add_trace(go.Scatter(x=sell['Date'], y=sell['Close'], mode='markers',
                                 name='Sell Signal', marker=dict(color='red', size=10)))

    st.plotly_chart(fig, use_container_width=True)

    st.write("📋 Last 10 signals:")
    st.dataframe(df[['Date', 'Close', 'Signal']].tail(10))

    # Download signals CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("💾 Download Signals CSV", data=csv, file_name='signals.csv', mime='text/csv')

else:
    st.info("📂 Upload a CSV file or enter a ticker symbol to begin.")