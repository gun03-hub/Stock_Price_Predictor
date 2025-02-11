import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Streamlit App Customization (Set background color)
st.markdown(
    """
    <style>
        body {
            background-color: #0f172a;
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .stApp {
            background: linear-gradient(to right, #1e293b, #334155);
        }
        .title {
            text-align: center;
            color: #facc15;
            font-size: 36px;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown("<h1 class='title'>Stock Price Predictor</h1>", unsafe_allow_html=True)

# Dropdown for stock selection
tickers = ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "META", "NFLX"]
ticker = st.selectbox("🔍 Select a Stock:", tickers, index=0)

# Load historical stock data
df = yf.download(ticker, start="2020-01-01", end=datetime.today().strftime('%Y-%m-%d'))

# Check if data is retrieved
if df.empty:
    st.error("⚠️ No data found for the selected stock. Please try another one!")
else:
    # Reset index to get 'Date' column
    df = df.reset_index()

    # Feature selection
    df["Date"] = pd.to_datetime(df["Date"])  # Ensure date format
    df["Days"] = (df["Date"] - df["Date"].min()).dt.days  # Convert to numerical feature

    # Selecting features and labels
    X = df[["Days"]].values  # Feature
    y = df["Close"].values  # Target

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Display stock price trend
    st.subheader("Stock Price Trend")
    fig, ax = plt.subplots()
    ax.plot(df["Date"], df["Close"], label="Actual Price", color="cyan")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock Price (USD)")
    ax.legend()
    st.pyplot(fig)

    # Future prediction
    future_days = st.slider("📅 Select number of days to predict:", 1, 30, 7)

    # Generate future dates
    future_dates_list = [df["Date"].max() + timedelta(days=i) for i in range(1, future_days + 1)]
    future_dates_series = pd.Series(future_dates_list)  # Ensure it's 1D

    # Predict future prices
    future_X = np.array([(df["Days"].max() + i) for i in range(1, future_days + 1)]).reshape(-1, 1)
    predicted_prices = model.predict(future_X)

    # Create DataFrame
    future_df = pd.DataFrame({"Date": future_dates_series, "Predicted Price": predicted_prices.flatten()})

    # Display predictions
    st.subheader("Future Stock Price Predictions")
    st.write(future_df)

    # Plot predicted prices
    st.subheader("Future Price Trend")
    fig2, ax2 = plt.subplots()
    ax2.plot(df["Date"], df["Close"], label="Actual Price", color="cyan")
    ax2.plot(future_df["Date"], future_df["Predicted Price"], label="Predicted Price", linestyle="dashed", color="red")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Stock Price (USD)")
    ax2.legend()
    st.pyplot(fig2)

    # Footer
    st.markdown(
        "<p style='text-align: center; color: lightgray;'>Developed by Gunjan Arora 🚀</p>",
        unsafe_allow_html=True,
    )
