import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="AI Stock Prediction", layout="wide")

# ---------------------------
# FULL UI STYLING
# ---------------------------
st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background-color: #B298E7;
}

/* MAIN TEXT */
h1, h2, h3, h4, h5, h6, p, span, div {
    color: white !important;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: white;
}
[data-testid="stSidebar"] * {
    color: black !important;
}

/* SIDEBAR INPUT BOXES */
[data-testid="stSidebar"] input {
    background-color: white !important;
    color: black !important;
}

/* METRIC CARDS */
[data-testid="stMetric"] {
    background-color: rgba(255,255,255,0.15);
    padding: 10px;
    border-radius: 10px;
}

/* TABLES */
.stDataFrame {
    background-color: white !important;
    color: black !important;
    border-radius: 10px;
}

/* GRAPHS */
.css-1r6slb0 {
    background-color: white !important;
    padding: 10px;
    border-radius: 10px;
}

/* ALERT BOX */
.stAlert {
    background-color: rgba(255,255,255,0.2);
    color: white;
}

/* SPACING */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# TITLE
# ---------------------------
st.title("📈 AI Stock Price Prediction Dashboard")

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("Dashboard Controls")

stock = st.sidebar.text_input("Enter Stock Symbol", "AAPL")
start = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2024-01-01"))

# ---------------------------
# DATA
# ---------------------------
data = yf.download(stock, start=start, end=end)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = data[['Open','High','Low','Close']]

if data.empty:
    st.error("No data found")
    st.stop()

# ---------------------------
# TABLE
# ---------------------------
st.subheader("📊 OHLC Dataset")
st.dataframe(data.tail(20), height=200)

# ---------------------------
# METRICS
# ---------------------------
st.subheader("📊 Dataset Overview")
c1, c2, c3 = st.columns(3)

c1.metric("Total Records", len(data))
c2.metric("Start Price", round(data['Close'].iloc[0],2))
c3.metric("Latest Price", round(data['Close'].iloc[-1],2))

# ---------------------------
# OHLC GRAPH
# ---------------------------
st.subheader("📉 OHLC Chart")
fig1, ax1 = plt.subplots(figsize=(6,3))
ax1.plot(data['Open'], label="Open")
ax1.plot(data['High'], label="High")
ax1.plot(data['Low'], label="Low")
ax1.plot(data['Close'], label="Close")
ax1.legend()
st.pyplot(fig1)

# ---------------------------
# TREND GRAPH
# ---------------------------
st.subheader("📈 Closing Trend")
fig2, ax2 = plt.subplots(figsize=(6,3))
ax2.plot(data['Close'], color='blue')
st.pyplot(fig2)

# ---------------------------
# PREPROCESS
# ---------------------------
data_close = data[['Close']].copy()
data_close.iloc[0:5] = np.nan
data_close = data_close.fillna(data_close.mean())

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_close)

X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    st.error("Not enough data")
    st.stop()

split = int(len(X)*0.8)
X_test = X[split:]
y_test = y[split:]

# ---------------------------
# MODEL
# ---------------------------
model = load_model("final_model.h5")

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y_test.reshape(-1,1))

# ---------------------------
# PERFORMANCE
# ---------------------------
mse = mean_squared_error(actual, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(actual, predictions)

st.subheader("🤖 Performance")
c1,c2,c3 = st.columns(3)
c1.metric("MSE", round(mse,4))
c2.metric("RMSE", round(rmse,4))
c3.metric("R²", round(r2,4))

# ---------------------------
# ACCURACY
# ---------------------------
direction_actual = np.sign(np.diff(actual.flatten()))
direction_pred = np.sign(np.diff(predictions.flatten()))
acc = np.mean(direction_actual == direction_pred)*100

st.subheader("📊 Direction Accuracy")
st.metric("Accuracy", f"{round(acc,2)} %")

# ---------------------------
# ACTUAL VS PREDICTED
# ---------------------------
st.subheader("📈 Actual vs Predicted")
fig3, ax3 = plt.subplots(figsize=(6,3))
ax3.plot(actual, label="Actual")
ax3.plot(predictions, label="Predicted")
ax3.legend()
st.pyplot(fig3)

# ---------------------------
# ERROR
# ---------------------------
st.subheader("📉 Error")
error = actual - predictions
fig4, ax4 = plt.subplots(figsize=(6,3))
ax4.plot(error, color='red')
st.pyplot(fig4)

# ---------------------------
# DISTRIBUTION
# ---------------------------
st.subheader("📊 Distribution")
fig5, ax5 = plt.subplots(figsize=(6,3))
ax5.hist(predictions, bins=40, color='purple')
st.pyplot(fig5)

# ---------------------------
# TABLE
# ---------------------------
st.subheader("📋 Prediction Table")
df = pd.DataFrame({
    "Actual": actual.flatten(),
    "Predicted": predictions.flatten()
})
st.dataframe(df.head(20), height=200)

# ---------------------------
# EXTRA FEATURE
# ---------------------------
st.subheader("📌 Insights")

trend = "UPTREND 📈" if predictions[-1] > actual[-1] else "DOWNTREND 📉"

st.info(f"""
Trend: {trend}  
R² Score: {round(r2,3)}  
Avg Error: {round(np.mean(np.abs(error)),2)}
""")