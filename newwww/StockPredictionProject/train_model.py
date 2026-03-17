import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------
# Stock Symbol
# -----------------------------
stock = "AAPL"

# -----------------------------
# Download Stock Data
# -----------------------------
data = yf.download(stock, start="2015-01-01", end="2024-01-01")

# Fix MultiIndex columns
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = data[['Close']]

# -----------------------------
# Create Artificial Missing Values
# -----------------------------
data.loc[data.index[0:5], 'Close'] = np.nan

# -----------------------------
# Fill Missing Values
# -----------------------------
data['Close'] = data['Close'].fillna(data['Close'].mean())

# -----------------------------
# Scaling
# -----------------------------
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data[['Close']])

# -----------------------------
# Create Time Sequences
# -----------------------------
X = []
y = []

for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

# -----------------------------
# Train Test Split
# -----------------------------
split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# -----------------------------
# Build LSTM Model
# -----------------------------
model = Sequential()

model.add(LSTM(50, input_shape=(X_train.shape[1],1)))
model.add(Dense(1))

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# -----------------------------
# Train Model
# -----------------------------
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32
)

# -----------------------------
# Save Model
# -----------------------------
model.save("final_model.h5")

print("✅ Model training complete and saved!")