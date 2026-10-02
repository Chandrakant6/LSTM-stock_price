# 1. Install required packages (run once in terminal / ! in notebook)
# pip install yfinance pandas numpy matplotlib scikit-learn tensorflow

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime as dt

# ────────────────────────────────────────────────
# Step 1: Download data (example: Apple – change ticker as you like)
# ────────────────────────────────────────────────
ticker = "AAPL"          # or RELIANCE.NS, SBIN.NS, ^NSEI, TSLA, etc.
start_date = "2018-01-01"
end_date   = dt.date.today().strftime("%Y-%m-%d")

df = yf.download(ticker, start=start_date, end=end_date)
df = df[['Close']]       # we use only closing price for simplicity
df.dropna(inplace=True)

print(f"Downloaded {len(df)} days of {ticker} data")

# ────────────────────────────────────────────────
# Step 2: Normalize data (very important for LSTM)
# ────────────────────────────────────────────────
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# ────────────────────────────────────────────────
# Step 3: Create sequences (time steps = look-back window)
# ────────────────────────────────────────────────
TIME_STEPS = 60          # ← most common choice (≈ 3 months of trading days)

X, y = [], []

for i in range(TIME_STEPS, len(scaled_data)):
    X.append(scaled_data[i-TIME_STEPS : i, 0])     # past 60 days
    y.append(scaled_data[i, 0])                    # next day

X, y = np.array(X), np.array(y)

# Reshape for LSTM → [samples, time_steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

print(f"X shape: {X.shape}   y shape: {y.shape}")

# ────────────────────────────────────────────────
# Step 4: Train / Validation split (~80-20)
# ────────────────────────────────────────────────
split = int(0.80 * len(X))
X_train, X_val   = X[:split], X[split:]
y_train, y_val   = y[:split], y[split:]

print(f"Training samples: {len(X_train)}   Validation: {len(X_val)}")

# ────────────────────────────────────────────────
# Step 5: Build LSTM model
# ────────────────────────────────────────────────
model = Sequential()

model.add(LSTM(units=64, return_sequences=True, input_shape=(TIME_STEPS, 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=64, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(units=32))
model.add(Dense(units=1))           # single value output → next close

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

# ────────────────────────────────────────────────
# Step 6: Train
# ────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=30,                # ↑ if you have time / GPU   (10–50 common)
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# ────────────────────────────────────────────────
# Step 7: Plot training loss
# ────────────────────────────────────────────────
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('lstm_training_loss.png', dpi=300, bbox_inches='tight')
plt.show()

# ────────────────────────────────────────────────
# Step 8: Make predictions on validation set
# ────────────────────────────────────────────────
predicted_scaled = model.predict(X_val)

# Inverse transform to real prices
predicted = scaler.inverse_transform(predicted_scaled)
actual    = scaler.inverse_transform(y_val.reshape(-1, 1))

# ────────────────────────────────────────────────
# Step 9: Plot actual vs predicted
# ────────────────────────────────────────────────
plt.figure(figsize=(14,6))
plt.plot(actual,    color='blue',  label='Actual Price')
plt.plot(predicted, color='red',   label='Predicted Price')
plt.title(f'{ticker} Stock Price Prediction – LSTM')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.savefig(f'{ticker}_lstm_prediction.png', dpi=300, bbox_inches='tight')
plt.show()

# Optional: print last few actual vs predicted
print("\nLast 10 actual vs predicted:")
for i in range(-10, 0):
    print(f"Actual: {actual[i][0]:.2f}   Predicted: {predicted[i][0]:.2f}")
