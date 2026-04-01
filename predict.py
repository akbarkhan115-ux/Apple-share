import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

print("Downloading Apple stock data (last 5 years)...")
df = yf.download("AAPL", period="5y")
df = df[["Close"]].reset_index()
df.columns = ["Date", "Close"]
df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)

# Prepare for Prophet (log transform for stability)
prophet_df = df.rename(columns={"Date": "ds", "Close": "y"})
prophet_df["y_orig"] = prophet_df["y"].copy()
prophet_df["y"] = np.log(prophet_df["y"])

# Feature engineering
prophet_df["MA_5"] = prophet_df["y"].rolling(window=5, min_periods=1).mean()
prophet_df["MA_10"] = prophet_df["y"].rolling(window=10, min_periods=1).mean()
prophet_df["MA_20"] = prophet_df["y"].rolling(window=20, min_periods=1).mean()
prophet_df["MA_50"] = prophet_df["y"].rolling(window=50, min_periods=1).mean()
prophet_df["EMA_10"] = prophet_df["y"].ewm(span=10, min_periods=1).mean()
prophet_df["EMA_20"] = prophet_df["y"].ewm(span=20, min_periods=1).mean()
prophet_df["Lag_1"] = prophet_df["y"].shift(1)
prophet_df["Lag_2"] = prophet_df["y"].shift(2)
prophet_df["Lag_3"] = prophet_df["y"].shift(3)
prophet_df["Lag_5"] = prophet_df["y"].shift(5)
# RSI (14-day)
delta = prophet_df["y"].diff()
gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
rs = gain / loss.replace(0, 1e-10)
prophet_df["RSI"] = 100 - (100 / (1 + rs))
# Bollinger Band width
bb_ma = prophet_df["y"].rolling(20, min_periods=1).mean()
bb_std = prophet_df["y"].rolling(20, min_periods=1).std().fillna(0)
prophet_df["BB_width"] = bb_std * 2
# Momentum
prophet_df["Momentum_5"] = prophet_df["y"] - prophet_df["y"].shift(5)
prophet_df["Momentum_10"] = prophet_df["y"] - prophet_df["y"].shift(10)
prophet_df = prophet_df.bfill()

regressor_cols = ["MA_5", "MA_10", "MA_20", "MA_50", "EMA_10", "EMA_20",
                  "Lag_1", "Lag_2", "Lag_3", "Lag_5",
                  "RSI", "BB_width", "Momentum_5", "Momentum_10"]

# Train/test split
split_date = df["Date"].max() - pd.DateOffset(years=1)
train = prophet_df[prophet_df["ds"] <= split_date]
test = prophet_df[prophet_df["ds"] > split_date]

print(f"Train: {len(train)} rows | Test: {len(test)} rows")

# Fit Prophet with regressors
model = Prophet(
    changepoint_prior_scale=0.02,
    seasonality_prior_scale=15.0,
    seasonality_mode="multiplicative",
    yearly_seasonality=20,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_range=0.95,
    n_changepoints=50,
)
model.add_country_holidays(country_name="US")
model.add_seasonality(name="monthly", period=30.5, fourier_order=8)
model.add_seasonality(name="quarterly", period=91.25, fourier_order=5)
for col in regressor_cols:
    model.add_regressor(col, mode="multiplicative")
model.fit(train)

# Evaluate on test set (inverse log transform)
test_pred = model.predict(test[["ds"] + regressor_cols])
y_true = test["y_orig"].values
y_pred = np.exp(test_pred["yhat"].values)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = np.mean(np.abs(y_true - y_pred))
print(f"\nTest RMSE: ${rmse:.2f}")
print(f"Test MAE:  ${mae:.2f}")

# Retrain on ALL data for future prediction
print("Retraining on full dataset for future forecast...")
full_model = Prophet(
    changepoint_prior_scale=0.02,
    seasonality_prior_scale=15.0,
    seasonality_mode="multiplicative",
    yearly_seasonality=20,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_range=0.95,
    n_changepoints=50,
)
full_model.add_country_holidays(country_name="US")
full_model.add_seasonality(name="monthly", period=30.5, fourier_order=8)
full_model.add_seasonality(name="quarterly", period=91.25, fourier_order=5)
for col in regressor_cols:
    full_model.add_regressor(col, mode="multiplicative")
full_model.fit(prophet_df)

# Predict next 1 year (business days)
future = full_model.make_future_dataframe(periods=252, freq="B")
future = future.merge(prophet_df[["ds"] + regressor_cols], on="ds", how="left")
for col in regressor_cols:
    future[col] = future[col].ffill()
forecast = full_model.predict(future)
# Inverse log transform
forecast["yhat"] = np.exp(forecast["yhat"])
forecast["yhat_lower"] = np.exp(forecast["yhat_lower"])
forecast["yhat_upper"] = np.exp(forecast["yhat_upper"])
future_only = forecast[forecast["ds"] > df["Date"].max()][["ds", "yhat", "yhat_lower", "yhat_upper"]]
future_only.columns = ["Date", "Predicted", "Lower", "Upper"]

print(f"\n{'='*60}")
print(f" Apple (AAPL) Price Forecast — Next 1 Year")
print(f"{'='*60}")
print(f" Last actual close: ${df['Close'].iloc[-1]:.2f} on {df['Date'].iloc[-1].date()}")
print(f" Forecast period:   {future_only['Date'].iloc[0].date()} → {future_only['Date'].iloc[-1].date()}")
print(f"{'='*60}\n")

# Show monthly summary
future_only = future_only.copy()
future_only["Month"] = future_only["Date"].dt.to_period("M")
monthly = future_only.groupby("Month").agg(
    Avg_Price=("Predicted", "mean"),
    Low=("Lower", "min"),
    High=("Upper", "max")
).round(2)

print("Monthly Forecast Summary:")
print("-" * 50)
print(f"{'Month':<12} {'Avg Price':>10} {'Low':>10} {'High':>10}")
print("-" * 50)
for idx, row in monthly.iterrows():
    print(f"{str(idx):<12} ${row['Avg_Price']:>9.2f} ${row['Low']:>9.2f} ${row['High']:>9.2f}")

print(f"\n1-Year Forecast Range: ${future_only['Lower'].min():.2f} — ${future_only['Upper'].max():.2f}")
print(f"Predicted price in 1 year: ${future_only['Predicted'].iloc[-1]:.2f}")
