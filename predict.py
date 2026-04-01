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

# Prepare for Prophet
prophet_df = df.rename(columns={"Date": "ds", "Close": "y"})

# Train/test split: 4 years train, 1 year test
split_date = df["Date"].max() - pd.DateOffset(years=1)
train = prophet_df[prophet_df["ds"] <= split_date]
test = prophet_df[prophet_df["ds"] > split_date]

print(f"Train: {len(train)} rows | Test: {len(test)} rows")

# Fit Prophet
model = Prophet(
    changepoint_prior_scale=0.1,
    seasonality_prior_scale=5.0,
    seasonality_mode="multiplicative",
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_range=0.9,
)
model.add_country_holidays(country_name="US")
model.fit(train)

# Evaluate on test set
test_pred = model.predict(test[["ds"]])
rmse = np.sqrt(mean_squared_error(test["y"].values, test_pred["yhat"].values))
print(f"\nTest RMSE: ${rmse:.2f}")

# Retrain on ALL data for future prediction
print("Retraining on full dataset for future forecast...")
full_model = Prophet(
    changepoint_prior_scale=0.1,
    seasonality_prior_scale=5.0,
    seasonality_mode="multiplicative",
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_range=0.9,
)
full_model.add_country_holidays(country_name="US")
full_model.fit(prophet_df)

# Predict next 1 year (business days)
future = full_model.make_future_dataframe(periods=252, freq="B")
forecast = full_model.predict(future)
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
