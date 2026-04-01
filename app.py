import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Apple Stock Forecast", layout="wide")
st.title("🍎 Apple (AAPL) Stock Price Forecast — Prophet Model")

# ── 1. Download last 5 years of data ──────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    df = yf.download("AAPL", period="5y")
    df = df[["Close", "Volume"]].copy()
    df.columns = ["Close", "Volume"]
    df = df.reset_index()
    df.columns = ["Date", "Close", "Volume"]
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.dropna()
    return df

with st.spinner("Downloading Apple stock data..."):
    df = load_data()

st.subheader("📊 Raw Data Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(df))
col2.metric("Date Range", f"{df['Date'].min().date()} → {df['Date'].max().date()}")
col3.metric("Latest Close", f"${df['Close'].iloc[-1]:.2f}")

# ── 2. Preprocessing ─────────────────────────────────────────────────────────
st.subheader("⚙️ Preprocessing")

# Handle missing dates by forward-filling
date_range = pd.date_range(start=df["Date"].min(), end=df["Date"].max(), freq="B")
df_full = pd.DataFrame({"Date": date_range})
df_full = df_full.merge(df, on="Date", how="left")
df_full["Close"] = df_full["Close"].ffill()
df_full["Volume"] = df_full["Volume"].ffill()
df_full = df_full.dropna()

# Outlier removal using IQR on daily returns
df_full["Return"] = df_full["Close"].pct_change()
Q1 = df_full["Return"].quantile(0.01)
Q3 = df_full["Return"].quantile(0.99)
mask = df_full["Return"].between(Q1, Q3) | df_full["Return"].isna()
outliers_removed = (~mask).sum()
df_full = df_full[mask].copy()
df_full = df_full.drop(columns=["Return"])

# Feature engineering — extra regressors for Prophet (on log scale)
df_full["Log_Close"] = np.log(df_full["Close"])
df_full["MA_5"] = df_full["Log_Close"].rolling(window=5, min_periods=1).mean()
df_full["MA_10"] = df_full["Log_Close"].rolling(window=10, min_periods=1).mean()
df_full["MA_20"] = df_full["Log_Close"].rolling(window=20, min_periods=1).mean()
df_full["MA_50"] = df_full["Log_Close"].rolling(window=50, min_periods=1).mean()
df_full["EMA_10"] = df_full["Log_Close"].ewm(span=10, min_periods=1).mean()
df_full["EMA_20"] = df_full["Log_Close"].ewm(span=20, min_periods=1).mean()
df_full["Lag_1"] = df_full["Log_Close"].shift(1)
df_full["Lag_2"] = df_full["Log_Close"].shift(2)
df_full["Lag_3"] = df_full["Log_Close"].shift(3)
df_full["Lag_5"] = df_full["Log_Close"].shift(5)
# RSI (14-day)
delta = df_full["Log_Close"].diff()
gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
rs = gain / loss.replace(0, 1e-10)
df_full["RSI"] = 100 - (100 / (1 + rs))
# Bollinger Band width
bb_std = df_full["Log_Close"].rolling(20, min_periods=1).std().fillna(0)
df_full["BB_width"] = bb_std * 2
# Momentum
df_full["Momentum_5"] = df_full["Log_Close"] - df_full["Log_Close"].shift(5)
df_full["Momentum_10"] = df_full["Log_Close"] - df_full["Log_Close"].shift(10)
df_full = df_full.bfill()

# Normalization info (for display)
scaler = MinMaxScaler()
df_full["Close_Scaled"] = scaler.fit_transform(df_full[["Close"]])

st.write(f"- Forward-filled missing business days")
st.write(f"- Removed **{outliers_removed}** extreme outlier days (1st/99th percentile returns)")
st.write(f"- Log-transformed target for stability")
st.write(f"- Added features: MA, EMA, Lags, RSI, Bollinger Bands, Momentum")
st.write(f"- Clean dataset: **{len(df_full)}** records")

# ── 3. Train / Test Split — 4 years train, 1 year test ───────────────────────
split_date = df_full["Date"].max() - pd.DateOffset(years=1)
train = df_full[df_full["Date"] <= split_date].copy()
test = df_full[df_full["Date"] > split_date].copy()

st.subheader("✂️ Train / Test Split")
st.write(f"- **Train:** {train['Date'].min().date()} → {train['Date'].max().date()}  ({len(train)} rows)")
st.write(f"- **Test:**  {test['Date'].min().date()} → {test['Date'].max().date()}  ({len(test)} rows)")

# ── 4. Prophet Model with Tuning ─────────────────────────────────────────────
st.subheader("🔮 Prophet Model Training & Prediction")

with st.spinner("Training Prophet model (tuned for lower RMSE)..."):
    # Prepare data for Prophet (log-transformed target)
    regressor_cols = ["MA_5", "MA_10", "MA_20", "MA_50", "EMA_10", "EMA_20",
                      "Lag_1", "Lag_2", "Lag_3", "Lag_5",
                      "RSI", "BB_width", "Momentum_5", "Momentum_10"]
    train_prophet = train[["Date", "Log_Close"] + regressor_cols].rename(columns={"Date": "ds", "Log_Close": "y"})
    test_prophet = test[["Date", "Close", "Log_Close"] + regressor_cols].rename(columns={"Date": "ds", "Log_Close": "y"})

    def build_prophet():
        m = Prophet(
            changepoint_prior_scale=0.02,
            seasonality_prior_scale=15.0,
            seasonality_mode="multiplicative",
            yearly_seasonality=20,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_range=0.95,
            n_changepoints=50,
        )
        m.add_country_holidays(country_name="US")
        m.add_seasonality(name="monthly", period=30.5, fourier_order=8)
        m.add_seasonality(name="quarterly", period=91.25, fourier_order=5)
        for col in regressor_cols:
            m.add_regressor(col, mode="multiplicative")
        return m

    model = build_prophet()
    model.fit(train_prophet)

    # Predict on test period
    test_forecast = model.predict(test_prophet[["ds"] + regressor_cols])
    # Inverse log for display
    test_forecast["yhat_orig"] = np.exp(test_forecast["yhat"])
    test_forecast["yhat_lower_orig"] = np.exp(test_forecast["yhat_lower"])
    test_forecast["yhat_upper_orig"] = np.exp(test_forecast["yhat_upper"])

    # For future forecast, retrain on ALL data
    all_prophet = df_full[["Date", "Log_Close"] + regressor_cols].rename(columns={"Date": "ds", "Log_Close": "y"})
    full_model = build_prophet()
    full_model.fit(all_prophet)

    # Build future dataframe with regressor values carried forward
    future_dates = full_model.make_future_dataframe(periods=252, freq="B")
    future_dates = future_dates.merge(
        all_prophet[["ds"] + regressor_cols], on="ds", how="left"
    )
    for col in regressor_cols:
        future_dates[col] = future_dates[col].ffill()

    full_forecast = full_model.predict(future_dates)
    full_forecast["yhat_orig"] = np.exp(full_forecast["yhat"])
    full_forecast["yhat_lower_orig"] = np.exp(full_forecast["yhat_lower"])
    full_forecast["yhat_upper_orig"] = np.exp(full_forecast["yhat_upper"])

# ── 5. Evaluation Metrics ─────────────────────────────────────────────────────
y_true = test_prophet["Close"].values
y_pred = test_forecast["yhat_orig"].values

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

st.subheader("📈 Model Performance on Test Set")
m1, m2, m3 = st.columns(3)
m1.metric("RMSE", f"${rmse:.2f}")
m2.metric("MAE", f"${mae:.2f}")
m3.metric("MAPE", f"{mape:.2f}%")

# ── 6. Visualizations ────────────────────────────────────────────────────────

# 6a. Full Historical + Forecast Chart
st.subheader("📉 Historical Data + Forecast (Next 1 Year)")
fig1 = go.Figure()

fig1.add_trace(go.Scatter(
    x=train["Date"], y=train["Close"],
    mode="lines", name="Train Data",
    line=dict(color="steelblue", width=1.5)
))
fig1.add_trace(go.Scatter(
    x=test["Date"], y=test["Close"],
    mode="lines", name="Test Data (Actual)",
    line=dict(color="orange", width=1.5)
))
fig1.add_trace(go.Scatter(
    x=test_forecast["ds"], y=test_forecast["yhat_orig"],
    mode="lines", name="Test Prediction",
    line=dict(color="red", width=1.5, dash="dash")
))

# Future forecast (beyond test)
future_only = full_forecast[full_forecast["ds"] > df_full["Date"].max()]
fig1.add_trace(go.Scatter(
    x=future_only["ds"], y=future_only["yhat_orig"],
    mode="lines", name="Future Forecast (1 Year)",
    line=dict(color="green", width=2)
))
fig1.add_trace(go.Scatter(
    x=pd.concat([future_only["ds"], future_only["ds"][::-1]]),
    y=pd.concat([future_only["yhat_upper_orig"], future_only["yhat_lower_orig"][::-1]]),
    fill="toself", fillcolor="rgba(0,200,0,0.1)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Confidence Interval"
))
fig1.update_layout(
    xaxis_title="Date", yaxis_title="Price (USD)",
    height=550, hovermode="x unified",
    legend=dict(orientation="h", y=-0.15)
)
st.plotly_chart(fig1, use_container_width=True)

# 6b. Test Period: Actual vs Predicted
st.subheader("🎯 Test Period — Actual vs Predicted")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=test["Date"], y=test["Close"],
    mode="lines", name="Actual",
    line=dict(color="orange", width=2)
))
fig2.add_trace(go.Scatter(
    x=test_forecast["ds"], y=test_forecast["yhat_orig"],
    mode="lines", name="Predicted",
    line=dict(color="red", width=2, dash="dash")
))
fig2.add_trace(go.Scatter(
    x=pd.concat([test_forecast["ds"], test_forecast["ds"][::-1]]),
    y=pd.concat([test_forecast["yhat_upper_orig"], test_forecast["yhat_lower_orig"][::-1]]),
    fill="toself", fillcolor="rgba(255,0,0,0.1)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Confidence Interval"
))
fig2.update_layout(
    xaxis_title="Date", yaxis_title="Price (USD)",
    height=450, hovermode="x unified"
)
st.plotly_chart(fig2, use_container_width=True)

# 6c. Prophet Components
st.subheader("📐 Forecast Components (Trend + Seasonality)")
fig_components = full_model.plot_components(full_forecast)
st.pyplot(fig_components)

# 6d. Residual Analysis
st.subheader("🔍 Residual Analysis")
residuals = y_true - y_pred
fig3 = make_subplots(rows=1, cols=2, subplot_titles=("Residuals Over Time", "Residual Distribution"))
fig3.add_trace(go.Scatter(
    x=test["Date"], y=residuals,
    mode="lines+markers", marker=dict(size=3),
    name="Residuals", line=dict(color="purple")
), row=1, col=1)
fig3.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
fig3.add_trace(go.Histogram(
    x=residuals, nbinsx=40, name="Distribution",
    marker_color="mediumpurple"
), row=1, col=2)
fig3.update_layout(height=400, showlegend=False)
st.plotly_chart(fig3, use_container_width=True)

# 6e. Data Tables
st.subheader("📋 Forecast Data")
tab1, tab2 = st.tabs(["Future Forecast", "Test Predictions"])
with tab1:
    future_display = future_only[["ds", "yhat_orig", "yhat_lower_orig", "yhat_upper_orig"]].copy()
    future_display.columns = ["Date", "Predicted Price", "Lower Bound", "Upper Bound"]
    future_display = future_display.round(2)
    st.dataframe(future_display, use_container_width=True, hide_index=True)
with tab2:
    test_display = test_forecast[["ds", "yhat_orig"]].copy()
    test_display["Actual"] = y_true
    test_display["Error"] = y_true - y_pred
    test_display.columns = ["Date", "Predicted", "Actual", "Error"]
    test_display = test_display.round(2)
    st.dataframe(test_display, use_container_width=True, hide_index=True)
