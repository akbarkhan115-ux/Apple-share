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
    df = df[["Close"]].copy()
    df.columns = ["Close"]
    df = df.reset_index()
    df.columns = ["Date", "Close"]
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
df_full = df_full.dropna()

# Outlier removal using IQR on daily returns
df_full["Return"] = df_full["Close"].pct_change()
Q1 = df_full["Return"].quantile(0.01)
Q3 = df_full["Return"].quantile(0.99)
mask = df_full["Return"].between(Q1, Q3) | df_full["Return"].isna()
outliers_removed = (~mask).sum()
df_full = df_full[mask].copy()
df_full = df_full.drop(columns=["Return"])

# Normalization info (for display; Prophet works on original scale)
scaler = MinMaxScaler()
df_full["Close_Scaled"] = scaler.fit_transform(df_full[["Close"]])

st.write(f"- Forward-filled missing business days")
st.write(f"- Removed **{outliers_removed}** extreme outlier days (1st/99th percentile returns)")
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
    # Prepare data for Prophet
    train_prophet = train[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
    test_prophet = test[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

    # Tuned Prophet model
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
    model.fit(train_prophet)

    # Predict on test period
    test_forecast = model.predict(test_prophet[["ds"]])

    # Predict next 1 year into the future
    future_dates = model.make_future_dataframe(periods=365, freq="B")
    full_forecast = model.predict(future_dates)

# ── 5. Evaluation Metrics ─────────────────────────────────────────────────────
y_true = test_prophet["y"].values
y_pred = test_forecast["yhat"].values

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
    x=test_forecast["ds"], y=test_forecast["yhat"],
    mode="lines", name="Test Prediction",
    line=dict(color="red", width=1.5, dash="dash")
))

# Future forecast (beyond test)
future_only = full_forecast[full_forecast["ds"] > df_full["Date"].max()]
fig1.add_trace(go.Scatter(
    x=future_only["ds"], y=future_only["yhat"],
    mode="lines", name="Future Forecast (1 Year)",
    line=dict(color="green", width=2)
))
fig1.add_trace(go.Scatter(
    x=pd.concat([future_only["ds"], future_only["ds"][::-1]]),
    y=pd.concat([future_only["yhat_upper"], future_only["yhat_lower"][::-1]]),
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
    x=test_forecast["ds"], y=test_forecast["yhat"],
    mode="lines", name="Predicted",
    line=dict(color="red", width=2, dash="dash")
))
fig2.add_trace(go.Scatter(
    x=pd.concat([test_forecast["ds"], test_forecast["ds"][::-1]]),
    y=pd.concat([test_forecast["yhat_upper"], test_forecast["yhat_lower"][::-1]]),
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
fig_components = model.plot_components(full_forecast)
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
    future_display = future_only[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    future_display.columns = ["Date", "Predicted Price", "Lower Bound", "Upper Bound"]
    future_display = future_display.round(2)
    st.dataframe(future_display, use_container_width=True, hide_index=True)
with tab2:
    test_display = test_forecast[["ds", "yhat"]].copy()
    test_display["Actual"] = y_true
    test_display["Error"] = y_true - y_pred
    test_display.columns = ["Date", "Predicted", "Actual", "Error"]
    test_display = test_display.round(2)
    st.dataframe(test_display, use_container_width=True, hide_index=True)
