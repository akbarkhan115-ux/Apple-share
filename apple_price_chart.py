import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Download Apple stock data for the last 5 years
aapl = yf.download("AAPL", period="5y")

# Plot the closing price
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(aapl.index, aapl["Close"], color="steelblue", linewidth=1.2)

ax.set_title("Apple (AAPL) Share Price — Last 5 Years", fontsize=16, fontweight="bold")
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Price (USD)", fontsize=12)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.grid(True, linestyle="--", alpha=0.5)
fig.tight_layout()

plt.savefig("apple_price_chart.png", dpi=150)
print("Chart saved to apple_price_chart.png")
plt.show()
