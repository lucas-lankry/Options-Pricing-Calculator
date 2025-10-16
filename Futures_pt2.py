import vectorbt as vbt 
import matplotlib.pyplot as plt 
import ta

futures_symbol = "ES=F"
futures_data = vbt.YFData.download(futures_symbol, missing_index='drop', start="2022-01-01", end="2022-04-01", interval="1d")
futures_price = futures_data.get('Close')

# Calculate RSI
RSI = ta.momentum.RSIIndicator(futures_price).rsi()
# Calculate Bollinger Bands
bbands = ta.volatility.BollingerBands(futures_price)
BB_upper = bbands.bollinger_hband()
BB_lower = bbands.bollinger_lband()
# Calculate MACD
macd = ta.trend.MACD(futures_price)
futures_MACD = macd.macd()
futures_MACD_signal = macd.macd_signal()

# Create subplots for each indicator
fig, axes = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
# Plot closing price
axes[0].plot(futures_price.index, futures_price, label="Close")
axes[0].set_title("S&P 500 E-Mini Futures - Closing Price")
axes[0].grid()
# Plot RSI
axes[1].plot(futures_price.index, RSI, label="RSI", color="g")
axes[1].axhline(30, linestyle="--", color="r", alpha=0.5)
axes[1].axhline(70, linestyle="--", color="r", alpha=0.5)
axes[1].set_title("Relative Strength Index (RSI)")
axes[1].grid()
# Plot Bollinger Bands
axes[2].plot(futures_price.index, futures_price, label="Close")
axes[2].plot(futures_price.index, BB_upper, label="Upper Bollinger Band", linestyle="--", color="r")
axes[2].plot(futures_price.index, BB_lower, label="Lower Bollinger Band", linestyle="--", color="r")
axes[2].set_title("Bollinger Bands")
axes[2].grid()
# Plot MACD
axes[3].plot(futures_price.index, futures_MACD, label="MACD", color="b")
axes[3].plot(futures_price.index, futures_MACD_signal, label="Signal Line", linestyle="--", color="r")
axes[3].axhline(0, linestyle="--", color="k", alpha=0.5)
axes[3].set_title("Moving Average Convergence Divergence (MACD)")
axes[3].grid()

plt.show()







# Calcul des bandes Bollinger (window 20, nb_dev=2 standard)
# window = 20
# ma = futures_price.rolling(window).mean()
# std = futures_price.rolling(window).std()
# upper = ma + 2 * std
# lower = ma - 2 * std

# plt.figure(figsize=(12, 6))
# plt.plot(futures_price, label='Close')
# plt.plot(upper, label='Bollinger Upper', linestyle='--', color='gray')
# plt.plot(lower, label='Bollinger Lower', linestyle='--', color='gray')
# plt.legend()
# plt.grid(True)
# plt.title('ES=F Close Price with Bollinger Bands')
# plt.show()