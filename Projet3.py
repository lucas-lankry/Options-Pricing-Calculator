# Analyse de séries temporelles (forecasting)

import vectorbt as vbt
from statsmodels.tsa.stattools import adfuller 
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
import yfinance as yf 



# 1. Charger les prix
SPY_price = vbt.YFData.download("SPY", missing_index='drop').get("Close")

# 2. Test ADF
result = adfuller(SPY_price.dropna())
print(f'p-value ADF : {result[1]:.4f}')

# 3. Train/test split
train_price = SPY_price[:-100]
test_price = SPY_price[-100:]

# 4. Déterminer d et construire le modèle
if result[1] > 0.05:
    print("🔁 Série non stationnaire → ARIMA(1,1,1) sur les prix bruts")
    d = 1
    model = ARIMA(train_price, order=(3, d, 3))
else:
    print("✅ Série stationnaire → ARIMA(1,0,1) sur les prix bruts")
    d = 0
    model = ARIMA(train_price, order=(3, d, 3))

# 5. Fit ARIMA
model_fit = model.fit()
print(model_fit.summary())

# 6. Prédictions (le modèle gère d lui-même)
forecast = model_fit.forecast(steps=100)

# 7. Évaluation
rmse = np.sqrt(mean_squared_error(test_price, forecast))
print(f'✅ RMSE ARIMA : {rmse:.2f}')

# 8. Plot
plt.figure(figsize=(12,5))
plt.plot(SPY_price.index[-200:], SPY_price[-200:], label='Réel')
plt.plot(test_price.index, forecast, label='Prévision ARIMA', linestyle='--')
plt.title('Prévision SP500 - ARIMA vs Réel')
plt.legend()
plt.grid(True)
plt.show()



# 4. Différenciation première
#sp500_diff = SPY_price.diff().dropna()
# fig, (ax1, ax2)= plt.subplots(1,2, figsize=(16,5), dpi=80)
# plot_acf(sp500_diff)
# plot_pacf(sp500_diff, method='ywm')
# ax1.tick_params(axis= 'both', labelsize = 12)
# ax2.tick_params(axis= 'both', labelsize = 12)
# plt.show()
