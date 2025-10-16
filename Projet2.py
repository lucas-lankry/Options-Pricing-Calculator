#Arbitrage statistique : Pairs Trading sur KO et PEP

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import vectorbt as vbt

# 1. Télécharger les données

ko = vbt.YFData.download("KO", missing_index='drop').get("Close")
pep = vbt.YFData.download("PEP", missing_index='drop').get("Close")


# 2. Fusionner les données
prices = pd.concat([ko.rename("KO"), pep.rename("PEP")], axis=1).dropna()


score, pvalue, _ = coint(prices["KO"], prices["PEP"])
print(f"Test de co-intégration : p-value = {pvalue:.4f}")
if pvalue > 0.05:
    print("⚠️ Pas de co-intégration détectée, la stratégie peut ne pas fonctionner.")
else:
    print("✅ Co-intégration détectée, on peut continuer.")

# 4. Régression linéaire KO ~ PEP
X = sm.add_constant(prices["PEP"])
model = sm.OLS(prices["KO"], X).fit()
beta = model.params["PEP"]
spread = prices["KO"] - beta * prices["PEP"]

# 5. Calcul du z-score
zscore = (spread - spread.mean()) / spread.std()

# 6. Signal de trading
long_signal = zscore < -1
short_signal = zscore > 1
exit_signal = abs(zscore) < 0.5

# 7. Backtest simplifié
positions = pd.DataFrame(index=prices.index)
positions["KO"] = 0
positions["PEP"] = 0
positions.loc[long_signal, "KO"] = 1
positions.loc[long_signal, "PEP"] = -beta
positions.loc[short_signal, "KO"] = -1
positions.loc[short_signal, "PEP"] = beta

# Maintenir la position jusqu'au signal de sortie
positions = positions.ffill()
positions[exit_signal] = 0

# 8. Calcul du PnL
returns = prices.pct_change().dropna()
pnl = (positions.shift() * returns).sum(axis=1)
equity_curve = pnl.cumsum()

# 9. Plot
plt.figure(figsize=(12, 6))
plt.plot(equity_curve, label="Equity Curve")
plt.title("Pairs Trading KO/PEP")
plt.legend()
plt.grid(True)
plt.show()
