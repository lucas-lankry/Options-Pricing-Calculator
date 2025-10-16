import vectorbt as vbt
import mplfinance as mpf
import pandas as pd

# Télécharger les données avec vectorbt
data = vbt.YFData.download('MSFT', missing_index='drop', start="2022-01-01", end="2022-12-31")

# Récupérer les données OHLCV
ohlcv = data.get(['Open', 'High', 'Low', 'Close', 'Volume'])

# Aplatir les colonnes si elles ont un multi-index
# if isinstance(ohlcv.columns, pd.MultiIndex):
#     ohlcv.columns = ohlcv.columns.droplevel(1)

# Affichage en bougies japonaises + volume intégré
mpf.plot(
    ohlcv,
    type='candle',
    volume=True,                     
    style='yahoo',                   # Tu peux changer pour 'charles', 'nightclouds', etc.
    title='MSFT Candlestick Chart with Volume (2022)',
    ylabel='Price ($)',
    ylabel_lower='Volume',
    figratio=(12, 5),
    figscale=1.2
)

