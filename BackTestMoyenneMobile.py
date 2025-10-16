#BACKTEST D'UNE STRATEGIE DE MOYENNE MOBILE SUR SP500

import vectorbt as vbt

SPY_price = vbt.YFData.download("SPY", missing_index='drop').get("Close")

print(SPY_price)

ma_50 = vbt.MA.run(SPY_price, window = 50)
ma_20 = vbt.MA.run(SPY_price, window = 20)

entries = ma_20.ma_crossed_above(ma_50)
exits= ma_20.ma_crossed_below(ma_50)

pf = vbt.Portfolio.from_signals(SPY_price, entries, exits)

print(pf.stats())

#pf.plot().write_html("spy_ma_backtest.html")

