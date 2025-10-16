import vectorbt as vbt 
import matplotlib.pyplot as plt 
import pandas as pd

aapl_data = vbt.YFData.download('AAPL', missing_index='drop', start = "2022-01-01", end="2023-01-01")
aapl_price = aapl_data.get('Close')

ma_3 = vbt.MA.run(aapl_price, window = 3).ma
ma_20 = vbt.MA.run(aapl_price, window = 20).ma
exponential_ma = vbt.MA.run(aapl_price, window = 20 ,ewm = True).ma

MA_short = vbt.MA.run(aapl_price, window = 3)
MA_long = vbt.MA.run(aapl_price, window = 20)


entries = MA_short.ma_crossed_above(MA_long)
exits = MA_short.ma_crossed_below(MA_long)

pf = vbt.Portfolio.from_signals(aapl_price, entries, exits)

#visualize with Vbt
# pf.plot().show()

plt.figure(figsize = (12,5))
plt.title("AAPL Close Price & MA")
plt.plot(aapl_price, label = 'Close Price', color = 'blue')
plt.plot(aapl_price.index, ma_3, label='MA (3 jours)', color='red')
plt.plot(aapl_price.index, ma_20, label='MA (20 jours)', color='orange')
#plt.plot(aapl_price.index, exponential_ma, label='EMA (20 jours)', color='green', alpha = 0.5)

plt.plot(entries[entries].index, aapl_price[entries], '^', color = 'g', markersize = 8, label='Entrée')
plt.plot(exits[exits].index, aapl_price[exits], 'v', color = 'r', markersize = 8, label='Sortie')


plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
