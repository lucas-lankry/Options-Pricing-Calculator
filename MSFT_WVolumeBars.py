from datetime import datetime 
import matplotlib.pyplot as plt
import vectorbt as vbt
import mplfinance as mpf
#import mplcursors


data = vbt.YFData.download('MSFT', missing_index='drop' ,start="2022-01-01", end="2022-12-31")


data_price = data.get('Close')
volume = data.get('Volume')
dates = data_price.index


fig, ax1=plt.subplots(figsize=(12,5))
ax1.plot(data_price, color='blue', label='Prix')
plt.title("MSFT CLOSE PRICE")
ax1.legend(["Close Price"])
ax1.grid(True)

ax2 = ax1.twinx()
bars = ax2.bar(dates, volume, color='red', alpha=0.3, label='Volume')
ax2.set_ylim(0 , 500000000)


plt.show()



#Curseur interactif
# line, = plt.plot(data_close)
#mplcursors.cursor(line, hover=True)
