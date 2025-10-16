import vectorbt as vbt 
import pandas as pd
import numpy as np 
import datetime


end_time = datetime.datetime.now()
start_time = end_time - datetime.timedelta(days = 2)

btc_price = vbt.YFData.download('BTC-USD', 
                                missing_index= 'drop',
                                start = start_time,
                                end = end_time,
                                interval= '1m').get("Close")


def custom_indicator(close, rsi_window = 14, ma_window = 50):
    rsi = vbt.RSI.run(close, window = rsi_window).rsi.to_numpy()
    ma = vbt.MA.run(close, ma_window ).ma.to_numpy()
    trend = np.where( rsi>70, -1, 0)
    trend = np.where( (rsi<30) & (close<ma), 1, trend)
    return trend

ind = vbt.IndicatorFactory(
    class_name = "Combination",
    short_name= "comb",
    input_names=["close"],
    param_names=["rsi_window", "ma_window"],
    output_names=["value"]
).from_apply_func(custom_indicator,
                  rsi_window=14,
                  ma_window=50)

res = ind.run(btc_price, 
              rsi_window =21, 
              ma_window= 50)

print(res.value.to_string())

entries= res.value == 1.0
exits = res.value == -1.0

pf = vbt.Portfolio.from_signals(btc_price, entries, exits)

print(pf.stats())


#print(btc_price)
 
