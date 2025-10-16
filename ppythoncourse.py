# from datetime import datetime, date, time 

# values = ([1,2],2,3,4,5)

# values[0].append(6)

# a, b, *_ = values
# print(*_)
import numpy as np
# tup = ("foo", "bar", "baz")
# lia = list(tup)
# print(lia)

import vectorbt as vbt

df = vbt.YFData.download("AMZN", start='2024-10-01', end='2025-10-02', missing_index='drop').get('Close')
df.index = df.index.tz_localize(None)

df.to_excel("amzn_close_prices.xlsx")

print(df.tail())
