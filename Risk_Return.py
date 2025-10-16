import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt



asset_return1 = [0.05, 0.3, -0.1, 0.35, 0.2]
asset_return2 = [0.5, -0.2, 0.3, 0.5, -0.3]

print(np.mean(asset_return1))
print(np.mean(asset_return2))
print(np.mean(asset_return1) == np.mean(asset_return2))

return_df = pd.DataFrame({"Asset1":asset_return1, "Asset2":asset_return2})
# print(return_df)
print(return_df.std())


# R1= return_df+1
# initial_inv = 100
# cum_value = R1.cumprod()*initial_inv

# print(cum_value)
# cum_value.plot.line()
#plt.show()

print(return_df.pct_change())




# return_df.plot.bar()
