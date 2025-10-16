import os
import random
import numpy as np
import vectorbt as vbt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from matplotlib import pyplot as plt


SEED = 8
random.seed(SEED)
np.random.seed(SEED)

start_date  = "2022-01-01"
end_date  = "2022-12-31"
stocks = ['GOOG','MSFT']

df = vbt.YFData.download(stocks, start = start_date, end= end_date, missing_index='drop').get('Close')

#MODELE reGRESSION LINEAIRE: GOOGLE = Alpha + MSFT*X + Epsilon

# build linear regression model
# Extract prices for two stocks of interest
# target var: Y; predictor: X
Y = df[stocks[0]]
X = df[stocks[1]]

# estimate linear regression coefficients of stock1 on stock2
X_with_constant = sm.add_constant(X)
model = OLS(Y, X_with_constant).fit()
residuals = Y - model.predict()

# access model weights
print(model.params)

# test stationarity of the residuals
adf_test = adfuller(residuals)
print(f"ADF test statistic: {adf_test[0]}")
print(f"p-value: {adf_test[1]}")
if adf_test[1] < 0.05:
    print("The two stocks are cointegrated.")
else:
    print("The two stocks are not cointegrated.")
