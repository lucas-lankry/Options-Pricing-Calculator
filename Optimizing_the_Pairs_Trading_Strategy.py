

import os
import math
import torch
import random
import numpy as np
from matplotlib import pyplot as plt
import torch.nn as nn
import pandas as pd
import vectorbt as vbt
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class QTS_OPTIMIZER(nn.Module):
    def __init__(self, ticker_pair, start_date, end_date, riskfree_rate=0.04):
        super(QTS_OPTIMIZER, self).__init__()
        self.ticker_pair = ticker_pair
        self.start_date = start_date
        self.end_date = end_date
        self.riskfree_rate = riskfree_rate
        self.stock = self.get_stock_data()

    def get_stock_data(self):
        print("===== DOWNLOADING STOCK DATA =====")
        df = vbt.YFData.download(["GOOG", "MSFT"], start=self.start_date, end=self.end_date).get('Close')
        print("===== DOWNLOAD COMPLETE =====")
        print(df.head())
        return pd.DataFrame(df)


    def forward(self, entry_threshold, exit_threshold, window_size=10):
        # add sma columns
        stock_df = self.stock.copy()
        # calculate the spread for GOOG and MSFT
        Y = stock_df[self.ticker_pair[0]]
        X = stock_df[self.ticker_pair[1]]
        # estimate linear regression coefficients
        X_with_constant = sm.add_constant(X)
        model = OLS(Y, X_with_constant).fit()
        # obtain the spread as the residuals
        spread = Y - model.predict()
        # calculate rolling mean and sd
        spread_mean = spread.rolling(window=window_size).mean()
        spread_std = spread.rolling(window=window_size).std()
        zscore = (spread - spread_mean) / spread_std
        # remove initial days with NA
        first_valid_idx = zscore.first_valid_index()
        zscore = zscore[first_valid_idx:]
        # initialize the daily positions to be zeros
        stock1_position = pd.Series(data=0, index=zscore.index)
        stock2_position = pd.Series(data=0, index=zscore.index)
        # generate daily entry and exit signals for each stock
        for i in range(1, len(zscore)):
            # zscore<-entry_threshold and no existing long position for stock 1
            if zscore[i] < -entry_threshold and stock1_position[i-1] == 0:
                stock1_position[i] = 1 # long stock 1
                stock2_position[i] = -1 # short stock 2
            # zscore>entry_threshold and no existing short position for stock 2
            elif zscore[i] > entry_threshold and stock2_position[i-1] == 0:
                stock1_position[i] = -1 # short stock 1
                stock2_position[i] = 1 # long stock 2
            # -exit_threshold<zscore<exit_threshold
            elif abs(zscore[i]) < exit_threshold:
                stock1_position[i] = 0 # exit existing position
                stock2_position[i] = 0
            # -entry_threshold<zscore<-exit_threshold or exit_threshold<zscore<entry_threshold
            else:
                stock1_position[i] = stock1_position[i-1] # maintain existing position
                stock2_position[i] = stock2_position[i-1]
        # Calculate the returns of each stock
        stock1_returns = (Y[first_valid_idx:].pct_change() * stock1_position.shift(1)).fillna(0)
        stock2_returns = (X[first_valid_idx:].pct_change() * stock2_position.shift(1)).fillna(0)
        # calculate the total returns of the strategy
        total_returns = stock1_returns + stock2_returns
        # calculate annualized return
        annualized_return = (1 + total_returns).prod()**(252/Y[first_valid_idx:].shape[0])-1
        # calculate annualized volatility
        annualized_vol = total_returns.std()*(252**0.5)
        if annualized_vol==0:
            annualized_vol = 100
        # calculate Sharpe ratio
        sharpe_ratio = (annualized_return - self.riskfree_rate) / annualized_vol
        return sharpe_ratio



qts = QTS_OPTIMIZER(ticker_pair=["GOOG","MSFT"], start_date="2022-01-01", end_date="2023-01-01")
print(qts(entry_threshold=2, exit_threshold=1))

# generate initial training dataset for optimization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
x1_bound = [1,3]
x2_bound = [0,1]

def generate_initial_data(n=10):
    # generate random initial locations
    train_x1 = x1_bound[0] + (x1_bound[1] - x1_bound[0]) * torch.rand(size=(n,1), device=device, dtype=dtype)
    train_x2 = torch.rand(size=(n,1), device=device, dtype=dtype)
    train_x = torch.cat((train_x1, train_x2), 1)
    # obtain the exact value of the objective function and add output dimension
    train_y = []
    for i in range(len(train_x)):
        train_y.append(qts(entry_threshold=train_x1[i], exit_threshold=train_x2[i]))
    train_y = torch.Tensor(train_y, device=device).to(dtype).unsqueeze(-1)
    # get the current best observed value, i.e., utility of the available dataset
    best_observed_value = train_y.max().item()
    return train_x, train_y, best_observed_value

train_x, train_y, best_observed_value = generate_initial_data(n=3)
print(train_x)
print(train_y)
print(best_observed_value)

# initialize GP model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood


def initialize_model(train_x, train_y):
    # create a single-task exact GP model instance
    # use a GP prior with Matern kernel and constant mean function by default
    model = SingleTaskGP(train_X=train_x, train_Y=train_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

mll, model = initialize_model(train_x, train_y)
list(model.named_hyperparameters())

# optimize GP hyperparameters
from botorch.fit import fit_gpytorch_mll

# fit hyperparameters (kernel parameters and noise variance) of a GPyTorch model
fit_gpytorch_mll(mll.cpu());
mll = mll.to(train_x)
model = model.to(train_x)
list(model.named_hyperparameters())


# define acquisition function
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition import UpperConfidenceBound
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient

# call helper functions to generate initial training data and initialize model
train_x, train_y, best_observed_value = generate_initial_data(n=3)
train_x_ei = train_x
train_x_qei = train_x
train_x_ucb = train_x
train_x_qkg = train_x
train_y_ei = train_y
train_y_qei = train_y
train_y_ucb = train_y
train_y_qkg = train_y
mll_ei, model_ei = initialize_model(train_x_ei, train_y_ei)
mll_qei, model_qei = initialize_model(train_x_qei, train_y_qei)
mll_ucb, model_ucb = initialize_model(train_x_ucb, train_y_ucb)
mll_qkg, model_qkg = initialize_model(train_x_qkg, train_y_qkg)
EI = ExpectedImprovement(model=model_ei, best_f=best_observed_value)
qEI = qExpectedImprovement(model=model_qei, best_f=best_observed_value)
beta = 0.8
UCB = UpperConfidenceBound(model=model_ucb, beta=beta)
num_fantasies = 64
qKG = qKnowledgeGradient(
    model=model_qkg,
    num_fantasies=num_fantasies,
    X_baseline=train_x,
    q=1
)


# optimize and get new observation
from botorch.optim import optimize_acqf
# get search bounds
bounds = torch.tensor([[x1_bound[0], x2_bound[0]], [x1_bound[1], x2_bound[1]]], device=device, dtype=dtype)
# parallel candidate locations generated in each iteration
BATCH_SIZE = 1
# number of starting points for multistart optimization
NUM_RESTARTS = 10
# number of samples for initialization
RAW_SAMPLES = 1024
def optimize_acqf_and_get_observation(acq_func):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidates, value = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
    )
    # observe new values
    new_x = candidates.detach()
    # sample output value
    new_y = qts(entry_threshold=new_x.squeeze()[0].item(), exit_threshold=new_x.squeeze()[1].item())
    # add output dimension
    new_y = torch.Tensor([new_y], device=device).to(dtype).unsqueeze(-1)
    # print("new fn value:", new_y)
    return new_x, new_y


def update_random_observations(best_random):
    """Simulates a random policy by drawing a new random points,
        observing their values, and updating the current best candidate to the running list.
    """
    new_x1 = x1_bound[0] + (x1_bound[1] - x1_bound[0]) * torch.rand(size=(1,1), device=device, dtype=dtype)
    new_x2 = torch.rand(size=(1,1), device=device, dtype=dtype)
    new_x = torch.cat((new_x1, new_x2), 1)
    new_y = qts(entry_threshold=new_x[0,0].item(), exit_threshold=new_x[0,1].item())
    best_random.append(max(best_random[-1], new_y))
    return best_random


# single trial
import time
N_ROUND = 20
verbose = True
beta = 0.8
best_random, best_observed_ei, best_observed_qei, best_observed_ucb, best_observed_qkg  = [], [], [], [], []
best_random.append(best_observed_value)
best_observed_ei.append(best_observed_value)
best_observed_qei.append(best_observed_value)
best_observed_ucb.append(best_observed_value)
best_observed_qkg.append(best_observed_value)
# run N_ROUND rounds of BayesOpt after the initial random batch
for iteration in range(1, N_ROUND + 1):
    t0 = time.monotonic()
    # fit the models
    fit_gpytorch_mll(mll_ei)
    fit_gpytorch_mll(mll_qei)
    fit_gpytorch_mll(mll_ucb)
    fit_gpytorch_mll(mll_qkg)
    # for best_f, we use the best observed exact values
    EI = ExpectedImprovement(model=model_ei, best_f=train_y_ei.max())
    qEI = qExpectedImprovement(model=model_qei,
                               best_f=train_y_ei.max(),
                               num_samples=1024
                               )
    UCB = UpperConfidenceBound(model=model_ucb, beta=beta)
    qKG = qKnowledgeGradient(
        model=model_qkg,
        num_fantasies=64,
        objective=None,
        X_baseline=train_x_qkg,
    )
    # optimize and get new observation
    new_x_ei, new_y_ei = optimize_acqf_and_get_observation(EI)
    new_x_qei, new_y_qei = optimize_acqf_and_get_observation(qEI)
    new_x_ucb, new_y_ucb = optimize_acqf_and_get_observation(UCB)
    new_x_qkg, new_y_qkg = optimize_acqf_and_get_observation(qKG)
    # update training points
    train_x_ei = torch.cat([train_x_ei, new_x_ei], dim=0)
    train_x_qei = torch.cat([train_x_qei, new_x_qei], dim=0)
    train_x_ucb = torch.cat([train_x_ucb, new_x_ucb], dim=0)
    train_x_qkg = torch.cat([train_x_qkg, new_x_qkg], dim=0)
    train_y_ei = torch.cat([train_y_ei, new_y_ei], dim=0)
    train_y_qei = torch.cat([train_y_qei, new_y_qei], dim=0)
    train_y_ucb = torch.cat([train_y_ucb, new_y_ucb], dim=0)
    train_y_qkg = torch.cat([train_y_qkg, new_y_qkg], dim=0)
    # update progress
    best_random = update_random_observations(best_random)
    best_value_ei = max(best_observed_ei[-1], new_y_ei.item())
    best_value_qei = max(best_observed_qei[-1], new_y_qei.item())
    best_value_ucb = max(best_observed_ucb[-1], new_y_ucb.item())
    best_value_qkg = max(best_observed_qkg[-1], new_y_qkg.item())
    best_observed_ei.append(best_value_ei)
    best_observed_qei.append(best_value_qei)
    best_observed_ucb.append(best_value_ucb)
    best_observed_qkg.append(best_value_qkg)
    # reinitialize the models so they are ready for fitting on next iteration
    mll_ei, model_ei = initialize_model(
        train_x_ei,
        train_y_ei
    )
    mll_qei, model_qei = initialize_model(
        train_x_qei,
        train_y_qei
    )
    mll_ucb, model_ucb = initialize_model(
        train_x_ucb,
        train_y_ucb
    )
    mll_qkg, model_qkg = initialize_model(
        train_x_qkg,
        train_y_qkg
    )
    t1 = time.monotonic()


iters = np.arange(N_ROUND + 1) * BATCH_SIZE
plt.plot(iters, best_random, label='random')
plt.plot(iters, best_observed_ei, label='EI')
plt.plot(iters, best_observed_qei, label='qEI')
plt.plot(iters, best_observed_ucb, label='UCB')
plt.plot(iters, best_observed_qkg, label='qKG')
plt.legend()
plt.xlabel("Sampling iteration")
plt.ylabel("Sharpe ratio")
plt.show()

