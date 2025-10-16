import vectorbt as vbt 
from bs4 import BeautifulSoup
import pandas as pd 
import numpy as np 
import requests 
import os 
from io import StringIO


#SCRAPING
def fetch_info():
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:101.0) Gecko/20100101 Firefox/101.0',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.5',
        }
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        # Trouver la première table ayant un header "Symbol"
        tables = soup.find_all('table', {'class': 'wikitable'})
         #  Loop over all tables to find the one with "Symbol" column
        for t in tables:
            df = pd.read_html(str(t))[0]
            if 'Symbol' in df.columns and 'Company' in df.columns:
                if 'Notes' in df.columns:
                    df = df.drop(columns=['Notes'])
                return df
        print('Table with Symbol not found')
        return None
    except Exception as e:
        print('Error loading data')
        return None

# Utilisation
dji_df = fetch_info()
#print(dji_df.head())

tickers = dji_df.Symbol.values.tolist()

df= vbt.YFData.download(tickers, start= "2021-01-01", end = "2022-09-01", missing_index='drop').get('Close')

mth_return_df = df.pct_change().resample("M").agg(lambda x: (x+1).prod()-1)
import pytz
# obtain the historical cumulative returns of past 6 months as the terminal return of current month
past_cum_return_df = (mth_return_df+1).rolling(6).apply(np.prod) - 1
past_cum_return_df.index = past_cum_return_df.index.tz_localize(None)

import datetime as dt
end_of_measurement_period = dt.datetime(2022,6,30)
formation_period = dt.datetime(2022,7,31)

end_of_measurement_period_return_df = past_cum_return_df.loc[end_of_measurement_period]
end_of_measurement_period_return_df = end_of_measurement_period_return_df.reset_index()

#print(end_of_measurement_period_return_df.head())

# highest momentum in the positive direction
#end_of_measurement_period_return_df.loc[end_of_measurement_period_return_df.iloc[:,1].idxmax()]


end_of_measurement_period_return_df['rank'] = pd.qcut(end_of_measurement_period_return_df.iloc[:,1], 5, labels=False)

long_stocks = end_of_measurement_period_return_df.loc[end_of_measurement_period_return_df["rank"]==4,"symbol"].values
short_stocks = end_of_measurement_period_return_df.loc[end_of_measurement_period_return_df["rank"]==0,"symbol"].values

print(long_stocks)
