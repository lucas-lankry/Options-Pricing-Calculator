import streamlit as st
import yfinance as yf 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.ticker as plt_ticker
import mpl_toolkits.mplot3d as axe3d
import pandas as pd 
from scipy.stats import norm

st.set_page_config(page_title = 'BSM Model', page_icon = " $ ",layout = "wide")

def black_scholes(S,K,T,r,sigma,option_type):
    d1 = calculate_d1(S,K,T,r,sigma)
    d2 = calculate_d2(d1, T, sigma)

    if option_type == 'Call':
        return((S*norm.cdf(d1))-((K*np.exp(-r*T))*(norm.cdf(d2))))
    elif option_type == 'Put':
        return((K*np.exp(-r*T))*(norm.cdf(-d2)))-(S*norm.cdf(-d1))
    else:
        raise ValueError("Invalid Inputs")
    
def calculate_d1(S, K, T, r, sigma):
    S = np.array(S, dtype = np.float64)
    if np.any(T)== 0 or np.any(sigma)== 0 or np.any(K)== 0:
        raise ValueError("Time to maturity, volatility, and strike price cannot be zero")
    return ((np.log(S/K))+(r+(sigma**2)*0.5)*T)/(sigma*np.sqrt(T))

def calculate_d2(d1, T, sigma):
    if np.any(d1):
        return d1 - (sigma*np.sqrt(T))
    else: 
        raise ValueError("d1 is missing")
    
def calculate_greeks(S, K, T, r, sigma, option_type):
    S = np.array(S, dtype= np.float64)
    T = np.array(T, dtype= np.float64)

    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(d1, T, sigma)

    #DELTA
    if option_type == 'Call':
        delta = norm.cdf(d1)
    else: 
        delta = -norm.cdf(-d1)

    #GAMMA
    gamma = (norm.pdf(d1)/(S*sigma*np.sqrt(T)))

    #VEGA
    vega = (norm.pdf(d1)*S*np.sqrt(T))

    #THETA
    if option_type == 'Call':
        theta = ((-S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*norm.cdf(d2))/365
    else:
        theta = ((-S*norm.pdf(d1)*sigma)/(2*np.sqrt(T)) + r*K*np.exp(-r*T)*norm.cdf(-d2))/365

    #RHO
    if option_type == 'Call':
        rho = (K*T*np.exp(-r*T)*norm.cdf(d2))
    else: 
        rho = (-K*T*np.exp(-r*T)*norm.cdf(-d2))

    return {"Name": ['Delta', 'Gamma','Vega','Theta','Rho'], "Value":[f'{delta:.4f}', f'{gamma:.4f}', f'{vega:.4f}', f'{theta:.4f}', f'{rho:.4f}'], "Raw": [delta, gamma, vega, theta, rho]}

#setup for page
def main():

    st.title("Black-Scholes-Merton Model for Option pricing")
    st.subheader("About the App:", divider = "rainbow")
    st.text(" ")
    st.markdown("""This is a simple yet powerful option pricing calculator based on the Black-Scholes model.
It allows users to: Compute European call and put option prices
                    Analyze key Greeks:

Delta (Δ): Measures how much the option price changes with a $1 change in the underlying asset.

Gamma (Γ): Measures how much Delta changes with a $1 change in the underlying asset.

Vega (V): Measures how much the option price changes with a 1 percent change in implied volatility.

Theta (θ): Measures how much the option price decreases per day as expiration approaches.

Rho (ρ): Measures how much the option price changes with a 1 percent change in the risk-free interest rate.

Visualize how these values evolve with changes in stock price and time to maturity"""

    )
    st.text(" ")
    st.subheader("Regarding the inputs:", divider = "rainbow")
    col3, col4 = st.columns(2)
    with col3:
        st.text(" ")
        st.markdown("""Uses yfinance API to get the last equity market price """)
    with col4:
        st.text(" ")
        st.markdown("please ensure that every input is valid")

#user inputs
with st.sidebar: 
    st.text("Use desktop for best result")
    col1, col2 = st.columns(2)
    with col1: 
        ticker = st.text_input("Enter stock ticker: ", value="AAPL")
        start_date= st.date_input("Start date: ", value = "today")
    with col2: 
        option_type = st.selectbox("option type: ", ("Call","Put"))
        end_date = st.date_input("Expiration date: ", value = "today")
    time_to_maturity = float(((end_date-start_date).days)/365)
    strike_price = float(st.number_input("Enter the strike price K: ", value = 0.00))
    col3, col4 = st.columns(2)
    with col3: 
        risk_free_rate = (float(st.number_input("Risk free rate as %: ", value = 0.00))/100)
    with col4:
        volatility = (float(st.number_input("Volatility as %:", value = 0.00))/100)
    slider = float(st.slider("Range of Stock Price as %: ", step = 5)/100)
    execute_op = st.button("Calculate",use_container_width = True)



#fetch data with yfinance
stock = yf.Ticker(ticker)
stock_data = stock.history(period = '1d')
if stock_data.empty:
    st.error("Data related to ticker not found")
else : 
    latest_price = (stock_data["Close"].iloc[-1])

if latest_price and execute_op :
    st.subheader("Summary of the inputs: ", divider = "rainbow")
    data = {"Name" : ["Ticker","Current price","Strike Price","Time till maturity", "risk free rate", "volatility", "option type", "range of stock price as %"],
            "value":[ticker, f"{latest_price:.2f}",f"{strike_price:.2f}",f"{time_to_maturity:.4f}",f"{risk_free_rate:.4f}",f"{volatility:.4f}",option_type, slider*100]}
    df =pd.DataFrame(data = data, index = [1,2,3,4,5,6,7,8])
    st.dataframe(df, hide_index = True)

#find the price of the option
    option_price = black_scholes(latest_price, strike_price, time_to_maturity,risk_free_rate,volatility,option_type)
#find the greeks
    greeks = calculate_greeks(latest_price, strike_price, time_to_maturity,risk_free_rate,volatility,option_type)

#present data to user
    st.subheader("Required output: ", divider = "rainbow")
    col_1, col_2 = st.columns(2)
    with col_1:
        st.metric(label  =f"{option_type} option price is: ", value = f"{option_price:.2f}", label_visibility = "visible", border = True)
    with col_2:
        st.metric(label = "Delta: ", value = greeks["Value"][0], border = True)

    col_3, col_4 = st.columns(2)
    with col_3:
        st.metric(label = "Gamma: ", value = greeks["Value"][1], border = True)
    with col_4:
        st.metric(label = "Vega: ", value = greeks["Value"][2], border = True)

    col_5, col_6 = st.columns(2)
    with col_5:
        st.metric(label = "Theta: ", value = greeks["Value"][3], border = True)
    with col_6:
        st.metric(label = "Rho: ", value = greeks["Value"][4], border = True)

    if slider: 
        st.subheader("$ Greeks vs Stock price", divider = "rainbow")
        Price_range = np.linspace(latest_price*(1-slider), latest_price*(1+slider), 150)
        # ✅ CORRECT - utilise price (variable de la boucle) + convertit en float
    
        delta_values = [calculate_greeks(price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)["Raw"][0] for price in Price_range]
        gamma_values = [calculate_greeks(price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)["Raw"][1] for price in Price_range]
        vega_values = [calculate_greeks(price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)["Raw"][2] for price in Price_range]
        theta_values = [calculate_greeks(price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)["Raw"][3] for price in Price_range]
        rho_values = [calculate_greeks(price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)["Raw"][4] for price in Price_range]
        
        # ✅ NORMALISATION pour voir toutes les courbes
        gamma_values = [g * 100 for g in gamma_values]  # Multiplie gamma par 100
        vega_values = [v / 100 for v in vega_values]    # Divise vega par 100
        theta_values = [abs(t) * 100 for t in theta_values]  # Valeur absolue de theta * 100
        rho_values = [r / 100 for r in rho_values]      # Divise rho par 100
        
        plt.style.use("seaborn-v0_8-darkgrid")
        figure, axis = plt.subplots(figsize = (10,6))
        axis.plot(Price_range, delta_values, label = 'Delta', color = '#1f77b4', lw = 2, alpha = 0.9)
        axis.plot(Price_range, gamma_values, label = 'Gamma', color = '#2ca02c', lw = 2, alpha = 0.9)
        axis.plot(Price_range, vega_values, label = 'Vega', color = '#ff7f0e', lw = 2, alpha = 0.9)
        axis.plot(Price_range, theta_values, label = 'Theta', color = '#d62728', lw = 2, alpha = 0.9)
        axis.plot(Price_range, rho_values, label = 'Rho', color = '#9467bd', lw = 2, alpha = 0.9)


        axis.axvline(x = strike_price, color = "black", linestyle = "-.", lw = 1, alpha = 0.1)
        axis.yaxis.set_major_locator(plt_ticker.AutoLocator())
        axis.set_facecolor("white")

        axis.set_xlabel("Stock Price", fontsize = 12, fontweight = "bold")
        axis.set_ylabel("Greeks as %", fontsize = 12, fontweight = "bold")
        axis.set_title("Option Greeks sensitivity", fontsize = 14, pad = 20, fontweight = "bold")
        axis.legend( frameon = True, framealpha = 0.9, shadow = False, facecolor = "white" )

        axis.grid(True, alpha = 0.3, linestyle = "--", color ="gray" )
        for spine in ["top", "right"]:
            axis.spines[spine].set_visible(True)
            axis.spines[spine].set_color("#333333")
        
        axis.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))

        st.pyplot(figure)

        #3d. PLOT

        st.subheader("3D plot: Option price vs Stock Price over time", divider = "rainbow")
        Time_value = np.linspace(0.01,time_to_maturity,200)
        Price_grid, Time_grid = np.meshgrid(Price_range, Time_value)
        chart_output = black_scholes(Price_grid, strike_price, Time_grid, risk_free_rate, volatility, option_type)

        figure3d = plt.figure(figsize = (10,8), dpi = 100)
        axis3d = figure3d.add_subplot(111, projection="3d")
        surface_plot = axis3d.plot_surface(Price_grid, Time_grid, chart_output, cmap= "viridis", edgecolor = "none", alpha = 0.9, rstride = 1, cstride = 1,antialiased = True, shade = True)
        
        cbar = figure3d.colorbar(surface_plot, shrink = 0.6,aspect = 20 , location = 'left', pad = 0.05)
        cbar.ax.yaxis.set_label_position("right")
        cbar.set_label("Option price", fontsize = 12, fontweight = "bold", rotation = 90)

        axis3d.set_facecolor("white")
        axis3d.view_init(elev = 25, azim = 45)

        axis3d.set_xlabel("Stock Price", fontsize = 12, fontweight = "bold")
        axis3d.set_ylabel("Time till maturity (in years)", fontsize = 12, fontweight = "bold")
        axis3d.set_title("3D plot : Option price vs Stock Price over time", fontsize = 14, fontweight = "bold")
        st.pyplot(figure3d)
    
    else:
        st.text(" ")
        st.warning("It looks like the app isnt running at the moment")

if __name__ == '__main__':
    main()











