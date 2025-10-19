import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plt_ticker
import mpl_toolkits.mplot3d as axe3d
import pandas as pd
import datetime
from scipy.stats import norm
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(page_title='BSM Model', page_icon="$", layout="wide")


def black_scholes(S, K, T, r, sigma, option_type):
    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(d1, T, sigma)
    
    if option_type == 'Call':
        return (S * norm.cdf(d1)) - ((K * np.exp(-r * T)) * (norm.cdf(d2)))
    elif option_type == 'Put':
        return (K * np.exp(-r * T)) * (norm.cdf(-d2)) - (S * norm.cdf(-d1))
    else:
        raise ValueError("Invalid Inputs")


def calculate_d1(S, K, T, r, sigma):
    S = np.array(S, dtype=np.float64)
    if np.any(T) == 0 or np.any(sigma) == 0 or np.any(K) == 0:
        raise ValueError("Time to maturity, volatility, and strike price cannot be zero")
    return ((np.log(S / K)) + (r + (sigma ** 2) * 0.5) * T) / (sigma * np.sqrt(T))


def calculate_d2(d1, T, sigma):
    if np.any(d1):
        return d1 - (sigma * np.sqrt(T))
    else:
        raise ValueError("d1 is missing")


def calculate_greeks(S, K, T, r, sigma, option_type):
    S = np.array(S, dtype=np.float64)
    T = np.array(T, dtype=np.float64)
    
    d1 = calculate_d1(S, K, T, r, sigma)
    d2 = calculate_d2(d1, T, sigma)
    
    # DELTA
    if option_type == 'Call':
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)
    
    # GAMMA
    gamma = (norm.pdf(d1) / (S * sigma * np.sqrt(T)))
    
    # VEGA
    vega = (norm.pdf(d1) * S * np.sqrt(T))
    
    # THETA
    if option_type == 'Call':
        theta = ((-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = ((-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    # RHO
    if option_type == 'Call':
        rho = (K * T * np.exp(-r * T) * norm.cdf(d2))
    else:
        rho = (-K * T * np.exp(-r * T) * norm.cdf(-d2))
    
    return {
        "Name": ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'],
        "Value": [f'{delta:.4f}', f'{gamma:.4f}', f'{vega:.4f}', f'{theta:.4f}', f'{rho:.4f}'],
        "Raw": [delta, gamma, vega, theta, rho]
    }


def create_pnl_heatmap(K, T, r, min_spot, max_spot, min_vol, max_vol, option_type, purchase_price):
    """Create a P&L heatmap for options"""
    # Create grid
    spot_range = np.linspace(min_spot, max_spot, 10)
    vol_range = np.linspace(min_vol, max_vol, 10)
    
    # Initialize P&L matrix
    pnl_matrix = np.zeros((len(vol_range), len(spot_range)))
    
    # Calculate P&L for each combination
    for i, vol in enumerate(vol_range):
        for j, spot in enumerate(spot_range):
            option_price = black_scholes(spot, K, T, r, vol, option_type)
            pnl_matrix[i, j] = option_price - purchase_price
    
    return pnl_matrix, spot_range, vol_range


def main():
    st.title("Black-Scholes-Merton Model for Option pricing")
    
    # User inputs
    with st.sidebar:
        st.text("Use desktop for best result")
        
        # Basic inputs
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.text_input("Enter stock ticker: ", value="AAPL")    
            start_date = st.date_input("Start date: ", value="today")    
            one_year_from_today = datetime.date.today()+datetime.timedelta(days = 365)
            end_date = st.date_input("Expiration date: ", value=one_year_from_today)
        
        time_to_maturity = float(((end_date - start_date).days) / 365)
        strike_price = float(st.number_input("Enter the strike price K: ", value=0.00))
        
        col3, col4 = st.columns(2)
        with col3:
            risk_free_rate = (float(st.number_input("Risk free rate as %: ", value=0.00)) / 100)
        with col4:
            volatility = (float(st.number_input("Volatility as %:", value=0.00)) / 100)
        
        # Heatmap parameters section
        st.markdown("---")
        st.subheader(" Heatmap Parameters")
        
        # Purchase prices
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            call_purchase_price = st.number_input("Call Purchase Price ($):", value=0.0, step=0.5,
                                                   help="Price you paid for the call option")
        with col_p2:
            put_purchase_price = st.number_input("Put Purchase Price ($):", value=0.0, step=0.5,
                                                  help="Price you paid for the put option")
        
        # Spot price range
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            min_spot = st.number_input("Min Spot Price ($):", value=50.0, step=10.0)
        with col_s2:
            max_spot = st.number_input("Max Spot Price ($):", value=200.0, step=10.0)
        
        # Volatility range (sliders)
        st.markdown("**Volatility Range:**")
        min_vol = st.slider("Min Volatility (%):", min_value=1, max_value=99, value=10, step=1) / 100
        max_vol = st.slider("Max Volatility (%):", min_value=int((min_vol * 100) + 1), max_value=100, value=80, step=1) / 100
        
        option_type = st.radio("Option Type for Greeks Chart:", ["Call", "Put"], horizontal=True)

        st.markdown("---")
        execute_op = st.button("Calculate", use_container_width=True)
    
    # Fetch data with yfinance
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period='1d')
    if stock_data.empty:
        st.error("Data related to ticker not found")
    else:
        latest_price = (stock_data["Close"].iloc[-1])
    
    if latest_price and execute_op:
        st.subheader("Summary of the inputs: ", divider="red")
        data = {
            "Name": ["Ticker", "Current price", "Strike Price", "Time till maturity", "Risk-free Rate", "Volatility"],
            "value": [ticker, f"{latest_price:.2f}", f"{strike_price:.2f}", f"{time_to_maturity:.4f}", f"{risk_free_rate:.4f}", f"{volatility:.4f}"]
        }
        df = pd.DataFrame(data=data, index=[1, 2, 3, 4, 5, 6])
        st.dataframe(df, hide_index=True)
        
        # # Find the price of the option
        # option_price = black_scholes(latest_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)
        # # Find the greeks
        # greeks = calculate_greeks(latest_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)

        # call_price = black_scholes(latest_price, strike_price, time_to_maturity,risk_free_rate, volatility, 'Call')
        # put_price = black_scholes(latest_price, strike_price, time_to_maturity,risk_free_rate, volatility, 'Put')


        # # Present data to user
        # st.subheader("Required output: ", divider="red")
        # col_1, col_2 = st.columns(2)
        # with col_1:
        #     st.metric(label="CALL Value", value=f"${call_price:.2f}", label_visibility="visible", border=True)
        # with col_2:
        #     st.metric(label="PUT Value", value=f"${put_price:.2f}", label_visibility="visible", border=True)

        #     st.metric(label="Delta: ", value=greeks["Value"][0], border=True)
        
        # col_3, col_4 = st.columns(2)
        # with col_3:
        #     st.metric(label="Gamma: ", value=greeks["Value"][1], border=True)
        # with col_4:
        #     st.metric(label="Vega: ", value=greeks["Value"][2], border=True)
        
        # col_5, col_6 = st.columns(2)
        # with col_5:
        #     st.metric(label="Theta: ", value=greeks["Value"][3], border=True)
        # with col_6:
        #     st.metric(label="Rho: ", value=greeks["Value"][4], border=True)
        # Calculate both Call and Put prices and Greeks
        call_price = black_scholes(latest_price, strike_price, time_to_maturity, risk_free_rate, volatility, 'Call')
        put_price = black_scholes(latest_price, strike_price, time_to_maturity, risk_free_rate, volatility, 'Put')
        call_greeks = calculate_greeks(latest_price, strike_price, time_to_maturity, risk_free_rate, volatility, 'Call')
        put_greeks = calculate_greeks(latest_price, strike_price, time_to_maturity, risk_free_rate, volatility, 'Put')

        # Present data to user
        st.subheader("Output: ", divider="red")

        # Option Prices
        col_1, col_2 = st.columns(2)
        with col_1:
            #st.metric(label="CALL Value", value=f"${call_price:.2f}", label_visibility="visible", border=True)
            st.markdown(
        f"""
        <div style='background-color: #90ee90; border-radius: 10px; padding: 20px 0; margin-bottom: 12px; text-align: center;'>
            <span style='font-size: 16px; color: #006400;'>CALL Value</span><br>
            <span style='font-size: 28px; font-weight: bold; color: #006400;'>${call_price:.2f}</span>
        </div>
        """, unsafe_allow_html=True,
    )
        with col_2:
            #st.metric(label="PUT Value", value=f"${put_price:.2f}", label_visibility="visible", border=True)
            st.markdown(
        f"""
        <div style='background-color: #ffcdd2; border-radius: 10px; padding: 20px 0; margin-bottom: 12px; text-align: center;'>
            <span style='font-size: 16px; color: #b71c1c;'>PUT Value</span><br>
            <span style='font-size: 28px; font-weight: bold; color: #b71c1c;'>${put_price:.2f}</span>
        </div>
        """, unsafe_allow_html=True,
    )

        st.markdown("---")

        # Greeks communs (Gamma et Vega sont identiques pour Call et Put)
        st.markdown("###  Greeks (Common)")
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.metric(label="Gamma (Γ)", value=call_greeks["Value"][1], border=True, 
                    help="Rate of change of Delta (same for Call & Put)")
        with col_g2:
            st.metric(label="Vega (V)", value=call_greeks["Value"][2], border=True,
                    help="Sensitivity to volatility (same for Call & Put)")

        st.markdown("---")

        # Greeks spécifiques à chaque option
        st.markdown("###  Call Greeks vs  Put Greeks")
        col_c1, col_c2, col_c3 = st.columns(3)

        with col_c1:
            st.markdown("**Delta (Δ)**")
            st.metric(label="Call Delta", value=call_greeks["Value"][0], border=True)
            st.metric(label="Put Delta", value=put_greeks["Value"][0], border=True)

        with col_c2:
            st.markdown("**Theta (Θ)**")
            st.metric(label="Call Theta", value=call_greeks["Value"][3], border=True)
            st.metric(label="Put Theta", value=put_greeks["Value"][3], border=True)

        with col_c3:
            st.markdown("**Rho (ρ)**")
            st.metric(label="Call Rho", value=call_greeks["Value"][4], border=True)
            st.metric(label="Put Rho", value=put_greeks["Value"][4], border=True)




        # HEATMAPS SECTION
        st.markdown("---")
        st.subheader(" Options Price - Interactive Heatmap", divider="red")
        st.markdown(
            """
            <div style="
                background-color: rgba(0, 51, 102, 0.35);
                border-radius: 8px;
                padding: 15px;
                margin-top: 10px;
                color: #ffffff;
                font-size: 15px;
                font-weight: 600;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
                ">
                Explore how options prices fluctuate with varying spot prices and volatility levels using interactive heatmaps parameters, all while maintaining a constant Strike Price.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if min_spot >= max_spot:
            st.error("⚠️ Min Spot Price must be less than Max Spot Price")
        elif min_vol >= max_vol:
            st.error("⚠️ Min Volatility must be less than Max Volatility")
        else:
            # --- Calcul des matrices ---
            call_pnl, spot_range, vol_range = create_pnl_heatmap(
                strike_price, time_to_maturity, risk_free_rate,
                min_spot, max_spot, min_vol, max_vol,
                'Call', call_purchase_price
            )
            put_pnl, spot_range_put, vol_range_put = create_pnl_heatmap(
                strike_price, time_to_maturity, risk_free_rate,
                min_spot, max_spot, min_vol, max_vol,
                'Put', put_purchase_price
            )

            # Harmonisation de l'échelle des couleurs
            common_vmin = min(call_pnl.min(), put_pnl.min())
            common_vmax = max(call_pnl.max(), put_pnl.max())

            # Création des labels pour les axes
            vol_labels = [f"{v*100:.0f}%" for v in vol_range]
            spot_labels = [f"${s:.0f}" for s in spot_range]

            # Création des colonnes d'affichage Streamlit
            col_call, col_put = st.columns(2)

            # === CALL HEATMAP INTERACTIVE ===
            with col_call:
                st.markdown("###  Call Price Heatmap")
                
                fig_call = go.Figure(data=go.Heatmap(
                    z=call_pnl,
                    x=spot_labels,
                    y=vol_labels,
                    colorscale='RdYlGn',
                    zmid=0,
                    zmin=common_vmin,
                    zmax=common_vmax,
                    colorbar=dict(
                        title=dict(text="P&L ($)", side="right"),
                        thickness=15,
                        len=0.7
                    ),
                    hovertemplate='Spot: %{x}<br>Volatility: %{y}<br>P&L: $%{z:.2f}<extra></extra>',
                    text=call_pnl,
                    texttemplate='<b>%{z:.2f}</b>',
                    textfont={"size": 11}
                ))
                
                fig_call.update_layout(
                    title=dict(
                        text='Call Option P&L',
                        font=dict(size=16, color="white"),
                        x=0.5,
                        xanchor='center'
                    ),
                    xaxis=dict(
                        title='Spot Price',
                        side='bottom',
                        tickfont=dict(size=10, color="white"),
                        title_font=dict(size=12, color="white")
                    ),
                    yaxis=dict(
                        title='Volatility',
                        tickfont=dict(size=10, color="white"),
                        title_font=dict(size=12, color="white")
                    ),
                    plot_bgcolor='#1a1a1a',
                    paper_bgcolor='#1a1a1a',
                    height=500,
                    margin=dict(l=80, r=80, t=60, b=60)
                )
                
                st.plotly_chart(fig_call, use_container_width=True)

            # === PUT HEATMAP INTERACTIVE ===
            with col_put:
                st.markdown("###  Put Price Heatmap")
                
                fig_put = go.Figure(data=go.Heatmap(
                    z=put_pnl,
                    x=spot_labels,
                    y=vol_labels,
                    colorscale='RdYlGn',
                    zmid=0,
                    zmin=common_vmin,
                    zmax=common_vmax,
                    colorbar=dict(
                        title=dict(text="P&L ($)", side="right"),
                        thickness=15,
                        len=0.7
                    ),
                    hovertemplate='Spot: %{x}<br>Volatility: %{y}<br>P&L: $%{z:.2f}<extra></extra>',
                    text=put_pnl,
                    texttemplate='<b>%{z:.2f}</b>',
                    textfont={"size": 11}
                ))
                
                fig_put.update_layout(
                    title=dict(
                        text='Put Option P&L',
                        font=dict(size=16, color="white"),
                        x=0.5,
                        xanchor='center'
                    ),
                    xaxis=dict(
                        title='Spot Price',
                        side='bottom',
                        tickfont=dict(size=10, color="white"),
                        title_font=dict(size=12, color="white")
                    ),
                    yaxis=dict(
                        title='Volatility',
                        tickfont=dict(size=10, color="white"),
                        title_font=dict(size=12, color="white")
                    ),
                    plot_bgcolor='#1a1a1a',
                    paper_bgcolor='#1a1a1a',
                    height=500,
                    margin=dict(l=80, r=80, t=60, b=60)
                )
                
                st.plotly_chart(fig_put, use_container_width=True)

                # Stats - Call
                max_profit_call = call_pnl.max()
                max_loss_call = call_pnl.min()
                current_pnl_call = black_scholes(latest_price, strike_price, time_to_maturity,
                                                risk_free_rate, volatility, 'Call') - call_purchase_price

                # Stats - Put
                max_profit_put = put_pnl.max()
                max_loss_put = put_pnl.min()
                current_pnl_put = black_scholes(latest_price, strike_price, time_to_maturity,
                                                risk_free_rate, volatility, 'Put') - put_purchase_price
                
                # Créer deux colonnes pour afficher côte à côte
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div style="
                    background-color: #1a1a1a;
                    border: 2px solid #003366;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 15px 0;
                    ">
                    <h3 style="
                        color: white;
                        margin-top: 0;
                        margin-bottom: 15px;
                        font-size: 20px;
                        font-weight: bold;
                        ">Call Stats:</h3>
                    <ul style="
                        color: white;
                        font-size: 15px;
                        line-height: 1.8;
                        list-style-type: none;
                        padding-left: 0;
                        margin: 0;
                        ">
                        <li style="margin-bottom: 8px;"><strong style="color: #00FF94;">Max Profit:</strong> <span style="color: white;">${max_profit_call:.2f}</span></li>
                        <li style="margin-bottom: 8px;"><strong style="color: #FF4757;">Max Loss:</strong> <span style="color: white;">${max_loss_call:.2f}</span></li>
                        <li style="margin-bottom: 0;"><strong style="color: #FFB800;">Current P&L:</strong> <span style="color: white;">${current_pnl_call:.2f}</span></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div style="
                    background-color: #1a1a1a;
                    border: 2px solid #003366;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 15px 0;
                    ">
                    <h3 style="
                        color: white;
                        margin-top: 0;
                        margin-bottom: 15px;
                        font-size: 20px;
                        font-weight: bold;
                        ">Put Stats:</h3>
                    <ul style="
                        color: white;
                        font-size: 15px;
                        line-height: 1.8;
                        list-style-type: none;
                        padding-left: 0;
                        margin: 0;
                        ">
                        <li style="margin-bottom: 8px;"><strong style="color: #00FF94;">Max Profit:</strong> <span style="color: white;">${max_profit_put:.2f}</span></li>
                        <li style="margin-bottom: 8px;"><strong style="color: #FF4757;">Max Loss:</strong> <span style="color: white;">${max_loss_put:.2f}</span></li>
                        <li style="margin-bottom: 0;"><strong style="color: #FFB800;">Current P&L:</strong> <span style="color: white;">${current_pnl_put:.2f}</span></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )


        
    # GREEKS CHARTS - VERSION INTERACTIVE
        st.markdown("---")
        st.subheader(" Greeks vs Stock price", divider="red")

        # Description des Greeks
        st.markdown(
            """
            <div style="
                background-color: rgba(0, 51, 102, 0.35);
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 15px;
                color: #ffffff;
                font-size: 15px;
                font-weight: 600;
                text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
                ">
                <strong>Greeks Definitions:</strong><br>
                • <strong>Delta</strong>: Rate of change of option price with respect to stock price<br>
                • <strong>Gamma</strong>: Rate of change of Delta with respect to stock price<br>
                • <strong>Vega</strong>: Sensitivity to volatility changes<br>
                • <strong>Theta</strong>: Time decay of the option (absolute value)<br>
                • <strong>Rho</strong>: Sensitivity to interest rate changes
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Use min_spot and max_spot for Price_range
        Price_range = np.linspace(min_spot, max_spot, 150)
            
        delta_values = [calculate_greeks(price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)["Raw"][0] for price in Price_range]
        gamma_values = [calculate_greeks(price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)["Raw"][1] for price in Price_range]
        vega_values = [calculate_greeks(price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)["Raw"][2] for price in Price_range]
        theta_values = [calculate_greeks(price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)["Raw"][3] for price in Price_range]
        rho_values = [calculate_greeks(price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)["Raw"][4] for price in Price_range]

        gamma_values = [g * 100 for g in gamma_values]
        vega_values = [v / 100 for v in vega_values]
        theta_values = [abs(t) * 100 for t in theta_values]
        rho_values = [r / 100 for r in rho_values]

        # Création du graphique interactif
        fig = go.Figure()

        # Configuration des couleurs et styles (couleurs plus vives pour fond noir)
        greeks_config = [
            {'name': 'Delta', 'values': delta_values, 'color': '#00D9FF', 'dash': 'solid'},
            {'name': 'Gamma', 'values': gamma_values, 'color': '#00FF94', 'dash': 'solid'},
            {'name': 'Vega', 'values': vega_values, 'color': '#FFB800', 'dash': 'solid'},
            {'name': 'Theta', 'values': theta_values, 'color': '#FF4757', 'dash': 'solid'},
            {'name': 'Rho', 'values': rho_values, 'color': '#C56EFF', 'dash': 'solid'}
        ]

        # Ajout des courbes pour chaque Greek
        for greek in greeks_config:
            fig.add_trace(go.Scatter(
                x=Price_range,
                y=greek['values'],
                name=greek['name'],
                mode='lines',
                line=dict(color=greek['color'], width=3, dash=greek['dash']),
                hovertemplate=f'<b>{greek["name"]}</b><br>Stock Price: %{{x:.2f}}<br>Value: %{{y:.4f}}<extra></extra>'
            ))

        # Ligne verticale pour le Strike Price
        fig.add_vline(
            x=strike_price, 
            line_dash="dash", 
            line_color="rgba(255, 255, 255, 0.5)",
            line_width=2,
            annotation=dict(
                text=f"Strike: ${strike_price}",
                font=dict(size=12, color="white"),
                bgcolor="rgba(0, 0, 0, 0.6)",
                bordercolor="white",
                borderwidth=1
            )
        )

        # Mise en forme du graphique avec fond noir
        fig.update_layout(
            title=dict(
                text='Option Greeks Sensitivity',
                font=dict(size=20, color="white", family="Arial"),
                x=0.5,
                xanchor='center'
            ),
            xaxis=dict(
                title=dict(
                    text='Stock Price ($)',
                    font=dict(size=16, color="white", family="Arial")
                ),
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.15)',
                gridwidth=1,
                zeroline=True,
                zerolinecolor='rgba(255, 255, 255, 0.3)',
                zerolinewidth=2,
                tickfont=dict(size=12, color="white"),
                tickformat='.2f'
            ),
            yaxis=dict(
                title=dict(
                    text='Greeks Values (normalized)',
                    font=dict(size=16, color="white", family="Arial")
                ),
                showgrid=True,
                gridcolor='rgba(255, 255, 255, 0.15)',
                gridwidth=1,
                zeroline=True,
                zerolinecolor='rgba(255, 255, 255, 0.3)',
                zerolinewidth=2,
                tickfont=dict(size=12, color="white"),
                tickformat='.3f'
            ),
            hovermode='x unified',
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            height=650,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=1.15,
                xanchor="left",
                x=0,
                bgcolor="rgba(26, 26, 26, 0.9)",
                bordercolor="white",
                borderwidth=2,
                font=dict(size=13, color="white")
            ),
            margin=dict(l=80, r=40, t=100, b=80)
        )

        # Affichage
        st.plotly_chart(fig, use_container_width=True)
        
        # 3D PLOT INTERACTIF
        st.markdown("---")
        st.subheader(" 3D plot: Option price vs Stock Price over time", divider="red")

        Time_value = np.linspace(0.01, time_to_maturity, 200)
        Price_grid, Time_grid = np.meshgrid(Price_range, Time_value)
        chart_output = black_scholes(Price_grid, strike_price, Time_grid, risk_free_rate, volatility, option_type)

        # Création du graphique 3D interactif avec Plotly
        fig = go.Figure(data=[go.Surface(
            x=Price_grid,
            y=Time_grid,
            z=chart_output,
            colorscale='Viridis',
            colorbar=dict(
                title=dict(
                    text="Option Price",
                    side="right"
                ),
                thickness=20,
                len=0.7,
                x=1.02
            ),
            hovertemplate='Stock Price: %{x:.2f}<br>Time to Maturity: %{y:.3f}<br>Option Price: %{z:.2f}<extra></extra>'
        )])

        # Mise en forme du graphique
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title='Stock Price',
                    backgroundcolor="white",
                    gridcolor="lightgray",
                    showbackground=True
                ),
                yaxis=dict(
                    title='Time to Maturity (years)',
                    backgroundcolor="white",
                    gridcolor="lightgray",
                    showbackground=True
                ),
                zaxis=dict(
                    title='Option Price',
                    backgroundcolor="white",
                    gridcolor="lightgray",
                    showbackground=True
                ),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            autosize=True,
            height=700,
            margin=dict(l=0, r=0, b=0, t=40)
        )

        # Affichage dans Streamlit
        st.plotly_chart(fig, use_container_width=True)


#END SECTION
        st.subheader("About the App:", divider="red")
    st.text(" ")
    st.markdown("""This is an option pricing calculator based on the Black-Scholes model.
It allows you to:
                 
                Compute European call and put option prices
                Analyze key Greeks:

Delta (Δ): Measures how much the option price changes with a $1 change in the underlying asset.

Gamma (Γ): Measures how much Delta changes with a $1 change in the underlying asset.

Vega (V): Measures how much the option price changes with a 1 percent change in implied volatility.

Theta (θ): Measures how much the option price decreases per day as expiration approaches.

Rho (ρ): Measures how much the option price changes with a 1 percent change in the risk-free interest rate.

                Visualize how these values evolve with changes in stock price and time to maturity"""
    )
    st.text(" ")
    st.subheader("Regarding the inputs:", divider="red")
    col3, col4 = st.columns(2)
    with col3:
        st.text(" ")
        st.markdown("""Uses yfinance API to get the last equity market price """)
    with col4:
        st.text(" ")
        st.markdown("Please ensure that every input is valid")


if __name__ == '__main__':
    main()