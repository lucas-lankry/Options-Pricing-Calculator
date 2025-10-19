# Black-Scholes Options Pricing Calculator

Interactive web application for options pricing and Greeks analysis using the Black-Scholes model. Built with Python and Streamlit for real-time market analysis.

## Overview

This tool provides comprehensive options pricing capabilities including:

- European call and put option valuation
- Real-time Greeks calculation (Delta, Gamma, Vega, Theta, Rho)
- 3D visualization of option price surfaces
- Live market data integration via Yahoo Finance API
- Interactive parameter sensitivity analysis

## Features

### Core Functionality

- **Black-Scholes Pricing:** Analytical solution for European options
- **Greeks Analysis:** Complete risk sensitivity metrics
- **Market Data Integration:** Real-time stock prices and volatility
- **Visual Analytics:** 3D surface plots for price and Greeks visualization
- **Parameter Exploration:** Interactive sliders for scenario analysis

## Visualizations

- Summary of the inputs
- Outputs
- Greeks vs Stock price: Showing sensitivity across multiple dimensions
- 3D plot: Option price vs Stock price over time

---

## Modes of Operation

**1. Manual Input Mode:**
- Enter custom parameters (spot price, strike, volatility, etc.)
- Adjust sliders to explore sensitivity
- View calculated option prices and Greeks

**2. Live Market Data Mode:**
- Input stock ticker symbol
- Fetch real-time market data
- Analyze current options pricing

---

## Black-Scholes Formula

The application implements the analytical Black-Scholes formula:

**Call Option:**  
\( C = S_0N(d_1) - Ke^{-rT}N(d_2) \)

**Put Option:**  
\( P = Ke^{-rT}N(-d_2) - S_0N(-d_1) \)

Where:  
\( d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}} \)  
\( d_2 = d_1 - \sigma\sqrt{T} \)

**Greeks Calculations:**
- Delta (\( \Delta \)): \( \partial V / \partial S \)
- Gamma (\( \Gamma \)): \( \partial^2 V / \partial S^2 \)
- Vega (\( \nu \)): \( \partial V / \partial \sigma \)
- Theta (\( \Theta \)): \( \partial V / \partial T \)
- Rho (\( \rho \)): \( \partial V / \partial r \)

---

## Technical Implementation

**Language:** Python 3.9+  
**Key Libraries:**
- `streamlit`: Interactive web interface
- `numpy`: Numerical computations
- `scipy.stats`: Statistical distributions for Black-Scholes
- `matplotlib`: Data visualization
- `yfinance`: Market data retrieval
- `pandas`: Data manipulation

**Mathematical Model:** Black-Scholes-Merton framework for European options

---

## Installation

Clone repository
git clone https://github.com/lucas-lankry/Options-Pricing-Calculator.git
cd Options-Pricing-Calculator

Install dependencies
pip install -r requirements.txt

Run application
streamlit run app.py


---

## Usage

(Indique ici toute instruction d’utilisation supplémentaire, exemples, etc.)

---

## Limitations & Future Enhancements

**Current Limitations:**
- European options only (no early exercise)
- Constant volatility assumption
- Single underlying asset

**Planned Enhancements:**
- American options pricing (binomial tree method)
- Implied volatility calculation
- Options strategy builder (spreads, straddles, etc.)
- Historical backtesting capabilities
- Monte Carlo simulation for exotic options

---

## Author

**Lucas Lankry**  
MSc Financial Markets & Investments | NC State – SKEMA  
CFA Level 3 Candidate  
[LinkedIn](https://linkedin.com/in/lucaslankry) | [GitHub](https://github.com/lucas-lankry) | [llankry@ncsu.edu](mailto:llankry@ncsu.edu)

---

## Acknowledgments

- Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"
- Hull, J. (2018). "Options, Futures, and Other Derivatives"
