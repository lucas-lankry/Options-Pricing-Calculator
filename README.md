# Black-Scholes Options Pricing Calculator

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-FF4B4B.svg)](https://streamlit.io)
[![NumPy](https://img.shields.io/badge/NumPy-1.21+-013243.svg)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.7+-8CAAE6.svg)](https://scipy.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.0+-3F4F75.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Web application for European options pricing, risk analysis, and portfolio Greeks calculation using the Black-Scholes-Merton framework**

Built with Python and Streamlit !

## 🚀 Live Demo
Try the application: [https://blackscholespy-ztryxbewvkzb8glghaekwt.streamlit.app/]

No installation required - works in any browser!



---

## Overview

This application implements the classical Black-Scholes-Merton model with advanced visualization and risk management capabilities. It provides real-time calculation of option prices and complete Greeks sensitivity analysis with interactive parameter exploration.

<img width="1901" height="944" alt="image" src="https://github.com/user-attachments/assets/aec01670-2cbf-4f54-874e-4f56cda08b4e" />

---

**Core Capabilities:**
- European call and put option valuation using analytical solutions
  
  <img width="1824" height="890" alt="image" src="https://github.com/user-attachments/assets/08e33312-468e-48f1-915d-0a46bddfe159" />
  
- Real-time Greeks calculation (Delta, Gamma, Vega, Theta, Rho)
  
<img width="1377" height="242" alt="image" src="https://github.com/user-attachments/assets/91dc5160-c1cf-48e5-b54c-9396ea3b635e" />
<img width="1375" height="659" alt="image" src="https://github.com/user-attachments/assets/ad11ef64-d0d1-4ec9-a991-143a7ccdf62f" />

- Interactive 3D surface plots for price evolution analysis
  
  <img width="1409" height="778" alt="image" src="https://github.com/user-attachments/assets/3dffe2f0-230c-499e-89de-f35bd175e093" />
  
- Live market data integration via Yahoo Finance API
- Profit/Loss heatmap generation across volatility and spot price ranges
  
  <img width="1387" height="700" alt="image" src="https://github.com/user-attachments/assets/ac8a9841-bf07-420d-be46-d4427e463158" />

- Comprehensive sensitivity analysis with parameter sliders
  
<img width="360" height="673" alt="image" src="https://github.com/user-attachments/assets/b1187121-cf8d-4b62-b1f8-760341fe6f82" /> <img width="525" height="622" alt="image" src="https://github.com/user-attachments/assets/e4f1cd38-2d40-4e16-ad91-f246d2251a14" />

<img width="453" height="591" alt="image" src="https://github.com/user-attachments/assets/3fcaf144-e0e6-4f82-b8d6-f821a1512ce3" /> <img width="452" height="623" alt="image" src="https://github.com/user-attachments/assets/a15e63b4-37ee-4c13-9bfe-b9580f78f9e0" />

---

## Features

### Analytical Pricing Engine
- Implementation of Black-Scholes-Merton closed-form solution
- Simultaneous call and put valuation
- Support for custom parameter inputs and market data feeds

### Risk Analytics Dashboard
- **Delta (Δ)**: First-order price sensitivity to underlying asset movements
- **Gamma (Γ)**: Second-order sensitivity measuring Delta convexity
- **Vega (ν)**: Sensitivity to implied volatility changes
- **Theta (Θ)**: Time decay measurement (per day)
- **Rho (ρ)**: Interest rate risk exposure

### Advanced Visualizations
- **Interactive 3D Surface Plots**: Option price evolution across spot price and time dimensions using Plotly
- **Greeks Sensitivity Charts**: Multi-line plots showing all Greeks across stock price ranges
- **P&L Heatmaps**: Profit/loss visualization across spot price and volatility grids
- **Real-time Updates**: Dynamic recalculation with parameter adjustments

### Data Integration
- Yahoo Finance API integration for live market data
- Automatic historical volatility calculation
- Support for any publicly traded equity ticker
- Real-time price fetching and validation

---

## Technical Implementation

### Architecture
**Language**: Python 3.9+

**Core Libraries:**
- `streamlit`: Interactive web framework for financial applications
- `numpy`: High-performance numerical computing and array operations
- `scipy.stats`: Statistical distributions for Black-Scholes calculations
- `plotly`: Interactive 3D visualizations and responsive charts
- `yfinance`: Real-time market data retrieval
- `pandas`: Data manipulation and time series analysis
- `matplotlib`: Additional plotting capabilities
- `seaborn`: Statistical data visualization

---

## Installation and Deployment

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Internet connection for market data access

### Setup Instructions
```bash
# Clone the repository
git clone https://github.com/lucas-lankry/Options-Pricing-Calculator.git
cd Options-Pricing-Calculator

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run BlackScholes.py
```

The application will launch in your default browser at `http://localhost:8501`

### Requirements
```
streamlit>=1.28.0
yfinance>=0.2.28
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
pandas>=2.0.0
plotly>=5.17.0
seaborn>=0.12.0
```

---

## Usage Guide

### Input Configuration

**Market Data Mode:**
1. Enter stock ticker symbol (e.g., AAPL, MSFT, TSLA)
2. Specify start date and expiration date
3. Input strike price
4. Set risk-free rate (as percentage)
5. Input volatility estimate (as percentage)
6. Click "Calculate" to generate analysis

**Parameter Inputs:**
- **Spot Price (S₀)**: Retrieved automatically via yfinance API
- **Strike Price (K)**: User-defined exercise price
- **Time to Maturity (T)**: Calculated from date inputs (in years)
- **Risk-Free Rate (r)**: Typically based on Treasury rates
- **Volatility (σ)**: Implied or historical volatility (annualized)

### Output Analysis

**Options Pricing:**
- Call option premium calculated using Black-Scholes formula
- Put option premium calculated using put-call parity relationship
- Both displayed with clear visual distinction

**Greeks Analysis:**
- Common Greeks (Gamma, Vega) displayed together
- Call-specific Greeks (Delta, Theta, Rho) vs Put-specific Greeks
- All values formatted to 4 decimal places for precision

**Interactive Visualizations:**
1. **P&L Heatmaps**: Adjust spot price and volatility ranges to explore profit/loss scenarios
2. **Greeks Sensitivity Chart**: Observe how all Greeks evolve across stock price ranges
3. **3D Surface Plot**: Visualize option price evolution over time and spot price simultaneously

---

## Mathematical Framework

The application implements the Black-Scholes-Merton model with the following formulations:

### Pricing Formulas

**Call Option:**  
$$C = S_0N(d_1) - Ke^{-rT}N(d_2)$$

**Put Option:**  
$$P = Ke^{-rT}N(-d_2) - S_0N(-d_1)$$

**Where:**  
$$d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}$$

$$d_2 = d_1 - \sigma\sqrt{T}$$

**Notation:**
- $S_0$ = Current spot price
- $K$ = Strike price
- $T$ = Time to maturity (years)
- $r$ = Risk-free interest rate
- $\sigma$ = Volatility (standard deviation of returns)
- $N(x)$ = Cumulative standard normal distribution function

### Greeks Formulas

**Delta:**
- Call: $\Delta_C = N(d_1)$
- Put: $\Delta_P = N(d_1) - 1$

**Gamma (identical for calls and puts):**
$$\Gamma = \frac{N'(d_1)}{S_0\sigma\sqrt{T}}$$

**Vega (identical for calls and puts):**
$$\nu = S_0N'(d_1)\sqrt{T}$$

**Theta:**
- Call: $\Theta_C = -\frac{S_0N'(d_1)\sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2)$
- Put: $\Theta_P = -\frac{S_0N'(d_1)\sigma}{2\sqrt{T}} + rKe^{-rT}N(-d_2)$

**Rho:**
- Call: $\rho_C = KTe^{-rT}N(d_2)$
- Put: $\rho_P = -KTe^{-rT}N(-d_2)$

**Note:** $N'(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}$ represents the standard normal probability density function

---

## Model Assumptions and Limitations

### Black-Scholes-Merton Assumptions
1. European-style options (exercise only at maturity)
2. No transaction costs or taxes
3. Risk-free rate is constant and known
4. Volatility (σ) is constant over the option's life
5. Log-normal distribution of asset prices
6. No dividends paid during option life
7. Markets are efficient (no arbitrage opportunities)
8. Continuous trading with infinite divisibility

### Current Limitations
- No support for American-style options (early exercise feature)
- Constant volatility assumption (no stochastic volatility models)
- Single underlying asset only (no multi-asset or basket options)
- No dividend adjustment mechanism
- Basic historical volatility estimation only

### Known Considerations
- Implied volatility surface not implemented
- Model performs best for at-the-money options with moderate time to expiration
- Extreme deep in-the-money or out-of-the-money options may show pricing discrepancies
- Short-dated options subject to greater pricing sensitivity

---

## Development Roadmap

### Planned Enhancements

**Phase 1: Advanced Pricing Models**
- American options pricing using binomial/trinomial tree methods
- Dividend adjustment capabilities for dividend-paying stocks
- Jump-diffusion models (Merton model)

**Phase 2: Volatility Analysis**
- Implied volatility calculator using Newton-Raphson method
- Volatility smile and term structure visualization
- Historical vs implied volatility comparison tools

**Phase 3: Portfolio Analytics**
- Multi-leg options strategy builder (spreads, straddles, butterflies, condors, iron condors)
- Portfolio-level Greeks aggregation
- Risk/reward profile visualization for complex strategies
- Maximum profit/loss calculations

**Phase 4: Backtesting and Simulation**
- Historical backtesting framework with transaction costs
- Monte Carlo simulation for exotic options
- Path-dependent options pricing (Asian, barrier, lookback)
- Scenario analysis and stress testing

**Phase 5: Data Export and Reporting**
- CSV/Excel export functionality
- PDF report generation
- API endpoint creation for programmatic access
- Integration with Bloomberg/Reuters data feeds

---

## Application Structure
```
Options-Pricing-Calculator/
│
├── BlackScholes.py      # Main application file
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── LICENSE                    # MIT License
```

---

## Performance Specifications

- Real-time calculation: < 100ms for single option
- Heatmap generation: 10x10 grid in < 500ms
- 3D surface plot: 200x200 mesh in < 1 second
- Greeks chart: 150 data points per Greek in < 200ms

**Optimization Features:**
- Vectorized NumPy operations for bulk calculations
- Efficient memory management for large grids

---

## Author

**Lucas Lankry**  
MSc Financial Markets & Investments | NC State University – SKEMA Business School  
CFA Level III Candidate

**Research Interests:** Quantitative Finance, Derivative Pricing, Risk Management, Financial Engineering

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Professional_Profile-0077B5?style=flat&logo=linkedin)](https://linkedin.com/in/lucaslankry)
[![GitHub](https://img.shields.io/badge/GitHub-Portfolio-181717?style=flat&logo=github)](https://github.com/lucas-lankry)
[![Email](https://img.shields.io/badge/Email-Academic_Contact-D14836?style=flat&logo=gmail)](mailto:llankry@ncsu.edu)

---

## References

1. Black, F., & Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637-654.

2. Merton, R. C. (1973). "Theory of Rational Option Pricing." *Bell Journal of Economics and Management Science*, 4(1), 141-183.

3. Hull, J. C. (2018). *Options, Futures, and Other Derivatives* (10th ed.). Pearson.

4. Wilmott, P. (2006). *Paul Wilmott on Quantitative Finance* (2nd ed.). Wiley.



