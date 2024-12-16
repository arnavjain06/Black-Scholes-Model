import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import matplotlib.pyplot as plt 
import seaborn as sns

import math
from scipy.stats import norm


#BS calculation functions
def d1 (S, K, r, t, vol):
    d1 = ((math.log(S/K) + (t * (r + 0.5 * vol**2))) / (vol * math.sqrt(t)))
    return d1

def d2 (d1, vol, t):
    d2 = d1 - vol * math.sqrt(t)
    return d2

def call_price(S, d1, K, r, t, d2):
    C = S * norm.cdf(d1) - K * math.exp(-r * t) * norm.cdf(d2)
    return C

def put_price(S, d1, K, r, t, d2):
    P = K * math.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1) 
    return P


#Title of the application
st.title("Black-Scholes Model")

#Sidebar for User Input
st.sidebar.header("Option Parameters")

#Inputs 
S = st.sidebar.number_input(
    "Current Price (S)", 
    min_value=0.0, 
    value=100.0,
    step=0.01,
    format="%.2f",
)

K = st.sidebar.number_input(
    "Strike Price (K)", 
    min_value=0.0, 
    value=90.0,
    step=0.01,
    format="%.2f",
)

r = st.sidebar.number_input(
    "Risk-Free Rate (%)", 
    min_value=0.0, 
    max_value=100.0, 
    value=5.0,
    step=0.01,
    format="%.2f",
) / 100

t = st.sidebar.number_input(
    "Time to Expiration (in years)", 
    min_value=0.0, 
    value=0.5,
    step=0.01,
    format="%.2f",
)

#volatility in percent
vol = st.sidebar.number_input(
    "Volatility σ (%)", 
    min_value=0.0, 
    max_value=100.0, 
    value=20.0,
    step=0.01,
    format="%.2f",
) / 100

#Heatmap Sidebar Section
st.sidebar.header("Heatmap Parameters")

#Heatmap with a range of volatility values and Spot Price values
min_vol = st.sidebar.slider("Min Volatility (HeatMap)", 0.01, 1.00, 0.1, 0.01)
max_vol = st.sidebar.slider("Max Volatility (HeatMap)", 0.01, 1.00, 0.3, 0.01)
min_spot = st.sidebar.slider("Min Spot Price (HeatMap)", 1.0, 100.0, 60.0, 1.0,)
max_spot = st.sidebar.slider("Max Spot Price (HeatMap)", 1.0, 120.0, 100.0, 1.0)

#Error/Exception handling
if S < 0:
    st.error("Underlying price (S) cannot be negative")
if K < 0:
    st.error("Strike price (K) cannot be negative")
if t < 0:
    st.error("Time to expiration cannot be negative")
if vol < 0:
    st.error("Volatility cannot be negative")
if r < 0:
    st.error("Risk-free rate cannot be negative")

calculated_d1 = d1(S, K, r, t, vol)
calculated_d2 = d2(calculated_d1, vol, t)

# Custom CSS for option price display
st.markdown("""
<style>
.call-price {
    background-color: #90EE90;  /* Light green */
    color: black;
    padding: 10px;
    border-radius: 5px;
    font-size: 20px;
    text-align: center;
    font-weight: bold;
}
.put-price {
    background-color: #FFB6C1;  /* Light red/pink */
    color: black;
    padding: 10px;
    border-radius: 5px;
    font-size: 20px;
    text-align: center;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown(f'<div class="call-price">Call Option Price: ${call_price(S, calculated_d1, K, r, t, calculated_d2):.2f}</div>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="put-price">Put Option Price: ${put_price(S, calculated_d1, K, r, t, calculated_d2):.2f}</div>', unsafe_allow_html=True)


st.header("Option Pricing Interactive Heatmap")
st.write("Heatmap displaying fluctuation of option pricing with varying spot prices and volatility levels assuming a constant Strike Price")



#numpy version compatibility issue
np.Inf = np.inf

spot_prices = np.linspace(min_spot, max_spot, 10)
volatilities = np.linspace(min_vol, max_vol, 10)

call_prices = np.zeros((10, 10))
put_prices = np.zeros((10, 10))


for i, S_ in enumerate(spot_prices):
    for j, vol_ in enumerate(volatilities):
        d1_ = d1(S_, K, r, t, vol_)
        d2_ = d2(d1_, vol_, t)
        call_prices[i, j] = call_price(S_, d1_, K, r, t, d2_)
        put_prices[i, j] = put_price(S_, d1_, K, r, t, d2_)

# Plot Call Option Heatmap
col1, col2 = st.columns(2)

with col1:
    st.subheader("Call Option Heatmap")
    fig_call, ax_call = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        call_prices, 
        xticklabels=np.round(volatilities, 2), 
        yticklabels=np.round(spot_prices, 2), 
        cmap="YlGnBu", 
        annot=True, 
        fmt=".2f",
        ax=ax_call
    )
    ax_call.set_xlabel("Volatility (σ)")
    ax_call.set_ylabel("Spot Price (S)")
    ax_call.set_title("Call Option Prices")
    st.pyplot(fig_call)

with col2:
    st.subheader("Put Option Heatmap")
    fig_put, ax_put = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        put_prices, 
        xticklabels=np.round(volatilities, 2), 
        yticklabels=np.round(spot_prices, 2), 
        cmap="OrRd", 
        annot=True, 
        fmt=".2f",
        ax=ax_put
    )
    ax_put.set_xlabel("Volatility (σ)")
    ax_put.set_ylabel("Spot Price (S)")
    ax_put.set_title("Put Option Prices")
    st.pyplot(fig_put)


st.header("Input Parameters and Calculations")
st.write(
    pd.DataFrame(
        {
            "Parameter": ["Current Price (S)", "Strike Price (K)", "Risk-Free Rate (r)", "Time to Expiration (t)", "Volatility (σ)", "d1", "d2"],
            "Value": [S, K, r, t, vol, calculated_d1, calculated_d2]
        }
    )
)
