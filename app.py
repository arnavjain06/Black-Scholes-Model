import math
from scipy.stats import norm

import pandas as pd
import streamlit as st


#BS calculation functions
def d1 (S, K, r, t, vol):
    d1 = ((math.log(S/K) + ((r + 0.5 * vol**2))) / (vol * math.sqrt(t)))
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
    "Underlying Price (S)", 
    min_value=0.0, 
    value=42.0,
    step=0.01,
    format="%.2f",
)

K = st.sidebar.number_input(
    "Strike Price (K)", 
    min_value=0.0, 
    value=40.0,
    step=0.01,
    format="%.2f",
)

r = st.sidebar.number_input(
    "Risk-Free Rate (%)", 
    min_value=0.0, 
    max_value=100.0, 
    value=10.0,
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

vol = st.sidebar.number_input(
    "Volatility (%)", 
    min_value=0.0, 
    max_value=100.0, 
    value=20.0,
    step=0.01,
    format="%.2f",
) / 100

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

#metrics
st.write(
    pd.DataFrame(
        {
            "d1": [calculated_d1], 
            "d2": [calculated_d2], 
            "Call option price": [call_price(S, calculated_d1, K, r, t, calculated_d2)], 
            "Put option price": [put_price(S, calculated_d1, K, r, t, calculated_d2)], 
        }
    )
)
