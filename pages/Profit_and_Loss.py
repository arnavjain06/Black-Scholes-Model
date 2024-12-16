import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.stats import norm

def d1(S, K, r, t, vol):
    return ((math.log(S / K) + (t * (r + 0.5 * vol**2))) / (vol * math.sqrt(t)))

def d2(d1, vol, t):
    return d1 - vol * math.sqrt(t)

def call_price(S, d1, K, r, t, d2):
    return S * norm.cdf(d1) - K * math.exp(-r * t) * norm.cdf(d2)

def put_price(S, d1, K, r, t, d2):
    return K * math.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)

st.sidebar.header("Option Parameters")

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

vol = st.sidebar.number_input(
    "Volatility σ (%)", 
    min_value=0.0, 
    max_value=100.0, 
    value=20.0, 
    step=0.01, 
    format="%.2f",
) / 100

call_purchase_price = st.sidebar.number_input(
    "Call Option Premium Paid", 
    min_value=0.0, 
    value=5.0, 
    step=0.01, 
    format="%.2f",
)

put_purchase_price = st.sidebar.number_input(
    "Put Option Premium Paid", 
    min_value=0.0, 
    value=5.0, 
    step=0.01, 
    format="%.2f",
)

min_vol = st.sidebar.slider("Min Volatility (HeatMap)", 0.01, 1.00, 0.1, 0.01)
max_vol = st.sidebar.slider("Max Volatility (HeatMap)", 0.01, 1.00, 0.3, 0.01)
min_spot = st.sidebar.slider("Min Spot Price (HeatMap)", 1.0, 100.0, 60.0, 1.0,)
max_spot = st.sidebar.slider("Max Spot Price (HeatMap)", 1.0, 120.0, 100.0, 1.0)

d1_current = d1(S, K, r, t, vol)
d2_current = d2(d1_current, vol, t)

call_current_price = call_price(S, d1_current, K, r, t, d2_current)
put_current_price = put_price(S, d1_current, K, r, t, d2_current)

call_intrinsic_value = max(0, S - K)
put_intrinsic_value = max(0, K - S)

call_profit_loss = call_intrinsic_value - call_purchase_price
put_profit_loss = put_intrinsic_value - put_purchase_price

st.header("Option Profit and Loss Analysis")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Call Option Premium Paid", f"${call_purchase_price:.2f}")
with col2:
    st.metric("Call Current Price", f"${call_current_price:.2f}")
with col3:
    st.metric("Call Intrinsic Value", f"${call_intrinsic_value:.2f}")

col4, col5, col6 = st.columns(3)
with col4:
    st.metric("Put Option Premium Paid", f"${put_purchase_price:.2f}")
with col5:
    st.metric("Put Current Price", f"${put_current_price:.2f}")
with col6:
    st.metric("Put Intrinsic Value", f"${put_intrinsic_value:.2f}")

col7, col8 = st.columns(2)
with col7:
    if call_profit_loss > 0:
        st.success(f"Call Option Profit: ${call_profit_loss:.2f}")
    elif call_profit_loss < 0:
        st.error(f"Call Option Loss: ${abs(call_profit_loss):.2f}")
    else:
        st.warning("Call Option Break-even")

with col8:
    if put_profit_loss > 0:
        st.success(f"Put Option Profit: ${put_profit_loss:.2f}")
    elif put_profit_loss < 0:
        st.error(f"Put Option Loss: ${abs(put_profit_loss):.2f}")
    else:
        st.warning("Put Option Break-even")

st.header("Option P&L Heatmaps")

spot_prices = np.linspace(min_spot, max_spot, 10)
volatilities = np.linspace(min_vol, max_vol, 10)

# Initialize profit/loss matrices
call_pnl = np.zeros((10, 10))
put_pnl = np.zeros((10, 10))

# Loop over spot prices and volatilities
for i, S_ in enumerate(spot_prices):
    for j, vol_ in enumerate(volatilities):
        d1_ = d1(S_, K, r, t, vol_)
        d2_ = d2(d1_, vol_, t)
        
        call_intrinsic_value = max(0, S_ - K)
        call_pnl[i, j] = call_intrinsic_value - call_purchase_price
        
        put_intrinsic_value = max(0, K - S_)
        put_pnl[i, j] = put_intrinsic_value - put_purchase_price

# Create heatmaps stacked vertically
fig, (ax_call, ax_put) = plt.subplots(2, 1, figsize=(15, 15))

# Plot Call PnL heatmap
sns.heatmap(
    call_pnl, 
    xticklabels=np.round(volatilities * 100, 2), 
    yticklabels=np.round(spot_prices, 2),
    cmap="RdYlGn",  # Green for profit, red for loss
    center=0,
    annot=True, 
    fmt=".2f",
    ax=ax_call
)
ax_call.set_xlabel("Volatility (σ)")
ax_call.set_ylabel("Spot Price (S)")
ax_call.set_title("Call Option Profit/Loss Heatmap")

# Plot Put PnL heatmap
sns.heatmap(
    put_pnl, 
    xticklabels=np.round(volatilities * 100, 2), 
    yticklabels=np.round(spot_prices, 2),
    cmap="RdYlGn",  # Green for profit, red for loss
    center=0,
    annot=True, 
    fmt=".2f",
    ax=ax_put
)
ax_put.set_xlabel("Volatility (σ)")
ax_put.set_ylabel("Spot Price (S)")
ax_put.set_title("Put Option Profit/Loss Heatmap")

st.pyplot(fig)

st.markdown("""
### Profit/Loss Calculation Explanation
- **Intrinsic Value**: The value of the option if exercised immediately
    - For Call: Max(0, Stock Price - Strike Price)
    - For Put: Max(0, Strike Price - Stock Price)
- **Profit/Loss**: Intrinsic Value - Premium Paid
- Green values indicate positive P&L (profit)
- Red values indicate negative P&L (loss)
""")