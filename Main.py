import streamlit as st

# Main app
st.title("Black-Scholes Model by Arnav Jain")
st.write("Use the sidebar to navigate between the pages")
st.header("Black-Scholes Description")
st.image("Images/BlackScholesEq.png")


st.markdown("""
### What is the Black-Scholes Model?

In 1973, developed by the economists Fischer Black and Myron Scholes, the Black-Scholes model is a mathematical formula that predicts the pricing of options before they're traded. In other words, it is a pricing calculator that accounts for multiple factors that influence an option's value.

### Variables Explained

- **S (Stock Price)**: The current market price of the underlying stock. 

- **K (Strike Price)**: The predetermined price at which you can buy for call options or sell for put options the stock. 

- **r (Risk-Free Rate)**: The interest rate you could earn on a safe investment, typically represented by government bonds. It is the financial industry's no-risk benchmark.

- **t (Time to Expiration)**: How long the option is valid, measured in years. It's the ticking clock that adds urgency and value to the option.

- **σ (Volatility)**: A measure of how much the stock price may vary. High volatility signals more potential for larger price swings. 

### How It Works

The model calculates option prices by considering these variables, essentially predicting how likely an option is to be profitable. It's not a fortune-telling device, but a sophisticated tool that gives traders a mathematical edge in understanding potential option values.
""")