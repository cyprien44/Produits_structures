# 📈 Autocallable Product Simulation

This Streamlit application simulates **autocallable structured products** on selected underlying assets (Apple, Microsoft, Google). It performs Monte Carlo simulations of asset paths and calculates payoff outcomes and present values based on user-defined barriers and strategy.

🌐 **Live app**: [pricing-autocall-cyprien.streamlit.app](http://pricing-autocall-cyprien.streamlit.app/)

## 🔧 Features

- Strategy selection: mono-asset, worst-of, best-of
- Underlying asset selection
- Full parameter customization: barriers, coupon rate, nominal value, observation frequency, etc.
- Visualization of implied volatility surfaces
- Monte Carlo simulation of price paths
- Display of payoff matrix and average present value

## ▶️ Run the app locally

Make sure you have the dependencies installed (see `requirements.txt`):

```bash
pip install -r requirements.txt
