import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from backend.monte_carlo import MonteCarlo

# CSS pour définir un fond noir
st.markdown("""
<style>
body {
    color: #fff;
    background-color: #000;
}
</style>
    """, unsafe_allow_html=True)


st.title("Simulation de produits autocallables")

# Paramètres du marché
st.header("Paramètres du marché")
num_subjacent = st.slider("Nombre de sous-jacents", min_value=1, max_value=10, value=3, key='num_subjacent')

spots = []
volatilities = []
dividend_yields = []

# Paramètres pour chaque sous-jacent
for i in range(num_subjacent):
    st.markdown(f"#### Sous-jacent {i+1}")
    cols = st.columns([1, 2, 2, 2])
    with cols[1]:
        spot = st.number_input(f"Prix initial", value=100.0, key=f'spot_{i}')
    with cols[2]:
        volatility = st.number_input(f"Volatilité", value=0.2, key=f'volatility_{i}')
    with cols[3]:
        dividend_yield = st.number_input(f"Rendement de dividende", value=0.02, key=f'dividend_yield_{i}')
    spots.append(spot)
    volatilities.append(volatility)
    dividend_yields.append(dividend_yield)

# Paramètres de simulation en ligne
st.header("Paramètres de simulation")
cols = st.columns(4)
with cols[0]:
    maturity = st.number_input("Maturité (en années)", value=1.0, key='maturity')
with cols[1]:
    risk_free_rate = st.number_input("Taux sans risque annuel", value=0.02, key='risk_free_rate')
with cols[2]:
    num_simu = st.number_input("Nombre de simulations", value=1000, format="%d", key='num_simu')
with cols[3]:
    seed = st.number_input("Seed ", value=10, format="%d", key='seed')

# Bouton de simulation au centre
st.markdown("""
    <style>
    div.stButton > button:first-child {
        width: 100%;
        border: 2px solid #4CAF50;
        border-radius: 5px;
        font-size: 20px;
        color: white;
        background-color: #4CAF50;
        padding: 10px 24px;
        margin: 10px 0;
        cursor: pointer;
    }
    div.stButton > button:hover {
        background-color: #45a049;
    }
    </style>""", unsafe_allow_html=True)

if st.button("Simuler les chemins de prix"):
    correlation_matrix = np.eye(num_subjacent)
    simulation = MonteCarlo(spots, maturity, risk_free_rate, dividend_yields, volatilities, correlation_matrix, num_simu=num_simu, seed=seed)
    sim = simulation.simulate_prices()  # Corrigé ici

    # Ajustez les dimensions ici, par exemple en augmentant la hauteur
    plt.figure(figsize=(14, num_subjacent * 5))  # Augmentez la hauteur pour moins d'aplatissement

    for i, prices in enumerate(sim.T):
        plt.subplot(num_subjacent, 1, i + 1)
        for path in prices:
            plt.plot(path)
        plt.title(f'Sous-jacent {i + 1}')
    plt.tight_layout()
    st.pyplot(plt)
