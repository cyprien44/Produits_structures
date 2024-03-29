import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from backend.monte_carlo import MonteCarlo
from backend.data.stock_data import StockData

st.title("Simulation de produits autocallables")

# Paramètres de sous-jacent
num_subjacent = st.slider("Nombre de sous-jacents", min_value=1, max_value=5, value=3)
spots = []
volatilities = []
dividend_yields = []

for i in range(num_subjacent):
    st.header(f"Sous-jacent {i+1}")
    spot = st.number_input(f"Prix initial du sous-jacent {i+1}", value=100.0)
    volatility = st.number_input(f"Volatilité du sous-jacent {i+1}", value=0.2)
    dividend_yield = st.number_input(f"Rendement de dividende du sous-jacent {i+1}", value=0.02)
    spots.append(spot)
    volatilities.append(volatility)
    dividend_yields.append(dividend_yield)

# Autres paramètres
maturity = st.number_input("Maturité (en années)", value=1.0)
risk_free_rate = st.number_input("Taux sans risque annuel", value=0.02)
num_simu = st.number_input("Nombre de simulations", value=10000, format="%d")
seed = st.number_input("Seed pour la génération aléatoire", value=10, format="%d")

if st.button("Simuler les chemins de prix"):
    # Assumer une matrice de corrélation simple pour la démonstration
    correlation_matrix = np.eye(num_subjacent)
    
    # Créer l'instance de simulation Monte Carlo
    simulation = MonteCarlo(spots, maturity, risk_free_rate, dividend_yields,
                            volatilities, correlation_matrix, num_simu=num_simu, seed=seed)
    
    # Exécuter la simulation
    sim = simulation.simulate_correlated_prices()
    
    # Afficher les chemins de prix simulés
    plt.figure(figsize=(14, 7))
    for i, prices in enumerate(sim.T):
        plt.subplot(2, int(np.ceil(num_subjacent / 2)), i + 1)
        for path in prices:
            plt.plot(path)
        plt.title(f'Sous-jacent {i + 1}')
    plt.tight_layout()
    st.pyplot(plt)
