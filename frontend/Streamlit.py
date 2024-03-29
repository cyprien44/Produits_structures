import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from backend.monte_carlo import MonteCarlo
from backend.models import Autocall

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

# Paramètres de simulation - Première ligne
st.header("Paramètres de simulation")
cols = st.columns([1, 1, 1, 1])
with cols[0]:
    maturity = st.number_input("Maturité (en années)", value=1.0, key='maturity')
with cols[1]:
    risk_free_rate = st.number_input("Taux sans risque annuel", value=0.02, key='risk_free_rate')
with cols[2]:
    num_simu = st.number_input("Nombre de simulations", value=2, format="%d", key='num_simu')
with cols[3]:
     day_conv = st.number_input("Nombre de steps", value=9, format="%d", key='day_conv')
    

# Paramètres de simulation - Deuxième ligne
cols2 = st.columns([1, 1, 1, 1])
with cols2[0]:
    nominal = st.number_input("Valeur nominale", value=1000, key='nominal')
with cols2[1]:
    coupon_rate = st.number_input("Taux de coupon", value=0.05, key='coupon_rate')
with cols2[2]:
    coupon_barrier = st.number_input("Barrière de coupon", value=1.1, key='coupon_barrier')
with cols2[3]:
    autocall_barrier = st.number_input("Barrière d'autocall", value=1.2, key='autocall_barrier')

# Paramètres supplémentaires si nécessaire
with st.expander("Plus de paramètres"):
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
    monte_carlo = MonteCarlo(spots, maturity, risk_free_rate, dividend_yields, volatilities, correlation_matrix, num_simu=num_simu, seed=seed, day_conv=day_conv)
    autocall = Autocall(monte_carlo, nominal=nominal, coupon_rate=coupon_rate, coupon_barrier=coupon_barrier, autocall_barrier=autocall_barrier, risk_free=risk_free_rate)
    
    autocall.calculate_average_present_value()
    
    # Modifier la fonction plot_simulations pour être compatible avec Streamlit
    def plot_simulations_streamlit(autocall):
        for actif_index, df in enumerate(autocall.monte_carlo.simulations):
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for sim_index in df.columns:
                ax.plot(df.index, df[sim_index], lw=1)
            
            ax.axhline(y=autocall.coupon_barrier * 100, color='g', linestyle='--', label=f'Coupon Barrier ({round(autocall.coupon_barrier*100, 1)}%)')
            ax.axhline(y=autocall.autocall_barrier * 100, color='r', linestyle='--', label=f'Autocall Barrier ({round(autocall.autocall_barrier*100, 1)}%)')
            
            ax.set_title(f'Wiener Process Simulation for Asset {actif_index + 1}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Process Value')
            ax.legend()
            ax.grid(True)
            
            st.pyplot(fig)  # Utiliser st.pyplot pour afficher le graphique dans Streamlit
    
    # Appeler la fonction modifiée pour afficher les graphiques dans Streamlit
    plot_simulations_streamlit(autocall)
    
    # Afficher les DataFrames des payoffs et des payoffs actualisés
    for i, df in enumerate(autocall.payoffs):
        st.write(f"Payoffs DataFrame for Asset {i+1}:")
        st.dataframe(df)

    # Displaying average present values
    # This part assumes calculate_average_present_value has been called
    st.write("Valeurs présentes moyennes calculées pour chaque actif et la moyenne globale:")
    for i, value in enumerate(autocall.average_price):
        st.write(f"Prix moyen final pour l'actif {i+1}: {value:.2f}")
    
    st.write(f"Prix moyen final sur tous les actifs: {autocall.overall_average:.2f}")
    