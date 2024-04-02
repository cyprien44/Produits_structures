import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from backend.monte_carlo import MonteCarlo
from backend.models import Autocall
import matplotlib.dates as mdates

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
        volatility = st.number_input(f"Volatilité", value=0.3, key=f'volatility_{i}')
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
     day_conv = st.number_input("Nombre de steps", value=360, format="%d", key='day_conv')
    

# Paramètres de simulation - Deuxième ligne
cols2 = st.columns([1, 1, 1, 1])
with cols2[0]:
    nominal = st.number_input("Valeur nominale", value=1000, key='nominal')
with cols2[1]:
    coupon_rate = st.number_input("Taux de coupon", value=0.05, key='coupon_rate')
with cols2[2]:
    coupon_barrier = st.number_input("Barrière de coupon", value=1.1, key='coupon_barrier')
with cols2[3]:
    autocall_barrier = st.number_input("Barrière d'autocall", value=1.3, key='autocall_barrier')

# Paramètres supplémentaires si nécessaire
with st.expander("Plus de paramètres"):
    observation_frequency = st.selectbox(
    "Fréquence d'observation",
    ['monthly', 'quarterly', 'semiannually', 'annually'],
    index=0  # Choix par défaut à 'monthly'
)
    seed = st.number_input("Seed ", value=24, format="%d", key='seed')
    

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
    monte_carlo = MonteCarlo(spots, maturity, risk_free_rate, dividend_yields, volatilities, correlation_matrix, num_simu=num_simu,day_conv=day_conv, observation_frequency=observation_frequency, seed=seed)
    autocall = Autocall(monte_carlo, nominal=nominal, coupon_rate=coupon_rate, coupon_barrier=coupon_barrier, autocall_barrier=autocall_barrier, risk_free=risk_free_rate)
    
    def plot_simulations_streamlit(autocall):
        for actif_index, df in enumerate(autocall.monte_carlo.simulations):
            fig, ax = plt.subplots(figsize=(10, 6))

            # Convertir l'index en datetime si ce n'est pas déjà le cas
            df.index = pd.to_datetime(df.index)

            # Tracer chaque simulation pour l'actif courant
            for sim_index in df.columns:
                ax.plot(df.index, df[sim_index], lw=1)

            # Ajouter une ligne horizontale pour la barrière de coupon et d'autocall
            ax.axhline(y=autocall.coupon_barrier * 100, color='g', linestyle='--', label=f'Coupon Barrier ({round(autocall.coupon_barrier*100, 1)}%)')
            ax.axhline(y=autocall.autocall_barrier * 100, color='r', linestyle='--', label=f'Autocall Barrier ({round(autocall.autocall_barrier*100, 1)}%)')

            # Ajouter une ligne verticale pour chaque date d'observation
            for obs_date in autocall.monte_carlo.observation_dates:
                ax.axvline(x=obs_date, color='lightblue', linestyle='--', linewidth=1, alpha=0.5)

            # Formater l'axe des x pour afficher les dates de manière lisible
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)

            ax.set_title(f'Monte carlo simulation for Asset {actif_index + 1}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Process Value')
            ax.grid(True, which='both', axis='y', linestyle='--', color='grey')
            ax.grid(False, which='both', axis='x')
            ax.legend()

            st.pyplot(fig) 

    plot_simulations_streamlit(autocall)

    autocall.calculate_average_present_value()

    st.markdown("---")  # Un séparateur visuel
        # Afficher les DataFrames des payoffs et des payoffs actualisés
    for i, df in enumerate(autocall.payoffs):
        st.write(f"Payoffs DataFrame for Asset {i+1}:")
        st.dataframe(df)

    st.markdown("---")  # Un séparateur visuel

    # Titre pour la section des prix moyens individuels
    st.markdown("#### Prix moyen final pour chaque actif:")

    # Créer une grille avec un certain nombre de colonnes équivalent au nombre d'actifs
    cols = st.columns(len(autocall.average_price))

    # Remplir chaque colonne avec le nom de l'actif et son prix moyen final
    for i, (col, value) in enumerate(zip(cols, autocall.average_price)):
        col.metric(label=f"Actif {i+1}", value=f"{value:.2f} €")

    st.markdown("---")  # Un séparateur visuel

    # Pour le prix moyen final sur tous les actifs, on utilise HTML pour une taille de police plus grande
    st.markdown(f"""
        <div style='text-align: center;'>
            <span style='font-size: 1.5em;'>Prix moyen final sur tous les actifs:</span>
            <br>
            <span style='font-size: 2.5em;'>{autocall.overall_average:.2f} €</span>
        </div>
        """, unsafe_allow_html=True)


    