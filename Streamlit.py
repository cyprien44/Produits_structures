import streamlit as st
import pandas as pd
from backend.monte_carlo import MonteCarlo
from backend.models import Autocall
from backend.data.stock_data import StockData
from frontend.display import plot_volatility_surface_streamlit, show_montecarlo_simulations, plot_simulations_streamlit

st.title("Simulation de produits autocallables")

# Sélection des dates de début et de fin
start_date = st.date_input("Date de début", pd.to_datetime("2024-03-01"))
end_date = st.date_input("Date de fin", pd.to_datetime("2025-03-01"))

# Vérification que la date de fin est postérieure à la date de début
if start_date >= end_date:
    st.error("La date de fin doit être postérieure à la date de début.")
else:

    name_to_ticker = {
    "Apple": "AAPL US Equity",
    "Microsoft": "MSFT US Equity",
    "Google": "GOOGL US Equity",
}

    # Liste prédéfinie des sous-jacents disponibles pour la sélection
    #available_stocks_keys = ['Apple', 'Microsoft', 'Google']
    selected_stocks_keys = st.multiselect("Choisissez les sous-jacents", list(name_to_ticker.keys()), default=list(name_to_ticker.keys()))

    # Paramètres de simulation - Première ligne
    st.header("Paramètres de simulation")
    cols = st.columns([1, 1, 1])

    with cols[0]:
        nominal = st.number_input("Valeur nominale", value=1000, key='nominal')
    with cols[1]:
        num_simu = st.number_input("Nombre de simulations", value=100, format="%d", key='num_simu')
    with cols[2]:
        day_conv = st.selectbox("Date de convention", [360, 365], key='day_conv')
        

    # Paramètres de simulation - Deuxième ligne
    cols2 = st.columns([1, 1, 1, 1])

    with cols2[0]:
        coupon_rate = st.number_input("Taux de coupon", value=0.05, key='coupon_rate')
    with cols2[1]:
        put_barrier = st.number_input("Barrière put", value=0.8, key='put_barrier')
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
        show_volatility = st.checkbox("Afficher les surfaces de volatilité implicite des sous-jacents")
        show_simulations = st.checkbox("Afficher les simulations de Monte Carlo")

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
        # Récupération des données des sous-jacents sélectionnés
        stock_data = {
            'Apple': ('AAPL US Equity'),
            'Microsoft': ('MSFT US Equity'),
            'Google': ('GOOGL US Equity'),
        }
        selected_stocks = [StockData(ticker=data, pricing_date=start_date.strftime('%Y%m%d')) for key, data in stock_data.items() if key in selected_stocks_keys]

        if show_volatility:
            plot_volatility_surface_streamlit(selected_stocks)

        monte_carlo = MonteCarlo(stocks=selected_stocks,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    num_simu=num_simu,
                    day_conv=day_conv,
                    seed=seed,
                    observation_frequency=observation_frequency)

        if show_simulations:
            show_montecarlo_simulations(monte_carlo)

        autocall = Autocall(monte_carlo=monte_carlo,
                            nominal=nominal,
                            coupon_rate=coupon_rate,
                            coupon_barrier=coupon_barrier,
                            autocall_barrier=autocall_barrier,
                            put_barrier=put_barrier)

        plot_simulations_streamlit(autocall)

        autocall.calculate_average_present_value()

        st.markdown("---")
        for stock, df in zip(monte_carlo.stocks, autocall.payoffs):
            st.write(f"Payoffs DataFrame for {stock.ticker}:")  # Utiliser .ticker ou .name
            st.dataframe(df)

        st.markdown("---")
        st.markdown("#### Prix moyen final pour chaque actif:")
        cols = st.columns(len(autocall.average_price))
        for stock, (col, value) in zip(monte_carlo.stocks, zip(cols, autocall.average_price)):
            col.metric(label=stock.ticker, value=f"{value:.2f} €")  # Utiliser .ticker ou .name

        st.markdown("---")
        st.markdown(f"""
            <div style='text-align: center;'>
                <span style='font-size: 1.5em;'>Prix moyen final sur tous les actifs:</span>
                <br>
                <span style='font-size: 2.5em;'>{autocall.overall_average:.2f} €</span>
            </div>
            """, unsafe_allow_html=True)


        