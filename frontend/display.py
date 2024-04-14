import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import matplotlib.dates as mdates
from matplotlib import cm
import plotly.graph_objects as go

def plot_volatility_surface_streamlit(stocks_list):
    for stock in stocks_list:
        volatility_df = stock.volatility_surface.data.copy()
        volatility_df['Maturity_Date'] = pd.to_datetime(volatility_df['Maturity_Date'])

        # Convertir explicitement les données en types appropriés
        volatility_df['Maturity_Days'] = volatility_df['Maturity_Days'].astype(float)
        volatility_df['Strike'] = volatility_df['Strike'].astype(float)
        volatility_df['Implied_Volatility'] = volatility_df['Implied_Volatility'].abs()

        fig = go.Figure(data=[go.Surface(z=volatility_df['Implied_Volatility'],
                                         x=volatility_df['Maturity_Days'],
                                         y=volatility_df['Strike'],
                                         colorscale='Viridis')])
        fig.update_layout(title=f'Implied Volatility Surface for {stock.ticker}', autosize=False,
                          width=500, height=500,
                          margin=dict(l=65, r=50, b=65, t=90))

        st.plotly_chart(fig)

def show_montecarlo_simulations(sim):
    num_rows = sim.stocks_nb
    num_cols = 1

    plt.figure(figsize=(20, 30))

    # Parcourir chaque sous-jacent
    for i, prices in enumerate(sim.simulations):
        plt.subplot(num_rows, num_cols, i + 1)
        for path in prices:
            plt.plot(path)
        plt.title(f'Chemins de prix simulés pour {sim.stocks[i].ticker}')

    st.pyplot(plt)
    plt.clf()


def plot_simulations_streamlit(autocall):
    for actif_index, (df, stock) in enumerate(zip(autocall.monte_carlo.simulations, autocall.monte_carlo.stocks)):
        fig, ax = plt.subplots(figsize=(10, 6))

        # Convertir l'index en datetime si ce n'est pas déjà le cas
        df.index = pd.to_datetime(df.index)

        # Tracer chaque simulation pour l'actif courant
        for sim_index in df.columns:
            ax.plot(df.index, df[sim_index], lw=1)

        # Ajouter une ligne horizontale pour la barrière de coupon et d'autocall
        ax.axhline(y=autocall.coupon_barrier * df.iloc[0, 0], color='g', linestyle='--',
                   label=f'Coupon Barrier ({round(autocall.coupon_barrier * df.iloc[0, 0], 1)})')
        ax.axhline(y=autocall.autocall_barrier * df.iloc[0, 0], color='r', linestyle='--',
                   label=f'Autocall Barrier ({round(autocall.autocall_barrier * df.iloc[0, 0], 1)})')
        ax.axhline(y=autocall.put_barrier * df.iloc[0, 0], color='orange', linestyle='--',
                   label=f'Put Barrier ({round(autocall.put_barrier * df.iloc[0, 0], 1)})')

        # Ajouter une ligne verticale pour chaque date d'observation
        for obs_date in autocall.monte_carlo.observation_dates:
            ax.axvline(x=obs_date, color='lightblue', linestyle='--', linewidth=1, alpha=0.5)

        # Formater l'axe des x pour afficher les dates de manière lisible
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)

        ax.set_title(f'Monte Carlo Simulation for {stock.ticker}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Process Value')
        ax.grid(True, which='both', axis='y', linestyle='--', color='grey')
        ax.grid(False, which='both', axis='x')
        ax.legend()

        st.pyplot(fig)
