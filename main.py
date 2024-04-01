from backend.monte_carlo import MonteCarlo
from backend.data.stock_data import StockData
from backend.data.rate_curve import ZeroCouponCurve
from backend.data.volatility import Volatility

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
import numpy as np
import plotly.graph_objects as go

if __name__ == '__main__':
    # Pour notre cas, nous prenons en date de départ le 2024-03-01 et en date de fin le 2034-03-01

    # Récupération de la courbe des taux
    US_rate = ZeroCouponCurve(date='20240301')
    #US_rate.plot_rate_curve()

    # Récupération des données des sous-jacents
    Apple = StockData(ticker='AAPL US Equity', pricing_date='20240301', rate=US_rate)
    Microsoft = StockData(ticker='MSFT US Equity', pricing_date='20240301', rate=US_rate)
    Google = StockData(ticker='GOOGL US Equity', pricing_date='20240301', rate=US_rate)

    '''Surface de volatilité implicite'''
    # Affichage de la surface de volatilité implicite
    Apple.volatility_surface.plot_volatility_surface()
    Microsoft.volatility_surface.plot_volatility_surface()
    Google.volatility_surface.plot_volatility_surface()


    # Assurez-vous que votre classe MonteCarlo est mise à jour pour accepter les nouveaux paramètres
    simulation = MonteCarlo(stocks=[Apple, Microsoft, Google],
                            start_date="2024-03-01",
                            end_date="2025-03-01",
                            num_simu=10000,
                            day_conv=360,
                            seed=10)

    # Simuler les chemins de prix corrélés
    sim = simulation.simulate_correlated_prices()


    """Afficher les chemins de prix simulés pour chaque sous-jacent."""

    # Obtenir le nombre de sous-jacents
    num_subjacent = len(sim.T)

    # Calculer le nombre de lignes et de colonnes pour le tableau de graphiques
    num_rows = int(np.ceil(np.sqrt(num_subjacent)))
    num_cols = int(np.ceil(num_subjacent / num_rows))

    plt.figure(figsize=(14, 7))

    # Parcourir chaque sous-jacent
    for i, prices in enumerate(sim.T):
        # Créer un sous-graphique pour chaque sous-jacent
        plt.subplot(num_rows, num_cols, i + 1)
        for path in prices:
            plt.plot(path)
        plt.title(f'Chemins de prix simulés pour le sous-jacent {i + 1}')

    plt.tight_layout()
    plt.show()




