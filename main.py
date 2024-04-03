from backend.monte_carlo import MonteCarlo
from backend.data.stock_data import StockData
from backend.models import Autocall
from backend.data.volatility import Volatility
from backend.data.rate_curve import ZeroCouponCurve
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib
import numpy as np


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
    montecarlo = MonteCarlo(stocks=[Apple, Microsoft, Google],
                            start_date="2024-03-01",
                            end_date="2025-03-01",
                            risk_free_rate=0.02,
                            num_simu=10000,
                            day_conv=360,
                            seed=10,
                            observation_frequency='monthly')

    # Simuler les chemins de prix corrélés
    sim = montecarlo.simulations_correlated


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


    # Paramètres hypothétiques pour l'initialisation d'Autocall
    nominal_value = 1000  # La valeur nominale du produit structuré
    coupon_rate = 0.05  # Taux de coupon, par exemple 5%
    coupon_barrier = 1.1  # Barrière de coupon, par exemple 110%
    autocall_barrier = 1.15  # Barrière d'autocall, par exemple 120%
    risk_free_rate = 0.02  # Taux sans risque

    # Initialisation d'Autocall avec l'objet MonteCarlo
    autocall = Autocall(
        monte_carlo = montecarlo,
        nominal=nominal_value,
        coupon_rate=coupon_rate,
        coupon_barrier=coupon_barrier,
        autocall_barrier=autocall_barrier,
        risk_free=risk_free_rate
    )

    # Génération des payoffs
    autocall.print_payoffs_dataframes()
    autocall.show_simulations()
    # Calcul du payoff moyen
    autocall.print_average_present_values()

