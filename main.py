from backend.monte_carlo import MonteCarlo
from backend.data.stock_data import StockData
from backend.data.rate_curve import ZeroCouponCurve
from backend.data.volatility import Volatility
from backend.data.correlation import *
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # Pour notre cas, nous prenons en date de départ le 2024-03-01 et en date de fin le 2034-03-01

    # Récupération de la courbe des taux
    US_rate = ZeroCouponCurve(date='20240301')
    #US_rate.plot_rate_curve()

    # Récupération des données des sous-jacents
    Apple = StockData(ticker='AAPL US Equity')
    Microsoft = StockData(ticker='MSFT US Equity')
    Google = StockData(ticker='GOOGL US Equity')

    # Récupération des corrélations
    correlation_matrix = get_correlation()

    # Récupération des volatilités implicites (pas terminé !!!!)
    #Apple_volatility = Volatility(stock=Apple, maturity_date='20340301', rate=US_rate.data, dividend_yield=Apple.dividend_yield)


    # Assurez-vous que votre classe MonteCarlo est mise à jour pour accepter les nouveaux paramètres
    simulation = MonteCarlo(spots=[Apple.spot_price, Microsoft.spot_price, Google.spot_price],
                            start_date="2024-03-01",
                            end_date="2025-03-01",
                            risk_free_rate=US_rate.data.iloc[0,0],
                            dividend_yields=[Apple.dividend_yield, Microsoft.dividend_yield, Google.dividend_yield],
                            volatilities=[0.2, 0.1, 0.15],
                            correlation_matrix=correlation_matrix,
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

    '''volatility = sj_1.volatility

    # Tracer le smile de volatilité pour les puts
    plt.plot(volatility['strike'], volatility['implied_Volatility'], 'o', label='Volatilité implicite')

    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.title('Volatility Smile')
    plt.legend()
    plt.show()

    print(sj_1.dividend_yield)'''


