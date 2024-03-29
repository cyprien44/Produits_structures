# fichier .run
from backend.monte_carlo import MonteCarlo
from backend.data.stock_data import StockData
from backend.data.rate_curve import ZeroCouponCurve
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    sj_1 = StockData(spot_price=170, strike=180, ticker='AAPL', maturity_date='2025-09-19')
    '''sj_2 = StockData(spot_price=100, volatility=0.1, dividend_yield=0.015)
    sj_3 = StockData(spot_price=100, volatility=0.4, dividend_yield=0)

    correlation_matrix = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])

    # Assurez-vous que votre classe MonteCarlo est mise à jour pour accepter les nouveaux paramètres
    simulation = MonteCarlo(spots=[sj_1.spot_price, sj_2.spot_price, sj_3.spot_price],
                            maturity=1.2,  # Assumant la même maturité pour simplification
                            risk_free_rate=0.02,
                            dividend_yields=[sj_1.dividend_yield, sj_2.dividend_yield, sj_3.dividend_yield],
                            volatilities=[sj_1.volatility, sj_2.volatility, sj_3.volatility],
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
    plt.show()'''

    '''volatility = sj_1.volatility

    # Tracer le smile de volatilité pour les puts
    plt.plot(volatility['strike'], volatility['implied_Volatility'], 'o', label='Volatilité implicite')

    plt.xlabel('Strike Price')
    plt.ylabel('Implied Volatility')
    plt.title('Volatility Smile')
    plt.legend()
    plt.show()

    print(sj_1.dividend_yield)'''

    # Calculer les taux zéro coupon
    curve_builder = ZeroCouponCurve({
            '13W': '^IRX',
            '5Y': '^FVX',
            '10Y': '^TNX',
            '30Y': '^TYX'
        })
    zero_coupon_rates = curve_builder.plot_curve()


