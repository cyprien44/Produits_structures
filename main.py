# fichier .run
from backend.monte_carlo import MonteCarlo
from backend.data.stock_data import StockData
from backend.models import Autocall
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    sj_1 = StockData(spot_price=100, volatility=0.2, dividend_yield=0.02)
    sj_2 = StockData(spot_price=100, volatility=0.1, dividend_yield=0.015)
    sj_3 = StockData(spot_price=100, volatility=0.4, dividend_yield=0)

    correlation_matrix = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])

    # Assurez-vous que votre classe MonteCarlo est mise à jour pour accepter les nouveaux paramètres
    montecarlo = MonteCarlo(spots=[sj_1.spot_price, sj_2.spot_price, sj_3.spot_price],
                            maturity=1.2,  # Assumant la même maturité pour simplification
                            risk_free_rate=0.02,
                            dividend_yields=[sj_1.dividend_yield, sj_2.dividend_yield, sj_3.dividend_yield],
                            volatilities=[sj_1.volatility, sj_2.volatility, sj_3.volatility],
                            correlation_matrix=correlation_matrix,
                            num_simu=3,
                            day_conv=360,
                            seed=10)


    
    
    # Simulation de Monte Carlo (peut avoir été réalisée auparavant)
    simulated_prices = montecarlo.simulate_prices()
    print(simulated_prices.T)
    # Paramètres hypothétiques pour l'initialisation d'Autocall
    nominal_value = 1000  # La valeur nominale du produit structuré
    coupon_rate = 0.05  # Taux de coupon, par exemple 5%
    coupon_barrier = 1.1  # Barrière de coupon, par exemple 110%
    autocall_barrier = 1.2  # Barrière d'autocall, par exemple 120%
    risk_free_rate = 0.02  # Taux sans risque

    # Initialisation d'Autocall avec l'objet MonteCarlo
    autocall_product = Autocall(
        monte_carlo = montecarlo,
        nominal=nominal_value,
        coupon_rate=coupon_rate,
        coupon_barrier=coupon_barrier,
        autocall_barrier=autocall_barrier,
        risk_free=risk_free_rate
    )

    # Génération des payoffs
    payoffs = autocall_product.generate_payoffs()

    # Calcul du payoff moyen
    average_payoff = autocall_product.calculate_average_payoff()
    print(f"Le payoff moyen est: {average_payoff}")



    
    ''''
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

