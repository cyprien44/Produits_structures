# fichier .run
from backend.monte_carlo import MonteCarlo
from backend.data.stock_data import StockData
from backend.models import Autocall
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    sj_1 = StockData(spot_price=100, volatility=0.2, dividend_yield=0.02)
    sj_2 = StockData(spot_price=100, volatility=0.1, dividend_yield=0.015)
    sj_3 = StockData(spot_price=100, volatility=0.3, dividend_yield=0)

    correlation_matrix = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])

    # Assurez-vous que votre classe MonteCarlo est mise à jour pour accepter les nouveaux paramètres
    montecarlo = MonteCarlo(spots=[sj_1.spot_price, sj_2.spot_price, sj_3.spot_price],
                            maturity=1, #1.2,  # Assumant la même maturité pour simplification
                            risk_free_rate=0.02,
                            dividend_yields=[sj_1.dividend_yield, sj_2.dividend_yield, sj_3.dividend_yield],
                            volatilities=[sj_1.volatility, sj_2.volatility, sj_3.volatility],
                            correlation_matrix=correlation_matrix,
                            num_simu=2,
                            day_conv= 360,
                            observation_frequency='monthly',
                            seed=27)

    # Simulation de Monte Carlo (peut avoir été réalisée auparavant)
    montecarlo.simulate_prices()
    
    # Paramètres hypothétiques pour l'initialisation d'Autocall
    nominal_value = 1000  # La valeur nominale du produit structuré
    coupon_rate = 0.05  # Taux de coupon, par exemple 5%
    coupon_barrier = 1.1  # Barrière de coupon, par exemple 110%
    autocall_barrier = 1.5  # Barrière d'autocall, par exemple 120%
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
    autocall.plot_simulations()
    # Calcul du payoff moyen
    autocall.print_average_present_values()

    


