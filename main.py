from backend.monte_carlo import MonteCarlo
from backend.data.stock_data import StockData
from backend.data.rate_curve import ZeroCouponCurve
from backend.models import Autocall

if __name__ == '__main__':
    # Pour notre cas, nous prenons en date de départ le 2024-03-01 et en date de fin le 2034-03-01

    # Récupération de la courbe des taux
    US_rate = ZeroCouponCurve(date='20240301')
    US_rate.plot_rate_curve()

    # Récupération des données des sous-jacents
    Apple = StockData(ticker='AAPL US Equity', pricing_date='20240301')
    Microsoft = StockData(ticker='MSFT US Equity', pricing_date='20240301')
    Google = StockData(ticker='GOOGL US Equity', pricing_date='20240301')

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

    simulation.show_simulations()

    '''autocall = Autocall(monte_carlo=simulation,
                        nominal=1000,
                        coupon_rate=0.06,
                        coupon_barrier=1.1,
                        autocall_barrier=1.40,
                        put_barrier=0.8)

    autocall.plot_simulations()
    autocall.print_payoff_dataframes()
    autocall.calculate_average_present_value()'''



