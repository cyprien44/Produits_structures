import numpy as np
from datetime import datetime
from scipy.interpolate import interp2d, interp1d
from backend.data.correlation import *

class MonteCarlo:
    """
    Classe Monte Carlo pour évaluer le prix de produits autocallables.
    """
    def __init__(self, stocks, start_date, end_date, num_simu=10000, day_conv=360, seed=None):
        """
        Initialisation de la classe Monte Carlo pour un nombre quelconque de sous-jacents.

        :param spots: Liste des prix initiaux des sous-jacents.
        :param maturity: Maturité du produit (en années).
        :param risk_free_rate: Taux sans risque annuel.
        :param dividend_yields: Liste des rendements de dividendes annuels pour chaque sous-jacent.
        :param correlation_matrix: Matrice de corrélation NxN pour les sous-jacents.
        :param num_simu: Nombre de chemins à simuler.
        :param day_conv: Nombre de jours de trading par an.
        """
        self.stocks = stocks
        self.spots = np.array([stock.spot_price for stock in stocks])
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.maturity = (self.end_date - self.start_date).days / 365
        self.dividend_yields = np.array([stock.dividend_yield for stock in stocks])
        self.correlation_matrix = get_correlation()
        self.num_simu = num_simu
        self.day_conv = day_conv
        self.num_time_steps = int(self.maturity * day_conv)
        self.delta_t = self.maturity / day_conv
        self.seed = seed
        self.generate_correlated_shocks()
        self.simulations = self.simulate_correlated_prices()

    def generate_correlated_shocks(self):
        """
        Génère des chocs corrélés pour tous les sous-jacents en utilisant la décomposition de Cholesky.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        L = np.linalg.cholesky(self.correlation_matrix)
        z_uncorrelated = np.random.normal(0.0, 1.0, (self.num_time_steps, self.num_simu, len(self.spots))) * self.delta_t ** 0.5
        self.z = np.einsum('ij, tkj -> tki', L, z_uncorrelated)

    def simulate_correlated_prices(self):
        """
        Simule les chemins de prix pour tous les sous-jacents en utilisant les chocs corrélés.
        """
        dt = self.delta_t
        simu = np.zeros((self.num_time_steps + 1, self.num_simu, len(self.spots)))
        simu[0, :, :] = self.spots

        # Create an interpolation function for the volatility and rate of each stock
        volatilities = [interp2d(stock.volatility_surface.volatility_surface['dates_in_years'],
                                 stock.volatility_surface.volatility_surface['strike'],
                                 stock.volatility_surface.volatility_surface['implied_Volatility'],) for stock in self.stocks]
        rates = [interp1d(stock.rate_curve.data['maturity_in_years'], stock.rate_curve.data['rates'],
                          fill_value="extrapolate") for stock in
                 self.stocks]

        for t in range(1, self.num_time_steps + 1):
            t_in_years = t / self.day_conv
            for i in range(len(self.spots)):
                # Get the precomputed volatility and rate
                volatility = volatilities[i](t_in_years, simu[t - 1, :, i]).flatten()
                rate = rates[i](t_in_years)
                simu[t, :, i] = simu[t - 1, :, i] * np.exp(
                    (rate - self.dividend_yields[i] - 0.5 * volatility ** 2) * dt + volatility * self.z[t - 1, :, i])
        return simu