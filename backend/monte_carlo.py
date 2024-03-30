import numpy as np
from datetime import datetime

class MonteCarlo:
    """
    Classe Monte Carlo pour évaluer le prix de produits autocallables.
    """
    def __init__(self, spots, start_date, end_date, risk_free_rate, dividend_yields, volatilities,
                 correlation_matrix, num_simu=10000, day_conv=360, seed=None):
        """
        Initialisation de la classe Monte Carlo pour un nombre quelconque de sous-jacents.

        :param spots: Liste des prix initiaux des sous-jacents.
        :param maturity: Maturité du produit (en années).
        :param risk_free_rate: Taux sans risque annuel.
        :param dividend_yields: Liste des rendements de dividendes annuels pour chaque sous-jacent.
        :param volatilities: Liste des volatilités annuelles pour chaque sous-jacent.
        :param correlation_matrix: Matrice de corrélation NxN pour les sous-jacents.
        :param num_simu: Nombre de chemins à simuler.
        :param day_conv: Nombre de jours de trading par an.
        """
        self.spots = np.array(spots)
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.maturity = (self.end_date - self.start_date).days / 365
        self.risk_free_rate = risk_free_rate
        self.dividend_yields = np.array(dividend_yields)
        self.volatilities = np.array(volatilities)
        self.correlation_matrix = correlation_matrix
        self.num_simu = num_simu
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
        for t in range(1, self.num_time_steps + 1):
            simu[t] = simu[t-1] * np.exp(
                (self.risk_free_rate - self.dividend_yields - 0.5 * self.volatilities**2)
                * dt + self.volatilities * self.z[t-1]
            )
        return simu
