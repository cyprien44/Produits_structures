import numpy as np
import pandas as pd

class MonteCarlo:
    """
    Classe Monte Carlo pour évaluer le prix de produits autocallables.
    """
    def __init__(self, spots, maturity, risk_free_rate, dividend_yields, volatilities,
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
        self.maturity = maturity
        self.risk_free_rate = risk_free_rate
        self.dividend_yields = np.array(dividend_yields)
        self.volatilities = np.array(volatilities)
        self.correlation_matrix = np.array(correlation_matrix)
        self.num_simu = num_simu
        self.num_time_steps = int(maturity * day_conv)
        self.delta_t = maturity / day_conv
        self.seed = seed
        self.generate_correlated_shocks()
        self.simulations = self.simulate_prices()

    def generate_correlated_shocks(self):
        """
        Génère des chocs corrélés pour tous les sous-jacents en utilisant la décomposition de Cholesky.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        L = np.linalg.cholesky(self.correlation_matrix)
        z_uncorrelated = np.random.normal(0.0, 1.0, (self.num_time_steps, self.num_simu, len(self.spots))) * self.delta_t ** 0.5
        self.z = np.einsum('ij, tkj -> tki', L, z_uncorrelated)

    def simulate_prices(self):
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
    
    def simulations_to_dataframes(self):
        """
        Converts simulation results into a list of DataFrames, one for each asset.
        Steps become the index and simulations become the columns.
        """
        # Assuming simulations have been run and self.simulations is populated
        dataframes = []
        for asset_index in range(self.simulations.shape[2]):
            # Extract simulation data for this asset
            asset_data = self.simulations[:, :, asset_index]
            
            # Create DataFrame: Steps as index, Simulations as columns
            df = pd.DataFrame(asset_data)
            df.index = [f'Step {step+1}' for step in range(self.num_time_steps + 1)]
            df.columns = [f'Simulation {sim+1}' for sim in range(self.num_simu)]
            
            dataframes.append(df)
        
        return dataframes
    
    def print_simulation_dataframes(self):
        """Print all simulation DataFrames sequentially."""
        # Generate DataFrames from simulations
        dataframes = self.simulations_to_dataframes()

        # Iterate over each DataFrame and print it
        for i, df in enumerate(dataframes):
            print(f"Simulation DataFrame for Asset {i+1}:")
            print(df)
            print("\n" + "-"*50 + "\n")

