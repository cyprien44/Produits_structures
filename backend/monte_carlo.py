import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

class MonteCarlo:
    def __init__(self, spots, maturity, risk_free_rate, dividend_yields, volatilities,
                 correlation_matrix, num_simu=10000, day_conv=360, seed=None, observation_frequency='monthly'):
        """
        Initialisation avec prise en compte de la fréquence d'observation.
        """
        self.spots = np.array(spots)
        self.maturity = maturity
        self.risk_free_rate = risk_free_rate
        self.dividend_yields = np.array(dividend_yields)
        self.volatilities = np.array(volatilities)
        self.correlation_matrix = np.array(correlation_matrix)
        self.num_simu = num_simu
        self.num_steps = None
        self.day_conv = day_conv
        self.delta_t = 1 / day_conv  # Ajustement pour les simulations quotidiennes
        self.seed = seed
        self.observation_frequency = observation_frequency
        self.simulation_dates = self.generate_simulation_dates()
        self.observation_dates = self.generate_observation_dates()
        self.generate_correlated_shocks()
        self.simulations = self.simulate_prices()
        

    def generate_simulation_dates(self):
        """
        Génère les dates de chaque étape de simulation sur une base quotidienne.
        """
        start_date = pd.Timestamp.today()
        end_date = start_date + pd.offsets.DateOffset(years=self.maturity)

        # Générer les dates de simulation quotidiennes
        dates = pd.date_range(start=start_date, end=end_date, freq='B').normalize()  # 'B' pour jours ouvrables

        return dates

    def generate_observation_dates(self):
        """
        Génère les dates d'observations basées sur la fréquence et ajuste selon les jours ouvrables.
        """
        start_date = pd.Timestamp.today()
        end_date = start_date + pd.offsets.DateOffset(years=self.maturity)

        if self.observation_frequency == 'monthly':
            freq = 'BM'
        elif self.observation_frequency == 'quarterly':
            freq = 'BQ'
        elif self.observation_frequency == 'semiannually':
            freq = 'BQ-FEB,AUG'
        elif self.observation_frequency == 'annually':
            freq = 'BA'
        else:
            raise ValueError("Fréquence d'observation non reconnue.")

        # Générer les dates d'observations
        dates = pd.date_range(start=start_date, end=end_date, freq=freq).normalize()

        return dates
    def generate_correlated_shocks(self):
        """
        Génère des chocs corrélés pour tous les sous-jacents en utilisant la décomposition de Cholesky.
        """
        self.num_steps = len(self.simulation_dates)
        if self.seed is not None:
            np.random.seed(self.seed)
        L = np.linalg.cholesky(self.correlation_matrix)
        z_uncorrelated = np.random.normal(0.0, 1.0, (self.num_steps, self.num_simu, len(self.spots))) * self.delta_t ** 0.5
        self.z = np.einsum('ij, tkj -> tki', L, z_uncorrelated)

    def simulate_prices(self):
        """
        Simule les chemins de prix avec des dates de simulation et retourne les résultats sous forme de DataFrames.
        """
        dt = self.delta_t
        simu = np.zeros((self.num_steps, self.num_simu, len(self.spots)))
        simu[0, :, :] = self.spots
        
        for t in range(1, self.num_steps):
            simu[t] = simu[t-1] * np.exp(
                (self.risk_free_rate - self.dividend_yields - 0.5 * self.volatilities**2) * dt + self.volatilities * self.z[t-1]
            )

        dataframes = []
        for asset_index in range(simu.shape[2]):
            asset_data = simu[:, :, asset_index]
            df = pd.DataFrame(asset_data, index=self.simulation_dates, columns=[f'Simulation {sim+1}' for sim in range(self.num_simu)])
            dataframes.append(df)

        return dataframes
    
    def print_simulation_dataframes(self):
        """Print all simulation DataFrames sequentially."""
        # Generate DataFrames from simulations
        dataframes = self.simulations

        # Iterate over each DataFrame and print it
        for i, df in enumerate(dataframes):
            print(f"Simulation DataFrame for Asset {i+1}:")
            print(df)
            print("\n" + "-"*50 + "\n")

