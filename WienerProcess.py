import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class WienerProcess:
    
    def __init__(
        self, 
        drift: np.ndarray,  # drift est maintenant un vecteur
        var_cov: np.ndarray,  # var_cov est la matrice de variance-covariance
        maturity,
        nb_simulations: int = 1000,
        nb_steps: int = 1,
        seed: float = 272
    ) -> None:
        self.__drift = drift
        self.__var_cov = var_cov
        self.__maturity = maturity
        self.__nb_simulations = nb_simulations
        self.__nb_steps = nb_steps
        self.__seed = seed
        self.__z = None
        self.__dt = None
        self.__rdmts = None
        self.__price = None
        
        # Vérifier la cohérence de var_cov et drift
        if var_cov.shape[0] != len(drift):
            raise ValueError("La matrice de variance-covariance et le vecteur drift doivent avoir le même nombre de lignes.")
        
        # Generate numbers
        self.__generate()
    
    def __generate(self):
        if self.__z is None:
            if self.__seed is not None: 
                np.random.seed(self.__seed)
            self.__dt = self.__maturity.maturity() / self.__nb_steps
            mean = np.zeros(len(self.__drift))  # Moyenne des distributions
            self.__z = np.random.multivariate_normal(mean, self.__var_cov, (self.__nb_simulations, self.__nb_steps)) * np.sqrt(self.__dt)

    def simul(self, use_dataframe: bool = False):

        drift_per_step = self.__drift * self.__dt
        drift_adjustment = np.repeat(drift_per_step[np.newaxis, :], self.__nb_simulations, axis=0)
        drift_adjustment = np.repeat(drift_adjustment[:, np.newaxis, :], self.__nb_steps, axis=1)
    
        self.__rdmts = self.__z + drift_adjustment
        
        self.__price = np.exp(np.cumsum(self.__rdmts, axis=1)) * 100
        
        if use_dataframe:
            dataframes = []
            col = [f'Simulation {i+1}' for i in range(self.__nb_simulations)]
            for actif_index in range(self.__price.shape[2]):
                # Extraire les données pour chaque actif et créer un DataFrame
                df = pd.DataFrame(self.__price[:, :, actif_index].T, columns=col)
                df.index = np.arange(1, self.__nb_steps + 1) * self.__dt
                df.index.name = 'Time'
                dataframes.append(df)
            return dataframes
        else:
            return self.__price
    
    def plot_simulations(self):
        data = self.simul(use_dataframe=True)
        nb_actifs = len(data) # Nombre d'actifs, c'est-à-dire la dernière dimension de data
        
        # Itérer à travers chaque actif pour créer un graphique distinct
        # Itérer à travers chaque DataFrame (actif) pour créer un graphique distinct
        for actif_index, df in enumerate(data):
            plt.figure(figsize=(10, 6))
            # Tracer chaque simulation pour l'actif courant
            for sim_index in df.columns:
                plt.plot(df.index, df[sim_index], lw=1)
            
            plt.title(f'Wiener Process Simulation for Asset {actif_index + 1}')
            plt.xlabel('Time')
            plt.ylabel('Process Value')
            plt.grid(True)
            plt.show()

