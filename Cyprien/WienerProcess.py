import numpy as np
import pandas as pd


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
        self.nb_simulations = nb_simulations
        self.__nb_steps = nb_steps
        self.__seed = seed
        self.nb_actifs = var_cov.shape[0] 
        self.__z = None
        self.__dt = None
        self.__rdmts = None
        self.__price = None
        self.dataframes = None
        
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
            self.__z = np.random.multivariate_normal(mean, self.__var_cov, (self.nb_simulations, self.__nb_steps)) * np.sqrt(self.__dt)

    def simul(self):

        drift_per_step = self.__drift * self.__dt
        drift_adjustment = np.repeat(drift_per_step[np.newaxis, :], self.nb_simulations, axis=0)
        drift_adjustment = np.repeat(drift_adjustment[:, np.newaxis, :], self.__nb_steps, axis=1)
    
        self.__rdmts = self.__z + drift_adjustment
        
        # Calcul initial des prix basés sur les rendements cumulatifs
        initial_prices = np.exp(np.cumsum(self.__rdmts, axis=1))

        # Normalisation des prix pour que le premier prix soit égal à 100
        self.__price = (initial_prices / initial_prices[:, 0, np.newaxis]) * 100
        
        dataframes = []
        col = [f'Simulation {i+1}' for i in range(self.nb_simulations)]
        for actif_index in range(self.__price.shape[2]):
            # Extraire les données pour chaque actif et créer un DataFrame
            df = pd.DataFrame(self.__price[:, :, actif_index].T, columns=col)
            df.index = np.arange(1, self.__nb_steps + 1) * self.__dt
            df.index.name = 'Time'
            dataframes.append(df)

        self.dataframes = dataframes

        return dataframes
    
    

