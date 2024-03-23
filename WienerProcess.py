

from maturity import Maturity
import numpy as np
import pandas as pd

class WienerProcess:
    
    def __init__(
        self, 
        drift:float, 
        volatility:float,
        maturity:Maturity,
        nb_simulations:int=1000,
        nb_steps:int=1,
        seed:float=272
    ) -> None:
        self.__drift=drift
        self.__volatility=volatility
        self.__maturity=maturity
        self.__nb_simulations=nb_simulations
        self.__nb_steps=nb_steps
        self.__seed=seed
        self.__z=None
        self.__dt=None
        
        # Generate numbers
        self.__generate()
    
    def __generate(self):
        if self.__z==None:
            if self.__seed!=None: 
                np.random.seed(self.__seed)
            self.__dt=self.__maturity.maturity()/self.__nb_steps
            self.__z = np.random.normal(0.0,1.0,[self.__nb_simulations, self.__nb_steps])*self.__dt**0.5
                                        # Pour la corrélation remplacer 1.0 par une matrice de variance covariance de type [[1;0.2];[0.2;1]]
                                        # probabilité E(x) = q * alpha * Var(X)^1/2 avec q quantile 95% 

    def simul(self, use_dataframe:bool=True):
        """Return simulations

        Returns:
            _type_: _description_
        """
        x=self.__z*self.__volatility+self.__drift*self.__dt
        if use_dataframe:
            col=np.arange(1, self.__nb_steps+1)*self.__dt
            return pd.DataFrame(
                x, 
                columns=col
            )
        return x
        