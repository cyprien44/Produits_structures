from WienerProcess import WienerProcess
from maturity import Maturity
import numpy as np

if __name__ == '__main__':
    maturity = Maturity(1) 

    drift = np.array([0.05, 0.03])

    var_cov = np.array([[1, 0.2], [0.2, 1]]) 

    wiener_process = WienerProcess(
        drift=drift,
        var_cov=var_cov,
        maturity=maturity,
        nb_simulations=10, 
        nb_steps=252, 
        seed=42 
    )

    #results = wiener_process.simul()
    #print(wiener_process.__z)
    #print(wiener_process.simul(use_dataframe= True))
    wiener_process.plot_simulations()
