from WienerProcess import WienerProcess
from autocall import Autocall
from maturity import Maturity
import numpy as np

if __name__ == '__main__':
    maturity = Maturity(1) 

    drift = np.array([0.1, 0.01])

    var_cov = np.array([[0.5, 0.2], [0.2, 0.4]]) 

    wiener_process = WienerProcess(
        drift=drift,
        var_cov=var_cov,
        maturity=maturity,
        nb_simulations=2, 
        nb_steps=5, 
        seed=47
    )
    # Paramètres pour l'Autocall
    nominal = 1000  # Montant nominal de la note
    coupon_rate = 0.06  # Taux de coupon, par exemple 6%
    coupon_barrier = 1.1  # Barrière de coupon, par exemple 100% du prix initial
    autocall_barrier = 1.15  # Barrière d'autocall, par exemple 120% du prix initial
    risk_free = 0.03
    # Création d'une instance d'Autocall
    autocall = Autocall(
        wiener_process=wiener_process,
        nominal=nominal,
        coupon_rate=coupon_rate,
        coupon_barrier=coupon_barrier,
        autocall_barrier=autocall_barrier,
        risk_free = risk_free
    )


    #print(autocall.generate_payoffs())
    #print(wiener_process.simul())
    #wiener_process.plot_simulations()
    #autocall.print_total_payments()
    #autocall.generate_payoffs()
    #print(autocall.generate_payoff_dataframes())
    #autocall.print_payoff_dataframes()
    print(autocall.calculate_average_present_value())

