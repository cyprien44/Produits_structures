import numpy as np
import pandas as pd
import math
from WienerProcess import WienerProcess

class Autocall:
    def __init__(
        self,
        wiener_process: WienerProcess,
        nominal: float,
        coupon_rate: float,
        coupon_barrier: float,
        autocall_barrier: float,
        risk_free: float,
        rate_type: str = "continuous"
    ):
        self.wiener_process = wiener_process
        self.nominal = nominal
        self.coupon_rate = coupon_rate
        self.coupon_barrier = coupon_barrier
        self.autocall_barrier = autocall_barrier
        self.risk_free = risk_free
        self.rate_type = rate_type
        self.payoffs = {i: [] for i in range(self.wiener_process.nb_actifs)}
        self.has_autocalled = {i: False for i in range(self.wiener_process.nb_actifs)}
        self.payoff_dataframes = []  

    def discount_factor(self, maturity, force_rate=None):
        rate = self.risk_free
        if force_rate is not None:
            rate = force_rate
        if self.rate_type == "continuous":
            return math.exp(-rate * maturity)
        elif self.rate_type == "compounded":
            return 1.0 / (1 + rate) ** maturity

    def generate_payoffs(self):
        dataframes = self.wiener_process.simul()
        
        for actif_index, df in enumerate(dataframes):
            for observation_date_index, observation_date in enumerate(df.index):
                # Calculer le coupon et le redemption à chaque date d'observation
                price = df.loc[observation_date,:]
                price_ratio = price / df.iloc[0,:]

                coupon_condition = price_ratio >= self.coupon_barrier
                autocall_condition = price_ratio >= self.autocall_barrier

                # Pour le max_price_ratio, exclure l'observation date actuelle si ce n'est pas la première date
                if observation_date_index > 0:
                    max_price_ratio = df.iloc[:observation_date_index].max() / df.iloc[0]
                else:
                    # Si c'est la première date, utiliser simplement la valeur actuelle
                    max_price_ratio = price_ratio
                    
                redemption_condition = max_price_ratio <= self.autocall_barrier

                coupon_payment = self.nominal * self.coupon_rate * coupon_condition * redemption_condition
                redemption_payment = self.nominal * autocall_condition * redemption_condition
                total_payment = coupon_payment + redemption_payment

                # Ajouter les payoffs pour cette date d'observation
                self.payoffs[actif_index].append({
                    'observation_date': observation_date,
                    'coupon_payment': coupon_payment,
                    'redemption_payment': redemption_payment,
                    'total_payment': total_payment 
                })
            
        return self.payoffs

    def generate_payoff_dataframes(self):

        if not any(self.payoffs.values()):  # Si les payoffs sont vides pour tous les actifs
            self.generate_payoffs() 
        
        # Convertir self.payoffs en DataFrames et stocker dans self.payoff_dataframes
        self.payoff_dataframes = []
        for actif_index, payoffs in self.payoffs.items():
            if payoffs:
                df = pd.DataFrame([{**{'observation_date': p['observation_date']}, **p['total_payment']} for p in payoffs])
                df.set_index('observation_date', inplace=True)
                self.payoff_dataframes.append(df)
            else:
                self.payoff_dataframes.append(pd.DataFrame())  # DataFrame vide si pas de payoffs

        return self.payoff_dataframes
    

    def print_payoff_dataframes(self):
        if not self.payoff_dataframes:
            self.generate_payoff_dataframes()
        
        for i, df in enumerate(self.payoff_dataframes):
            if not df.empty:
                print(f"Payoffs pour l'actif {i+1}:")
                print(df)
                print("\n")
            else:
                print(f"Aucun payoff pour l'actif {i+1}.")
    
    def calculate_average_present_value(self):

        if not self.payoff_dataframes:  # Si les DataFrames ne sont pas encore générés
            self.generate_payoff_dataframes()

        price_by_simul = []
        average_price = []
        for df in self.payoff_dataframes:
            if not df.empty:
                discounted_df = df.apply(lambda x: x * self.discount_factor(x.name), axis=1)
                # Calculer la somme des valeurs actualisées pour obtenir le total actualisé par simulation
                total_discounted = discounted_df.sum()
                print(total_discounted)
                # Ajouter la somme des paiements actualisés pour cette DataFrame à la liste
                price_by_simul.append(total_discounted)  # Ajoute la somme totale de tous les paiements actualisés de cette DataFrame
                average_price.append(np.mean(total_discounted))

        return average_price