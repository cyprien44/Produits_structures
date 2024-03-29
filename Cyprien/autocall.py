import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


                # Pour le max_price_ratio, exclure l'observation date actuelle si ce n'est pas la première date
                if observation_date_index > 0:
                    max_price_ratio = df.iloc[:observation_date_index,:].max() / df.iloc[0,:]
                else:
                    # Si c'est la première date, utiliser simplement la valeur actuelle
                    max_price_ratio = price_ratio

                coupon_condition = price_ratio >= self.coupon_barrier
                autocall_condition = price_ratio >= self.autocall_barrier
                redemption_condition = max_price_ratio <= self.autocall_barrier
                
                
                # à la maturité on doit rendre le nominal meme si autocal_condition n'est pas rempli
                if observation_date_index == (len(df.index) - 1):
                    for i in range(len(redemption_condition)):
                        if bool(redemption_condition[i]):
                            redemption_condition.iloc[i] = autocall_condition.iloc[i] = True

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

    def plot_simulations(self):

        for actif_index, df in enumerate(self.wiener_process.dataframes):
            plt.figure(figsize=(10, 6))
            
            # Tracer chaque simulation pour l'actif courant
            for sim_index in df.columns:
                plt.plot(df.index, df[sim_index], lw=1)
            
            # Ajouter une ligne horizontale pour la barrière de coupon
            plt.axhline(y= self.coupon_barrier * 100, color='g', linestyle='--', label=f'Coupon Barrier ({round(self.coupon_barrier*100,1)}%)')
            
            # Ajouter une ligne horizontale pour la barrière d'autocall
            plt.axhline(y= self.autocall_barrier * 100, color='r', linestyle='--', label=f'Autocall Barrier ({round(self.autocall_barrier*100,1)}%)')
            
            plt.title(f'Wiener Process Simulation for Asset {actif_index + 1}')
            plt.xlabel('Time')
            plt.ylabel('Process Value')
            plt.legend()  # Affiche la légende
            plt.grid(True)
            plt.show()
    
    
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
                # Ajouter la somme des paiements actualisés pour cette DataFrame à la liste
                price_by_simul.append(total_discounted)  # Ajoute la somme totale de tous les paiements actualisés de cette DataFrame
                average_price.append(np.mean(total_discounted))

        # Imprimer les résultats de manière plus claire
        for i, value in enumerate(average_price):
            print(f"Prix moyen final pour l'actif {i+1}: {value:.2f}")
            
        overall_average = np.mean(average_price)
        print(f"Prix moyen final sur tous les actifs: {overall_average:.2f}")
