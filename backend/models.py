import numpy as np
import pandas as pd

class Autocall:
    def __init__(self, monte_carlo, nominal, coupon_rate, coupon_barrier, autocall_barrier, risk_free):
        self.monte_carlo = monte_carlo
        self.nominal = nominal
        self.coupon_rate = coupon_rate
        self.coupon_barrier = coupon_barrier
        self.autocall_barrier = autocall_barrier
        self.risk_free = risk_free
        self.payoffs, self.payoffs_discount = self.generate_payoffs()


    def discount_factor(self, step, total_steps):
        time = step / total_steps * self.monte_carlo.maturity  # Convertir en fraction de la maturité totale
        return np.exp(-self.risk_free * time)


    def generate_payoffs(self):

        payoffs_dataframes = []
        discounted_payoffs_dataframes = []

        for actif_index, df in enumerate(self.monte_carlo.simulations):
            num_steps = df.shape[0]
            num_simulations = df.shape[1]

            payoffs_actif = np.zeros((num_steps, num_simulations))
            discounted_payoffs_actif = np.zeros((num_steps, num_simulations))

            for step, time_step in enumerate(df.index):
                current_prices = df.iloc[step].values
                initial_prices = df.iloc[0].values
                price_ratios = current_prices / initial_prices

                coupon_condition = price_ratios >= self.coupon_barrier
                autocall_condition = price_ratios >= self.autocall_barrier
                max_price_ratios = df.iloc[:step].max(axis=0) / initial_prices
                redemption_condition = max_price_ratios <= self.autocall_barrier

                if step == (len(df.index) - 1):
                    for i in range(len(redemption_condition)):
                        if bool(redemption_condition[i]):
                            redemption_condition[i] = autocall_condition[i] = True

                coupon_payment = self.nominal * self.coupon_rate * coupon_condition * redemption_condition
                redemption_payment = self.nominal * autocall_condition * redemption_condition
                total_payment = coupon_payment + redemption_payment

                payoffs_actif[step, :] = total_payment
                discount = self.discount_factor(step, num_steps)
                discounted_payoffs_actif[step, :] = (total_payment * discount)

            df_payoffs = pd.DataFrame(payoffs_actif, index=df.index, columns=[f'Simulation {sim+1}' for sim in range(num_simulations)])
            df_discounted_payoffs = pd.DataFrame(discounted_payoffs_actif, index=df.index, columns=[f'Simulation {sim+1}' for sim in range(num_simulations)])
            
            payoffs_dataframes.append(df_payoffs)
            discounted_payoffs_dataframes.append(df_discounted_payoffs)

        return payoffs_dataframes, discounted_payoffs_dataframes


    
    def print_payoffs_dataframes(self):
        """
        Affiche les DataFrames des payoffs pour chaque actif.
        """
        if not self.payoffs:  # Si les payoffs n'ont pas encore été générés
            self.generate_payoffs()
            
        for i, df in enumerate(self.payoffs):
            print(f"Payoffs DataFrame for Asset {i+1}:")
            print(df)

    def calculate_average_payoff(self):
        all_payoffs = self.generate_payoffs()
        # Moyenne sur toutes les simulations à chaque étape
        average_payoff_per_step = np.mean(all_payoffs, axis=1)
        # Moyenne sur toutes les étapes
        return np.mean(average_payoff_per_step, axis=0) 



'''
def price_autocall(self):
    """
    Calcule le prix d'un produit autocallable en utilisant les chemins de prix simulés.
    Cette fonction doit être adaptée en fonction des spécificités de votre produit autocallable.
    """
    paths = self.simulate_price_paths()
    # Exemple de calcul simplifié - à personnaliser :
    payoff = np.maximum(paths[-1] - self.strike, 0)
    discounted_payoff = np.exp(-self.risk_free_rate * self.maturity) * payoff
    price = np.mean(discounted_payoff)
    return price'''