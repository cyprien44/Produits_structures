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
        self.payoffs = []
        self.payoffs_discount = []

    def discount_factor(self, time_step):
        return np.exp(-self.risk_free * time_step)

    def generate_payoffs(self):
        paths = self.monte_carlo.simulations
        num_steps = paths.shape[0]
        num_simulations = paths.shape[1]
        num_actifs = paths.shape[2]

        payoffs_dataframes = []
        discounted_payoffs_dataframes = []

        for actif_index in range(num_actifs):
            payoffs_actif = np.zeros((num_steps, num_simulations))
            discounted_payoffs_actif = np.zeros((num_steps, num_simulations))

            for step in range(num_steps):
                current_prices = paths[step, :, actif_index]
                price_ratios = current_prices / paths[0, :, actif_index]

                coupon_condition = price_ratios >= self.coupon_barrier
                autocall_condition = price_ratios >= self.autocall_barrier
                max_price_ratios = np.maximum.accumulate(price_ratios, axis=0)
                redemption_condition = max_price_ratios <= self.autocall_barrier

                if step == num_steps - 1:
                    redemption_condition = np.ones_like(redemption_condition, dtype=bool)

                coupon_payment = self.nominal * self.coupon_rate * coupon_condition * redemption_condition
                redemption_payment = self.nominal * autocall_condition * redemption_condition
                total_payment = coupon_payment + redemption_payment

                payoffs_actif[step, :] = total_payment
                discount = self.discount_factor(step / num_steps)
                discounted_payoffs_actif[step, :] = total_payment * discount

            columns = [f'Simulation {i+1}' for i in range(num_simulations)]
            df_payoffs = pd.DataFrame(payoffs_actif, index=[f'Step {step+1}' for step in range(num_steps)], columns=columns)
            df_discounted_payoffs = pd.DataFrame(discounted_payoffs_actif, index=[f'Step {step+1}' for step in range(num_steps)], columns=columns)
            payoffs_dataframes.append(df_payoffs)
            discounted_payoffs_dataframes.append(df_discounted_payoffs)

        self.payoffs = payoffs_dataframes
        self.payoffs_discount = discounted_payoffs_dataframes

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