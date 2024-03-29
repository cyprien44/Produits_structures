import numpy as np

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
        paths = self.monte_carlo
        num_steps = paths.shape[0]
        num_simulations = paths.shape[1]

        for step in range(num_steps):
            current_prices = paths[step, :, :]
            coupon_condition = current_prices >= self.coupon_barrier
            autocall_condition = current_prices >= self.autocall_barrier

            # La barrière de redemption est basée sur le max historique des prix
            max_price_ratios = np.maximum.accumulate(current_prices, axis=0)
            redemption_condition = max_price_ratios <= self.autocall_barrier

            # Pour la dernière étape, payer le nominal si les conditions ne sont pas remplies
            if step == num_steps - 1:
                redemption_condition = np.ones_like(redemption_condition, dtype=bool)

            coupon_payment = self.nominal * self.coupon_rate * coupon_condition * redemption_condition
            redemption_payment = self.nominal * autocall_condition * redemption_condition
            total_payment = coupon_payment + redemption_payment
            
            # Ajouter les paiements actualisés pour cette étape
            self.payoffs.append(total_payment)

            # Actualiser les paiements
            discount = self.discount_factor(step / num_steps)
            discounted_payment = total_payment * discount

            # Ajouter les paiements actualisés pour cette étape
            self.payoffs_discount.append(discounted_payment)

        return np.array(self.payoffs)

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