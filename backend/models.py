# formules des payoffs etc


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
    return price