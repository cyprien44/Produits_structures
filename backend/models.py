import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Autocall:
    def __init__(self, monte_carlo, nominal, coupon_rate, coupon_barrier, autocall_barrier, risk_free):
        self.monte_carlo = monte_carlo
        self.nominal = nominal
        self.coupon_rate = coupon_rate
        self.coupon_barrier = coupon_barrier
        self.autocall_barrier = autocall_barrier
        self.risk_free = risk_free
        self.payoffs, self.payoffs_discount = self.generate_payoffs()
        self.average_price = None
        self.overall_average = None


    def discount_factor(self, step, total_steps):
        time = step / total_steps * self.monte_carlo.maturity  # Convertir en fraction de la maturité totale
        return np.exp(-self.risk_free * time)
    
    def plot_simulations(self):

        for actif_index, df in enumerate(self.monte_carlo.simulations):
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


    def generate_payoffs(self):
        # Initialiser des listes pour stocker les DataFrames des payoffs et des payoffs actualisés pour chaque actif
        payoffs_dataframes = []
        discounted_payoffs_dataframes = []

        # Parcourir chaque actif dans les simulations générées par Monte Carlo
        for actif_index, df in enumerate(self.monte_carlo.simulations):
            # Obtenir le nombre d'étapes et de simulations pour l'actif actuel
            num_steps = df.shape[0]
            num_simulations = df.shape[1]

            # Initialiser des tableaux pour stocker les payoffs et les payoffs actualisés à chaque étape pour chaque simulation
            payoffs_actif = np.zeros((num_steps, num_simulations))
            discounted_payoffs_actif = np.zeros((num_steps, num_simulations))

            # Parcourir chaque étape de simulation pour l'actif actuel
            for step, time_step in enumerate(df.index):
                # Obtenir les prix courants et les prix initiaux pour calculer les ratios de prix
                current_prices = df.iloc[step].values
                initial_prices = df.iloc[0].values
                price_ratios = current_prices / initial_prices

                # Calculer les conditions de paiement du coupon et d'autocall basées sur les ratios de prix
                coupon_condition = price_ratios >= self.coupon_barrier
                autocall_condition = price_ratios >= self.autocall_barrier

                # S'assurer que le prix n'a jamais dépassé l'autocall barrière dans le passé sinon fin de contrat
                max_price_ratios = df.iloc[:step].max(axis=0) / initial_prices
                no_redemption_condition = max_price_ratios <= self.autocall_barrier

                # À la dernière étape, s'assurer de payer le nominal si les conditions d'autocall ne sont pas remplies
                if step == (len(df.index) - 1):
                    for i in range(len(no_redemption_condition)):
                        if bool(no_redemption_condition[i]):
                            no_redemption_condition[i] = autocall_condition[i] = True

                # Calculer les paiements de coupon et de rachat, puis le paiement total pour chaque simulation
                coupon_payment = self.nominal * self.coupon_rate * coupon_condition * no_redemption_condition
                redemption_payment = self.nominal * autocall_condition * no_redemption_condition
                total_payment = coupon_payment + redemption_payment

                # Stocker le paiement total et le paiement total actualisé à l'étape courante
                payoffs_actif[step, :] = total_payment
                discount = self.discount_factor(step, num_steps)
                discounted_payoffs_actif[step, :] = (total_payment * discount)

            # Créer des DataFrames pour les payoffs et les payoffs actualisés et les ajouter aux listes
            df_payoffs = pd.DataFrame(payoffs_actif, index=df.index, columns=[f'Simulation {sim+1}' for sim in range(num_simulations)])
            df_discounted_payoffs = pd.DataFrame(discounted_payoffs_actif, index=df.index, columns=[f'Simulation {sim+1}' for sim in range(num_simulations)])
            
            payoffs_dataframes.append(df_payoffs)
            discounted_payoffs_dataframes.append(df_discounted_payoffs)

        # Retourner les listes des DataFrames des payoffs et des payoffs actualisés
        return payoffs_dataframes, discounted_payoffs_dataframes

    def print_payoffs_dataframes(self):
        """
        Affiche les DataFrames des payoffs pour chaque actif.
        """
        for i, df in enumerate(self.payoffs):
            print(f"Payoffs DataFrame for Asset {i+1}:")
            print(df)

    def calculate_average_present_value(self):
        """Calcule la valeur présente moyenne pour chaque actif et la moyenne globale."""
        price_by_simul = []
        average_price = []
        for df in self.payoffs_discount:
            # Calculer la somme des valeurs actualisées pour obtenir le total actualisé par simulation
            total_discounted = df.sum(axis=0)  # Sum along rows to get the sum for each simulation
            # Ajouter la somme des paiements actualisés pour cette DataFrame à la liste
            price_by_simul.append(total_discounted)
            average_price.append(total_discounted.mean())  # Calculate the mean across all simulations for the current asset

        self.average_price = average_price
        # Calculer la moyenne globale sur tous les actifs
        self.overall_average = np.mean(average_price)

    def print_average_present_values(self):
        """Affiche les valeurs présentes moyennes calculées pour chaque actif et la moyenne globale."""
        if self.average_price is None or self.overall_average is None:
            self.calculate_average_present_value()

        # Imprimer les résultats de manière plus claire
        for i, value in enumerate(self.average_price):
            print(f"Prix moyen final pour l'actif {i+1}: {value:.2f}")
        
        print(f"Prix moyen final sur tous les actifs: {self.overall_average:.2f}")



