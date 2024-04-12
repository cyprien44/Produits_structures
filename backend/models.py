import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class Models:
    def __init__(self, spot_price, strike, risk_free_rate, maturity, dividend_yield, volatility):
        self.spot_price = spot_price
        self.strike = strike
        self.risk_free_rate = risk_free_rate
        self.maturity = maturity
        self.dividend_yield = dividend_yield
        self.volatility = volatility

    def black_scholes(self, call_or_put='call'):
        d1 = (np.log(self.spot_price / self.strike) + (self.risk_free_rate - self.dividend_yield + 0.5 * self.volatility ** 2) * self.maturity) / (self.volatility * np.sqrt(self.maturity))
        d2 = d1 - self.volatility * np.sqrt(self.maturity)
        if call_or_put == 'call':
            return self.spot_price * np.exp(-self.dividend_yield * self.maturity) * norm.cdf(d1) - self.strike * np.exp(-self.risk_free_rate * self.maturity) * norm.cdf(d2)
        elif call_or_put == 'put':
            return self.strike * np.exp(-self.risk_free_rate * self.maturity) * norm.cdf(-d2) - self.spot_price * np.exp(-self.dividend_yield * self.maturity) * norm.cdf(-d1)

class Autocall:
    def __init__(self, monte_carlo, nominal, coupon_rate, coupon_barrier, autocall_barrier,put_barrier, risk_free):
        self.monte_carlo = monte_carlo
        self.nominal = nominal
        self.coupon_rate = coupon_rate
        self.coupon_barrier = coupon_barrier
        self.autocall_barrier = autocall_barrier
        self.put_barrier = put_barrier
        self.risk_free = risk_free
        self.payoffs, self.payoffs_discount = self.generate_payoffs()
        self.average_price = None
        self.overall_average = None
        self.figs = []  


    def discount_factor(self, step, total_steps):
        time = step / total_steps * self.monte_carlo.maturity  # Convertir en fraction de la maturité totale
        return np.exp(-self.risk_free * time)
    
    def plot_simulations(self):
        self.figs = []  # Initialiser une liste pour stocker les figures si vous avez plusieurs actifs
        
        for actif_index, (df, stock) in enumerate(zip(self.monte_carlo.simulations, self.monte_carlo.stocks)):
            fig, ax = plt.subplots(figsize=(10, 6))

            # Convertir l'index en datetime si ce n'est pas déjà le cas
            df.index = pd.to_datetime(df.index)

            # Tracer chaque simulation pour l'actif courant
            for sim_index in df.columns:
                ax.plot(df.index, df[sim_index], lw=1)

            # Ajouter une ligne horizontale pour la barrière de coupon et d'autocall
            ax.axhline(y=self.coupon_barrier * df.iloc[0,0], color='g', linestyle='--', label=f'Coupon Barrier ({round(self.coupon_barrier*df.iloc[0,0], 1)})')
            ax.axhline(y=self.autocall_barrier * df.iloc[0,0], color='r', linestyle='--', label=f'Autocall Barrier ({round(self.autocall_barrier*df.iloc[0,0], 1)})')
            ax.axhline(y=self.put_barrier * df.iloc[0,0], color='orange', linestyle='--', label=f'Put Barrier ({round(self.put_barrier*df.iloc[0,0], 1)})')

            # Ajouter une ligne verticale pour chaque date d'observation
            for obs_date in self.monte_carlo.observation_dates:
                ax.axvline(x=obs_date, color='lightblue', linestyle='--', linewidth=1, alpha=0.5)

            # Formater l'axe des x pour afficher les dates de manière lisible
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)

            ax.set_title(f'Monte Carlo Simulation for {stock.ticker}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Process Value')
            ax.grid(True, which='both', axis='y', linestyle='--', color='grey')
            ax.grid(False, which='both', axis='x')
            ax.legend()

            plt.tight_layout()
            self.figs.append(fig)  # Ajouter la figure à la liste des figures


    def show_simulations(self):
        # S'assurer que les figures ont été générées
        if not hasattr(self, 'figs') or not self.figs:
            self.plot_simulations()

        for fig in self.figs:
            plt.show()

    def generate_payoffs(self):
        # Initialiser des listes pour stocker les DataFrames des payoffs et des payoffs actualisés pour chaque actif
        payoffs_dataframes = []
        discounted_payoffs_dataframes = []

        # Parcourir chaque actif dans les simulations générées par Monte Carlo
        for actif_index, df in enumerate(self.monte_carlo.simulations):
            # Obtenir le nombre d'étapes et de simulations pour l'actif actuel
            num_steps = len(self.monte_carlo.observation_dates)
            num_simulations = df.shape[1]

            # Initialiser des tableaux pour stocker les payoffs et les payoffs actualisés à chaque étape pour chaque simulation
            payoffs_actif = np.zeros((num_steps, num_simulations))
            discounted_payoffs_actif = np.zeros((num_steps, num_simulations))

            # Parcourir chaque étape de simulation pour l'actif actuel
            for step, time_step in enumerate(self.monte_carlo.observation_dates):
                # Obtenir les prix courants et les prix initiaux pour calculer les ratios de prix
                current_prices = df.loc[time_step].values
                initial_prices = df.iloc[0].values
                price_ratios = current_prices / initial_prices

                # Calculer les conditions de paiement du coupon et d'autocall basées sur les ratios de prix
                coupon_condition = price_ratios >= self.coupon_barrier
                autocall_condition = price_ratios >= self.autocall_barrier

                # S'assurer que le prix n'a jamais dépassé l'autocall barrière dans le passé sinon fin de contrat
                if step > 0:
                    # Filtrer df pour ces dates d'observation
                    filter_df = df.loc[self.monte_carlo.observation_dates[:step]]
                    # Calculer max_price_ratios en se basant sur les prix filtrés
                    max_price_ratios = filter_df.max(axis=0) / initial_prices
                    no_redemption_condition = max_price_ratios <= self.autocall_barrier
                else:
                    # Si nous sommes à la première observation, utiliser les ratios actuels comme max_ratios
                    max_price_ratios = price_ratios
                    no_redemption_condition = True

                # À la dernière étape, s'assurer de payer le nominal si la barrière put n'a pas été franchit et si les conditions d'autocall ne sont pas remplies
                if step == (num_steps - 1):
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
            
            # --------------------------------------------------------------------------------------------------------------------------------
            #Part put barrière
            filter_df = df.loc[self.monte_carlo.observation_dates]
            #Je regarde le plus petit ratio de prix sur toutes les observations dates 
            min_price_ratios = filter_df.min(axis=0) / initial_prices
            #Si le plus petit des ratios est inférieur à la barrière put alors cette barrière a été franchit
            put_condition = min_price_ratios <= self.put_barrier
            final_price_ratios = filter_df.iloc[-1].values / initial_prices

            #Boucle sur toutes les simulations
            for i in range(len(no_redemption_condition)):
                #Si il n'y a pas eu déjà de redemption, que le barrière put a au moins été franchit une fois et que le dernier prix est inférieur au prix initial alors il faut imputer la perte
                if no_redemption_condition[i] and put_condition[i] and (final_price_ratios[i]<1):
                    
                    # J'annule tous les paiements de coupons précédents
                    payoffs_actif[:, i] = discounted_payoffs_actif[:,i] = 0
                    #Inputer la perte sur le dernier payoff
                    payoffs_actif[-1, i] = self.nominal * final_price_ratios[i]

                    discount = self.discount_factor(num_steps, num_steps)
                    discounted_payoffs_actif[-1, i] = (payoffs_actif[-1, i] * discount)
            # --------------------------------------------------------------------------------------------------------------------------------
                    
            # Créer des DataFrames pour les payoffs et les payoffs actualisés et les ajouter aux listes
            df_payoffs = pd.DataFrame(payoffs_actif, index= self.monte_carlo.observation_dates, columns=[f'Simulation {sim+1}' for sim in range(num_simulations)])
            df_discounted_payoffs = pd.DataFrame(discounted_payoffs_actif, index= self.monte_carlo.observation_dates,  columns=[f'Simulation {sim+1}' for sim in range(num_simulations)])
            
            payoffs_dataframes.append(df_payoffs)
            discounted_payoffs_dataframes.append(df_discounted_payoffs)

        # Retourner les listes des DataFrames des payoffs et des payoffs actualisés
        return payoffs_dataframes, discounted_payoffs_dataframes

    def print_payoffs_dataframes(self):
        """
        Affiche les DataFrames des payoffs pour chaque actif, utilisant le nom de l'actif.
        """
        for i, (df, stock) in enumerate(zip(self.payoffs, self.monte_carlo.stocks)):
            print(f"Payoffs DataFrame for {stock.ticker}:")  # Utilisation de `name` comme identifiant
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

        for stock, value in zip(self.monte_carlo.stocks, self.average_price):
            print(f"Prix moyen final pour {stock.ticker}: {value:.2f} €")  # Utilisation de `stock.name`

        print(f"Prix moyen final sur tous les actifs: {self.overall_average:.2f} €")



