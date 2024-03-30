import yfinance as yf
from scipy.optimize import fsolve
from backend.models import Models
from datetime import datetime
import pandas as pd

class StockData:
    def __init__(self, spot_price, strike, ticker=None, maturity_date=None):
        """
        Initialisation des données du sous-jacent.
        :param ticker: Ticker du sous-jacent.
        :param spot_price: Prix spot du sous-jacent.
        :param strike: Prix d'exercice du sous-jacent.
        :param dividend_yield: Rendement du dividende annuel.
        :param maturity_date: Date de maturité du sous-jacent.
        :param volatility: Volatilité implicite du sous-jacent.
        """
        self.ticker = ticker
        self.spot_price = spot_price
        self.strike = strike
        self.dividend_yield = self.get_dividend_yield()
        self.maturity_date = maturity_date
        self.volatility = self.implied_volatility()

    def get_dividend_yield(self):
        stock = yf.Ticker(self.ticker)
        dividend_yield = stock.info['dividendYield']
        return dividend_yield

    def implied_volatility(self):
        option_chain = yf.Ticker(self.ticker).option_chain(self.maturity_date)
        calls, puts = self.filter_moneyness(option_chain.calls, option_chain.puts)

        # Fixe les taux et dividendes pour l'instant
        r = 0.01  # Taux sans risque
        d = 0.02  # Dividende

        # Calcul de la volatilité implicite pour les calls
        for index, row in calls.iterrows():
            S = self.spot_price
            K = row['strike']
            T = (datetime.strptime(self.maturity_date, "%Y-%m-%d") - datetime.now()).days / 365.0
            market_price = row['lastPrice']

            # Fonction à résoudre
            f = lambda sigma: Models(S, K, T, r, d, sigma).black_scholes_call() - market_price

            # Utilisation de fsolve pour trouver la volatilité implicite
            implied_vol = fsolve(f, 0.2)  # 0.2 est une estimation initiale
            calls.at[index, 'implied_Volatility'] = implied_vol[0]

        # Calcul de la volatilité implicite pour les puts
        for index, row in puts.iterrows():
            S = self.spot_price
            K = row['strike']
            T = (datetime.strptime(self.maturity_date, "%Y-%m-%d") - datetime.now()).days / 365.0
            market_price = row['lastPrice']

            # Fonction à résoudre
            f = lambda sigma: Models(S, K, T, r, d, sigma).black_scholes_put() - market_price

            # Utilisation de fsolve pour trouver la volatilité implicite
            implied_vol = fsolve(f, 0.2)  # 0.2 est une estimation initiale
            puts.at[index, 'implied_Volatility'] = implied_vol[0]

        return pd.concat([calls, puts])

    def filter_moneyness(self, calls, puts, moneyness_range=[0.6, 1.6]):
        calls = calls[(calls['strike'] > self.spot_price) & (calls['strike'] < moneyness_range[1] * self.spot_price)]
        puts = puts[(puts['strike'] > moneyness_range[0] * self.spot_price) & (puts['strike'] < self.spot_price)]
        return calls, puts
