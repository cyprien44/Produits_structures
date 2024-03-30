from scipy.optimize import fsolve
from datetime import datetime
import yfinance as yf
import pandas as pd
from backend.models import Models

class Volatility:
    def __init__(self, stock, maturity_date, rate, dividend_yield):
        self.stock = stock
        self.maturity_date = maturity_date
        self.rate = rate
        self.dividend_yield = dividend_yield
        self.implied_volatility = self.implied_volatility()

    def implied_volatility(self):
        option_chain = yf.Ticker(self.stock.ticker).option_chain(self.maturity_date)
        calls, puts = self.filter_moneyness(option_chain.calls, option_chain.puts)

        r = self.get_rate_for_maturity()
        d = self.dividend_yield

        for index, row in calls.iterrows():
            S = self.stock.spot_price
            K = row['strike']
            T = (datetime.strptime(self.maturity_date, "%Y-%m-%d") - datetime.now()).days / 365.0
            market_price = row['lastPrice']

            f = lambda sigma: Models(S, K, T, r, d, sigma).black_scholes_call() - market_price
            implied_vol = fsolve(f, 0.2)
            calls.at[index, 'implied_Volatility'] = implied_vol[0]

        for index, row in puts.iterrows():
            S = self.stock.spot_price
            K = row['strike']
            T = (datetime.strptime(self.maturity_date, "%Y-%m-%d") - datetime.now()).days / 365.0
            market_price = row['lastPrice']

            f = lambda sigma: Models(S, K, T, r, d, sigma).black_scholes_put() - market_price
            implied_vol = fsolve(f, 0.2)
            puts.at[index, 'implied_Volatility'] = implied_vol[0]

        return pd.concat([calls, puts])

    def filter_moneyness(self, calls, puts, moneyness_range=[0.6, 1.6]):
        calls = calls[(calls['strike'] > self.stock.spot_price) & (calls['strike'] < moneyness_range[1] * self.stock.spot_price)]
        puts = puts[(puts['strike'] > moneyness_range[0] * self.stock.spot_price) & (puts['strike'] < self.stock.spot_price)]
        return calls, puts

    def get_rate_for_maturity(self):
        # Convertir la date de maturité en timestamp
        maturity_timestamp = pd.to_datetime(self.maturity_date).timestamp()

        # Définir l'index du DataFrame comme des timestamps
        self.rate.index = pd.to_datetime(self.rate.index).timestamp()

        # Interpoler les taux
        interpolated_data = self.rate.interpolate(method='linear')

        # Récupérer le taux pour la date de maturité
        rate_for_maturity = interpolated_data.loc[maturity_timestamp]

        return rate_for_maturity