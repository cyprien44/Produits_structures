import numpy as np
from scipy.stats import norm

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