from scipy.optimize import fsolve
from datetime import datetime
import yfinance as yf
import pandas as pd
from backend.models import Models
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.dates

class Volatility:
    def __init__(self, stock, pricing_date, rate):
        self.spot_price = stock.spot_price
        self.yf_tickers = stock.ticker.split()[0]
        self.pricing_date = pricing_date
        self.rate = rate
        self.dividend_yield = stock.dividend_yield
        self.volatility_surface = self.calculate_volatility_surface()

    def calculate_volatility_surface(self):
        option_chain_dates = yf.Ticker(self.yf_tickers).options
        volatility_surface = pd.DataFrame()

        for date in option_chain_dates:
            option_chain = yf.Ticker(self.yf_tickers).option_chain(date)
            calls, puts = self.filter_moneyness(option_chain.calls, option_chain.puts)

            # Add the expiry date to the calls and puts dataframes
            calls['expiry_date'] = date
            puts['expiry_date'] = date

            r = self.rate.interpolate_rate(date=datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d"))
            d = self.dividend_yield

            def error_function(volatility, market_price, strike, time_to_maturity, risk_free_rate, dividend_yield,
                               spot_price, option_type):
                model = Models(spot_price, strike, risk_free_rate, time_to_maturity, dividend_yield, volatility)
                if option_type == 'call':
                    model_price = model.black_scholes('call')
                else:
                    model_price = model.black_scholes('put')
                error = model_price - market_price
                return error

            # Calculate implied volatility for calls and puts
            calls['implied_Volatility'] = calls.apply(lambda row: fsolve(error_function, 0.2, args=(row['lastPrice'], row['strike'], (datetime.strptime(date, "%Y-%m-%d") - datetime.strptime(self.pricing_date,"%Y%m%d")).days / 365.0, r, d,self.spot_price, 'call'))[0], axis=1)
            puts['implied_Volatility'] = puts.apply(lambda row: fsolve(error_function, 0.2, args=(row['lastPrice'], row['strike'], (datetime.strptime(date, "%Y-%m-%d") - datetime.strptime(self.pricing_date,"%Y%m%d")).days / 365.0, r, d,self.spot_price, 'put'))[0], axis=1)

            # Concatenate calls and puts and add to the volatility surface DataFrame
            volatility_surface = pd.concat([volatility_surface, pd.concat([puts, calls])])

            def date_to_years(date):
                return (datetime.strptime(date, "%Y-%m-%d") - datetime.strptime(self.pricing_date,
                                                                                "%Y%m%d")).days / 365.0

            volatility_surface['dates_in_years'] = volatility_surface['expiry_date'].apply(date_to_years)

        return volatility_surface

    def filter_moneyness(self, calls, puts, moneyness_range=[0.6, 1.6], volume_threshold=10):
        calls = calls[(calls['strike'] > self.spot_price) &
                      (calls['strike'] < moneyness_range[1] * self.spot_price) &
                      (calls['volume'] > volume_threshold)]
        puts = puts[(puts['strike'] > moneyness_range[0] * self.spot_price) &
                    (puts['strike'] < self.spot_price) &
                    (puts['volume'] > volume_threshold)]
        return calls, puts

    def plot_volatility_surface(self):
        # Create a new figure
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        # Convert dates to datetime and then to numerical format for plotting
        self.volatility_surface['expiry_date'] = pd.to_datetime(self.volatility_surface['expiry_date'])
        self.volatility_surface['date_num'] = matplotlib.dates.date2num(self.volatility_surface['expiry_date'])
        # Create a surface plot
        ax.plot_trisurf(self.volatility_surface['date_num'], self.volatility_surface['strike'],
                        self.volatility_surface['implied_Volatility'], cmap=cm.coolwarm, linewidth=0.2)
        # Format the x-axis to display dates
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))
        # Set labels
        ax.set_xlabel('Date')
        ax.set_ylabel('Strike Price')
        ax.set_zlabel('Implied Volatility')
        # Set title
        ax.set_title('Implied Volatility Surface')
        # Show the plot
        plt.show()
