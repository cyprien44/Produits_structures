import pandas as pd
import numpy as np
import yfinance as yf
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class ZeroCouponCurve:
    def __init__(self, tickers):
        """
        Initialisation avec les tickers des bons du Trésor américain pour différentes maturités.
        """
        self.tickers = {self.ticker_to_years(ticker): symbol for ticker, symbol in tickers.items()}

    @staticmethod
    def ticker_to_years(ticker):
        if 'W' in ticker:
            # Convertir les semaines en années
            return int(ticker.replace('W', '')) / 52
        elif 'M' in ticker:
            # Convertir les mois en années
            return int(ticker.replace('M', '')) / 12
        elif 'Y' in ticker:
            # Convertir les années en années
            return int(ticker.replace('Y', ''))
        else:
            raise ValueError(f"Le format du ticker {ticker} n'est pas reconnu.")

    def fetch_data(self):
        """
        Récupère les taux d'intérêt pour les bons du Trésor américain.
        """
        data = yf.download(list(self.tickers.values()), period="1d")['Adj Close']
        data.columns = self.tickers.keys()  # Renommer les colonnes pour correspondre aux maturités
        return data

    def build_curve(self):
        """
        Construit la courbe des taux zéro coupon à partir des taux d'intérêt des bons du Trésor.
        """
        rates = self.fetch_data()
        # Convertir les taux en base annuelle et calculer les taux zéro coupon comme exemple simplifié
        # Dans la pratique, cette conversion dépendrait de la spécificité des instruments et de leur prix
        zero_coupon_curve = rates / 100
        return zero_coupon_curve

    def plot_curve(self):
        """
        Trace la courbe des taux zéro coupon en utilisant uniquement les points originaux.
        """
        zero_coupon_curve = self.build_curve()

        plt.figure(figsize=(10, 6))

        # Loop over each column in the DataFrame
        plt.plot(zero_coupon_curve.columns, zero_coupon_curve.values[0], 'o-', label='Taux')

        plt.title('Zero Coupon Curve Over Time')
        plt.xlabel('Date')
        plt.ylabel('Yield (%)')
        plt.legend()
        plt.show()