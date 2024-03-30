import pandas as pd
import re
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class ZeroCouponCurve:
    def __init__(self, date='20240301'):
        """
        Initialisation avec les tickers des bons du Trésor américain pour différentes maturités.
        """
        self.date = date
        self.data = self.get_data_from_json()

    def get_data_from_json(self):
        """
        Récupère les données de Bloomberg pour les bons du Trésor américain.
        """
        df = pd.read_json('backend/data/rate.json')

        start_date = datetime.strptime(self.date, '%Y%m%d')

        new_columns = []

        for col in df.columns:
            # Extraire la partie numérique et l'unité de temps
            match = re.match(r"\('S0023Z (\d+)([DWMY]) BLC2 Curncy', 'Last_Price'\)", col)
            if match:
                number, unit = match.groups()
                number = int(number)

                # Calculer la nouvelle date en fonction de l'unité de temps
                if unit == 'D':
                    new_date = start_date + timedelta(days=number)
                elif unit == 'W':
                    new_date = start_date + timedelta(weeks=number)
                elif unit == 'M':
                    new_date = start_date + pd.DateOffset(months=number)
                elif unit == 'Y':
                    new_date = start_date + pd.DateOffset(years=number)

                new_columns.append(new_date)
            else:
                new_columns.append(col)

        # Remplacer les colonnes du DataFrame
        df.columns = new_columns

        return df/100

    def plot_rate_curve(self):
        """
        Tracer la courbe des taux.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.data.columns, self.data.values[0])
        plt.title('Courbe des taux')
        plt.xlabel('Date')
        plt.ylabel('Taux')
        plt.show()
