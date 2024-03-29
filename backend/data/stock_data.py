#récupération spot, dividende, correlation des stocks (correl jsp)

class StockData:
    def __init__(self, spot_price, volatility, dividend_yield=0):
        """
        Initialisation des données du sous-jacent.
        :param spot_price: Prix spot du sous-jacent.
        :param volatility: Volatilité annuelle du sous-jacent.
        :param dividend_yield: Rendement du dividende annuel.
        """
        self.spot_price = spot_price
        self.volatility = volatility
        self.dividend_yield = dividend_yield
