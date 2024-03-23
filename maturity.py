from datetime import datetime


class Maturity:
    def __init__(self,
                 maturity_in_years: float = None,
                 begin_date: datetime = None,
                 end_date: datetime = None,
                 day_count_convention: str = "ACT/360") -> None:
        self.__day_count_convention = day_count_convention
        if maturity_in_years != None:
            self.maturity_in_year = maturity_in_years
        else:
            self.maturity_in_year = (end_date - begin_date).days / self.__denom()

    def maturity(self):
        return self.maturity_in_year

    def __denom(self):
        if self.__day_count_convention == "ACT/365":
            return 365.0
        elif self.__day_count_convention == "ACT/360":
            return 360.0

        raise Exception("day_count_convention " + self.__day_count_convention + " error")