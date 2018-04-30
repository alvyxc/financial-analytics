import Util
import MPT
import numpy as np


class DataPathConfig:
    spy_500_list_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    spy_500_csv_file = "../data/spy500_list.cvs"
    local_stocks_data_path = "../data/"
    spy_500_filtered_csv_file = "../data/filtered_spy500_list.csv"
    data_source = "morningstar"


class MarketStock:
    data_path = None
    market_index = "SPY"
    market_stock_df = None
    mean_return = 0
    return_stdev = 0
    annual_return = 0
    annual_stdev = 0
    monthly_return = 0
    monthly_stdev = 0
    monthly_closing = 0
    annual_closing = 0
    start_date = ""
    end_date = ""
    risk_free_rate = 0.03

    def __init__(self):
        self.data_path = DataPathConfig()
        self.market_stock_df = Util.read_stock(self.data_path, self.market_index)

    def update(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        close_tag = ConfigGlobal.close_tag
        start_date = Util.add_date(start_date, -5)
        data = Util.filter_stock_df(self.market_stock_df, start_date, end_date, close_tag)
        returns = data.pct_change()
        returns = returns.iloc[1:]
        self.mean_return = returns.mean()
        self.return_stdev = returns.std()

        self.annual_return = round(self.mean_return * 252, 2)
        self.annual_stdev = round(self.return_stdev * np.sqrt(252), 2)
        self.monthly_return = round(self.mean_return * 21, 2)
        self.monthly_stdev = round(self.return_stdev * np.sqrt(21), 2)

        self.annual_closing = returns.groupby(returns.index.get_level_values('Date').year) \
            .apply(Util.total_return_from_returns)

        self.monthly_closing = \
            returns.groupby([returns.index.get_level_values('Date').year, returns.index.get_level_values('Date').month]) \
                .apply(Util.total_return_from_returns)

    def get_closing_price(self, closing_date):
        closing_price_record = self.market_stock_df.loc[(self.market_stock_df.index.get_level_values('Date') \
                                            == closing_date)][ConfigGlobal.close_tag]
        adjusted_closing_date = closing_date
        while closing_price_record.empty:
            adjusted_closing_date = Util.add_date(adjusted_closing_date, -1)
            closing_price_record = self.market_stock_df.loc[(self.market_stock_df.index.get_level_values('Date') \
                                            == adjusted_closing_date)][ConfigGlobal.close_tag]
        closing_price = closing_price_record[0]
        return closing_price


class ConfigGlobal:
    close_tag = "Close"
    market_stock = None
    spy_500_list = []
    start_date = '01/01/2016'
    end_date = '04/01/2018'
    data_path = DataPathConfig()

    def __init__(self):
        self.spy_500_list = Util.get_spy500_list(self.data_path.spy_500_filtered_csv_file)
        self.market_stock = MarketStock()

class TradeAction:
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"