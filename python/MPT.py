import pandas_datareader.data as web
import numpy as np
import pandas as pd
import Common
import Util
import matplotlib.pyplot as plt


def get_stock_stat(stock, start_date, end_date, config):

    start_date = Util.add_date(start_date, -5)
    data = web.DataReader(stock, data_source=config.data_source, start=start_date,\
                          end=end_date)[config.close_tag]
    data.sort_index(inplace = True)
    returns = data.pct_change()
    returns = returns.iloc[1:]
    stock_stat = Common.StockStat()
    stock_stat.mean_return = returns.mean()
    stock_stat.return_stdev = returns.std()

    stock_stat.annual_return = round(stock_stat.mean_return * 252, 2)
    stock_stat.annual_stdev = round(stock_stat.return_stdev * np.sqrt(252), 2)
    stock_stat.monthly_return = round(stock_stat.mean_return * 21, 2)
    stock_stat.monthly_stdev = round(stock_stat.return_stdev * np.sqrt(21), 2)

    stock_stat.annual_closing = returns.groupby(returns.index.get_level_values('Date').year) \
        .apply(Util.total_return_from_returns)

    stock_stat.monthly_closing = \
        returns.groupby([returns.index.get_level_values('Date').year, returns.index.get_level_values('Date').month]) \
        .apply(Util.total_return_from_returns)

    return stock_stat


def get_beta(stock_closing, index_closing):
    normalize_index_closing = index_closing
    if len(stock_closing.index) == 0:
        print 'No data from this stock to calculate beta'
        return 0

    if len(stock_closing.index) < len(index_closing.index):
        print 'Not enough historic data, truncate date to calculate beta'
        norm_len = len(index_closing.index) - len(stock_closing.index)
        normalize_index_closing = index_closing.iloc[norm_len:]

    cov_val = np.cov(stock_closing, normalize_index_closing)[0][1]
    var_val = np.var(index_closing)
    beta = cov_val / var_val
    return beta


class StockStat:
    data_path = None
    stock_name = ""
    stock_df = None
    market_stock = None
    daily_return = 0
    mean_return = 0
    return_stdev = 0
    annual_return = 0
    annual_stdev = 0
    monthly_return = 0
    monthly_stdev = 0
    monthly_closing = 0
    annual_closing = 0
    beta = 0
    capm_expected_return = 0
    start_date = None
    end_date = None
    annual_camp_ratio = 0.0
    global_config = None

    def __init__(self, stock_name, market_stock, global_config):
        self.data_path = Common.DataPathConfig()
        self.stock_name = stock_name
        self.market_stock = market_stock
        self.stock_df = Util.read_stock(self.data_path, stock_name)
        self.global_config = global_config

    def __lt__(self, other):
        return self.annual_camp_ratio < other.annual_camp_ratio

    def __str__(self):
        return "mean_return = " + str(self.mean_return) + " return_stdev = " + str(self.return_stdev) + \
               " annual_return = " + str(self.annual_return) + " annual_stdev = " + str(self.annual_stdev)

    def update(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date
        close_tag = Common.ConfigGlobal.close_tag
        start_date = Util.add_date(start_date, -5)
        data = Util.filter_stock_df(self.stock_df, start_date, end_date, close_tag)
        returns = data.pct_change()
        returns = returns.iloc[1:]

        self.daily_return = returns
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

        self.beta = get_beta(self.monthly_closing, self.market_stock.monthly_closing)
        self.capm_expected_return = self.market_stock.risk_free_rate \
                     + self.beta*(self.market_stock.annual_return - self.market_stock.risk_free_rate)
        self.annual_camp_ratio = self.annual_return / abs(self.capm_expected_return) * 1.0

    def get_closing_price(self, closing_date):
        closing_price_record = self.stock_df.loc[(self.stock_df.index.get_level_values('Date') \
                                         == closing_date)][self.global_config.close_tag]
        adjusted_closing_date = closing_date
        while closing_price_record.empty:
            adjusted_closing_date = Util.add_date(adjusted_closing_date, -1)
            closing_price_record = self.stock_df.loc[(self.stock_df.index.get_level_values('Date') \
                                    == adjusted_closing_date)][self.global_config.close_tag]
        closing_price = closing_price_record[0]
        return closing_price


class ProfilioStock:
    stock_stat = None
    buy_in_date = ""
    num_share = 0

    def __init__(self, stock_stat, buy_in_date, num_share):
        self.stock_stat = stock_stat
        self.buy_in_date = buy_in_date
        self.num_share = num_share

class PorfilioStat:
    stock_stats = []
    weights = []
    capm_expected_return = 0
    porfolio_beta = 0
    market_stock = None
    cash_balance = 0
    initial_investment_date = ""
    max_num_stocks = 0
    porfolio_market_df = None
    initial_capital = 0

    def __init__(self, market_stock, capital, initial_date, stocks_in_porfolio):
        self.market_stock = market_stock
        self.initial_capital = capital
        self.cash_balance = capital
        self.initial_investment_date = initial_date
        self.max_num_stocks = stocks_in_porfolio
        self.porfolio_market_df = pd.DataFrame(columns = ['Date', 'p_return', 'm_return'])

    def add_stocks(self, stock):
        s = ProfilioStock(stock, self.initial_investment_date, 0)
        self.stock_stats.append(s)

    def update(self, start_date, end_date):
        p_beta = 0
        p_annual_return = 0
        self.weights = np.repeat(1.0/len(self.stock_stats), len(self.stock_stats))
        self.market_stock.update(start_date, end_date)
        for stat in self.stock_stats:
            s = stat.stock_stat
            s.update(start_date, end_date)
            p_beta += s.beta
            p_annual_return += s.annual_return

        self.porfolio_beta = p_beta / len(self.stock_stats)
        self.annual_return = p_annual_return / len(self.stock_stats)
        self.capm_expected_return = self.market_stock.risk_free_rate \
                                    + self.porfolio_beta * (self.market_stock.annual_return \
                                                            - self.market_stock.risk_free_rate)

    def initialize_investment(self):
        current_date = self.initial_investment_date
        fund_per_stock = self.cash_balance / len(self.stock_stats)
        for stat in self.stock_stats:
            s = stat.stock_stat
            price = s.get_closing_price(current_date)
            shares = fund_per_stock // price
            stat.num_share = shares
            stat.buy_in_date = current_date
            self.cash_balance -= (shares * price)

    def get_porfolio_value(self, trading_date, output):
        if output:
            print ("Stock,  Shares,  Buy Price,  Current Price")
        total_porfilio_value = 0
        for stat in self.stock_stats:
            s = stat.stock_stat
            price_org = s.get_closing_price(stat.buy_in_date)
            price_cur = s.get_closing_price(trading_date)
            gain = price_cur * stat.num_share - price_org * stat.num_share
            if output:
                print("%s,    %d,    %f,    %f" \
                    % (s.stock_name, stat.num_share, price_org,  price_cur))
            total_porfilio_value += (price_cur * stat.num_share)

        total_porfilio_value += self.cash_balance
        if output:
            print ("Cash balance %f" % self.cash_balance)
            print ("Total Porfolio value %s" % total_porfilio_value)
        return total_porfilio_value


    def print_porfolio_capm(self):
        print ("Porfolio Beta %f" % self.porfolio_beta)
        print ("Porfolio annual return %f" % self.annual_return)
        print ("Porfolio capm %f" % self.capm_expected_return)

    def plot_porfolio_ef(self):
        stock_len = len(self.stock_stats)
        annual_return_cov_matrix = [[0 for x in range(stock_len)] for y in range(stock_len)]
        i, j = 0, 0
        for i in range(0, stock_len):
            for j in range(0, stock_len):
                annual_return_cov_matrix[i][j] = Util.get_cov(self.stock_stats[i].stock_stat.monthly_closing, \
                                                           self.stock_stats[j].stock_stat.monthly_closing) * 12

        annual_return_array = []
        for stat in self.stock_stats:
            annual_return_array.append(stat.stock_stat.annual_return)

        num_porfolio = 10000

        port_returns = []
        port_volatility = []
        sharpe_ratio = []
        stock_weights = []
        for p in range(num_porfolio):
            weights = np.random.random(stock_len)
            weights /= np.sum(weights)
            returns = np.dot(weights, annual_return_array)
            volatility = np.sqrt(np.dot(weights.T, np.dot(annual_return_cov_matrix, weights)))
            sharpe = returns / volatility
            sharpe_ratio.append(sharpe)
            port_returns.append(returns)
            port_volatility.append(volatility)
            stock_weights.append(weights)

        portfolio = {'Returns': port_returns,
                     'Volatility': port_volatility,
                     'Sharpe Ratio': sharpe_ratio}

        df = pd.DataFrame(portfolio)
        min_volatility = df['Volatility'].min()
        min_variance_port = df.loc[df['Volatility'] == min_volatility]

        my_volatility = np.sqrt(np.dot(self.weights.T, np.dot(annual_return_cov_matrix, self.weights)))

        df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                        cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
        plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue', marker='D', s=200)
        plt.scatter(x=my_volatility, y=self.annual_return, c='red', marker='D', s=200)
        plt.xlabel('Volatility (Std. Deviation)')
        plt.ylabel('Expected Returns')
        plt.title('Efficient Frontier')
        plt.show()
        plt.show(block=True)
        plt.interactive(False)

    def sell_stock(self, trading_date, stock_to_sell):
        stock = stock_to_sell.stock_stat
        price = stock.get_closing_price(trading_date)
        self.cash_balance += (price * stock_to_sell.num_share)
        self.stock_stats.remove(stock_to_sell)

    def buy_stock(self, trading_date, stock_to_buy):
        num_stock_allowed = self.max_num_stocks - len(self.stock_stats)
        if num_stock_allowed <= 0:
            print("Error: max number if porfolio reached.")
        fund_per_stock = self.cash_balance / num_stock_allowed
        price = stock_to_buy.get_closing_price(trading_date)
        shares = fund_per_stock // price
        p_stock = ProfilioStock(stock_to_buy, trading_date, shares)
        self.stock_stats.append(p_stock)
        self.cash_balance -= (shares * price)

    def update_porf_return(self, trading_date):
        p_value = self.get_porfolio_value(trading_date, False)
        p_return = (p_value - self.initial_capital) * 1.0 / self.initial_capital
        m_initial_value = self.market_stock.get_closing_price(self.initial_investment_date)
        m_value = self.market_stock.get_closing_price(trading_date)
        m_return = (m_value - m_initial_value) * 1.0 / m_initial_value
        row = [trading_date, p_return, m_return]
        self.porfolio_market_df.loc[len(self.porfolio_market_df)] = row

    def plot_porfolio_return(self):
        date_col = pd.to_datetime(self.porfolio_market_df['Date'], format="%m/%d/%Y")
        p_return_col = self.porfolio_market_df['p_return'] * 100.0
        plt.plot(date_col, p_return_col)
        plt.title('Portfolio Return')
        plt.ylabel('Return %')
        plt.show()
        plt.show(block=True)
        plt.interactive(False)

    def plot_porfolio_vs_market_return(self, title):
        date_col = pd.to_datetime(self.porfolio_market_df['Date'], format="%m/%d/%Y")
        p_return_col = self.porfolio_market_df['p_return'] * 100.0
        m_return_col = self.porfolio_market_df['m_return'] * 100.0
        plt.plot(date_col, p_return_col, color='red', label='Porfolio')
        plt.plot(date_col, m_return_col, color='blue', label='Market')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
        title_s = 'Portfolio Return: ' + title
        plt.title(title_s)
        plt.ylabel('Return %')
        plt.show()
        plt.show(block=True)
        plt.interactive(False)

