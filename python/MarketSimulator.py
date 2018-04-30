import MPT
import Common
import Util
import MovingAverageModel
import LinearRegressionModel
import SVRModel
import pandas as pd
import numpy as np
from datetime import datetime as dt
from sklearn.metrics import accuracy_score, f1_score, precision_score

class MarketSimulator:
    global_config = None
    market_stock = None
    stock_porfolio = None
    max_num_stocks_in_porfolio = 25
    min_expected_return = 0
    max_risk = 0
    historic_data_start_date = "01/01/2010"
    historic_data_end_date = "01/01/2014"
    simulation_start_date = "01/01/2014"
    simulator_end_date = "04/01/2018"
    global_stocks_stat = []
    initial_fund = 500000
    current_trade_date = "01/01/2014"
    stocks_on_market = []

    def __init__(self, global_config, market_stock):
        self.global_config = global_config
        self.market_stock = market_stock
        self.market_stock.update(self.historic_data_start_date, self.historic_data_end_date)
        self.stock_porfolio = MPT.PorfilioStat(market_stock, self.initial_fund, self.simulation_start_date, \
                                               self.max_num_stocks_in_porfolio)
        num = 0
        for stock_name in global_config.spy_500_list:
            s = MPT.StockStat(stock_name, self.market_stock, global_config)
            s.update(self.historic_data_start_date, self.historic_data_end_date)
            self.global_stocks_stat.append(s)
            num += 1
            #if num >= 30:
            #    break
        self.global_stocks_stat.sort(reverse=True)

    def initialize_portfolio(self):
        num_stock = 0
        for sa in self.global_stocks_stat:
            if num_stock < self.max_num_stocks_in_porfolio:
                self.stock_porfolio.add_stocks(sa)
            else:
                self.stocks_on_market.append(sa)
            num_stock += 1

        self.stocks_on_market.sort(reverse=True)
        self.stock_porfolio.update(self.historic_data_start_date, self.historic_data_end_date)
        self.stock_porfolio.initialize_investment()

    def start_simulation(self, model, date_increment):
        current_date = self.simulation_start_date
        end_date = self.simulator_end_date
        stock = self.stock_porfolio.stock_stats[0].stock_stat
        while dt.strptime(current_date, "%m/%d/%Y") <= dt.strptime(self.simulator_end_date, "%m/%d/%Y"):
        #while dt.strptime(current_date, "%m/%d/%Y") <= dt.strptime("06/01/2014", "%m/%d/%Y"):
            stocks_to_sell = []
            for i in range(len(self.stock_porfolio.stock_stats)):
                p_stock = self.stock_porfolio.stock_stats[i]
                stock = self.stock_porfolio.stock_stats[i].stock_stat
                action = model.get_trade_action(current_date, stock)
                if action == Common.TradeAction.SELL:
                    print("Selling stock %s on date: %s" % (stock.stock_name, current_date))
                    stocks_to_sell.append(p_stock)

            for ps in stocks_to_sell:
                self.stocks_on_market.append(ps.stock_stat)
                self.stock_porfolio.sell_stock(current_date, ps)

            self.stocks_on_market.sort(reverse=True)
            i = 0
            stock_to_remove_from_market = []
            while i < len(self.stocks_on_market) and \
                    len(self.stock_porfolio.stock_stats) < self.max_num_stocks_in_porfolio:
                stock = self.stocks_on_market[i]
                action = model.get_trade_action(current_date, stock)
                if action == Common.TradeAction.BUY:
                    print("Buying stock %s on date: %s" % (stock.stock_name, current_date))
                    self.stock_porfolio.buy_stock(current_date, stock)
                    stock_to_remove_from_market.append(stock)
                i += 1

            for s in stock_to_remove_from_market:
                self.stocks_on_market.remove(s)

            self.stock_porfolio.update_porf_return(current_date)
            current_date = Util.add_date(current_date, date_increment)


    def print_porfilio_value(self, trading_date):
        self.stock_porfolio.get_porfolio_value(trading_date, True)

    def plot_porfilio_return(self):
        self.stock_porfolio.plot_porfolio_return()

    def plot_porfolio_vs_market_return(self, title):
        self.stock_porfolio.plot_porfolio_vs_market_return(title)

    def build_models_performance_df(self, stock, end_date):
        current_date = self.simulation_start_date
        linear_regression_model = LinearRegressionModel.LinearRegressionModel(global_config)
        svr_rbf_model = SVRModel.SVRModel(global_config, 'rbf')
        svr_linear_model = SVRModel.SVRModel(global_config, 'linear')
        svr_poly_model = SVRModel.SVRModel(global_config, 'poly')
        performance_df = pd.DataFrame(columns = ['Date', 'price', 'l_price', 'sr_price', 'sl_price',
                                                 'sp_price', 'trend', 'l_trend', 'sr_trend', 'sl_trend', 'sp_trend'])
        while dt.strptime(current_date, "%m/%d/%Y") <= dt.strptime(end_date, "%m/%d/%Y"):
            price = stock.get_closing_price(current_date)
            l_pred_price = linear_regression_model.get_predict_price(current_date, stock)
            sr_pred_price = svr_rbf_model.get_predict_price(current_date, stock)
            sl_pred_price = svr_linear_model.get_predict_price(current_date, stock)
            sp_pred_price = svr_poly_model.get_predict_price(current_date, stock)
            prev_date = Util.add_date(current_date, -1)
            prev_price = stock.get_closing_price(prev_date)
            trend = Util.get_comp_val(prev_price, price)
            l_trend = Util.get_comp_val(prev_price, l_pred_price)
            sr_trend = Util.get_comp_val(prev_price, sr_pred_price)
            sl_trend = Util.get_comp_val(prev_price, sl_pred_price)
            sp_trend = Util.get_comp_val(prev_price, sp_pred_price)
            row = [current_date, price, l_pred_price, sr_pred_price, sl_pred_price, sp_pred_price, \
                   trend, l_trend, sr_trend, sl_trend, sp_trend]
            performance_df.loc[len(performance_df)] = row
            current_date = Util.add_date(current_date, 1)
        return performance_df

    def print_models_stat(self, perf_df):
        l_diff = perf_df['price'] - perf_df['l_price']
        sr_diff = perf_df['price'] - perf_df['sr_price']
        sl_diff = perf_df['price'] - perf_df['sl_price']
        sp_diff = perf_df['price'] - perf_df['sp_price']
        print ("Linear Regression std: %f " % np.std(l_diff))
        print ("SVR with rbf kernel std: %f" % np.std(sr_diff))
        print ("SVR with linear kernel std:  %f" % np.std(sl_diff))
        print ("SVR with poly kernel std: %f" % np.std(sp_diff))
        l_acc = perf_df['trend'] - perf_df['l_trend']
        l_acc_val = sum(i == 0 for i in l_acc) * 1.0 / len(l_acc)
        sr_acc = perf_df['trend'] - perf_df['sr_trend']
        sr_acc_val = sum(i == 0 for i in sr_acc) * 1.0 / len(sr_acc)
        sl_acc = perf_df['trend'] - perf_df['sl_trend']
        sl_acc_val = sum(i == 0 for i in sl_acc) * 1.0 / len(sl_acc)
        sp_acc = perf_df['trend'] - perf_df['sp_trend']
        sp_acc_val = sum(i == 0 for i in sp_acc) * 1.0 / len(sp_acc)
        print ("Linear Regression accuracy: %f" % l_acc_val)
        print ("SVR with rbf kernel accuracy: %f" % sr_acc_val)
        print ("SVR with linear kernel accuracy: %f" % sl_acc_val)
        print ("SVR with poly kernel accuracy: %f" % sp_acc_val)



## Main Run Porfolio Simulation
global_config = Common.ConfigGlobal()
market_simulator = MarketSimulator(global_config, global_config.market_stock)

# Print Model Performance
print_model_performance = False
if print_model_performance:
    end_date = Util.add_date(market_simulator.simulation_start_date, 40)
    stock = MPT.StockStat('AAPL', global_config.market_stock, global_config)
    perf_df = market_simulator.build_models_performance_df(stock, end_date)
    pred_price_df = perf_df[['Date', 'price', 'l_price', 'sr_price', 'sl_price', 'sp_price']]
    trend_df = perf_df[['Date','trend', 'l_trend', 'sr_trend', 'sl_trend', 'sp_trend']]
    print("Models stat for %s" % ('AAPL'))
    with pd.option_context('display.max_rows', 20, 'display.max_columns', None):
        print(pred_price_df)
        print(trend_df)

    market_simulator.print_models_stat(perf_df)

    print("Models stat for %s" % ('AMZN'))
    stock = MPT.StockStat('AMZN', global_config.market_stock, global_config)
    perf_df = market_simulator.build_models_performance_df(stock, end_date)
    market_simulator.print_models_stat(perf_df)

    print("Model stat for %s " % ('T'))
    stock = MPT.StockStat('T', global_config.market_stock, global_config)
    perf_df = market_simulator.build_models_performance_df(stock, end_date)
    market_simulator.print_models_stat(perf_df)

plot_models = False
if plot_models:
    linear_regression_model = LinearRegressionModel.LinearRegressionModel(global_config, 30)
    svr_rbf_model = SVRModel.SVRModel(global_config, 'rbf', 30)
    svr_linear_model = SVRModel.SVRModel(global_config, 'linear', 30)
    svr_poly_model = SVRModel.SVRModel(global_config, 'poly', 30)
    model = svr_linear_model
    stock = MPT.StockStat('AAPL', global_config.market_stock, global_config)
    model.get_predict_price(market_simulator.simulation_start_date, stock, True)

# Print Porfolio Stat
market_simulator.initialize_portfolio()
print_porfolio_stat = False
if print_porfolio_stat:
    market_simulator.stock_porfolio.get_porfolio_value(market_simulator.simulation_start_date, True)
    market_simulator.stock_porfolio.print_porfolio_capm()
    market_simulator.stock_porfolio.plot_porfolio_ef()

# Run market simulation
run_market_simulation = True
if run_market_simulation:
    date_increment = 7
    moving_average_model = MovingAverageModel.MovingAverageModel(global_config)
    linear_regression_model = LinearRegressionModel.LinearRegressionModel(global_config, date_increment)
    svr_rbf_model = SVRModel.SVRModel(global_config, 'rbf', date_increment)
    svr_linear_model = SVRModel.SVRModel(global_config, 'linear', date_increment)
    svr_poly_model = SVRModel.SVRModel(global_config, 'poly', date_increment)
    model = svr_rbf_model
    market_simulator.start_simulation(model, date_increment)
    market_simulator.print_porfilio_value(market_simulator.simulator_end_date)
    market_simulator.plot_porfolio_vs_market_return(model.get_name())