import Util
import Common
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegressionModel:
    global_config = None
    num_day = -42
    num_days_to_pred = 1

    def __init__(self, config, num_days_to_pred=1):
        self.global_config = config
        self.num_days_to_pred = num_days_to_pred

    def get_predict_model(self, dates, prices):
        # defining the linear regression model
        linear_mod = linear_model.LinearRegression()
        # converting to matrix of n X 1
        dates = np.reshape(dates, (len(dates), 1))
        dates = preprocessing.scale(dates)
        prices = np.reshape(prices, (len(prices), 1))
        # fitting the data points in the model
        linear_mod.fit(dates, prices)
        return linear_mod

    def get_predict_price(self, current_date, stock, plot=False):
        day_past = Util.add_date(current_date, self.num_day)
        day_past_df = Util.filter_stock_df(stock.stock_df, day_past, current_date, self.global_config.close_tag)
        date_index = day_past_df.index.get_level_values('Date').tolist()
        dates_int_val = []
        dx = 1.0
        for d in date_index:
            dates_int_val.append(dx)
            dx += 1.0
        prices = day_past_df.tolist()
        pred_date_int_val = []
        model = self.get_predict_model(dates_int_val, prices)

        for i in range(self.num_days_to_pred):
            pred_date_int_val.append(dx)
            dx += 1.0

        # normalized pred date
        dates_training_data = dates_int_val + pred_date_int_val
        all_dates_normalized = preprocessing.scale(dates_training_data)
        pred_date_normalized = all_dates_normalized[-self.num_days_to_pred:]
        pred_date_normalized = np.reshape(pred_date_normalized, (len(pred_date_normalized), 1))
        predicted_prices = model.predict(pred_date_normalized)
        if plot:
            self.plot_model(stock, predicted_prices, current_date)
        return predicted_prices.mean()


    def plot_model(self, stock, predicted_prices, current_date):
        dates = []
        prices = []
        trade_date = current_date
        for i in range(self.num_days_to_pred):
            dates.append(trade_date)
            prices.append(stock.get_closing_price(trade_date))
            trade_date = Util.add_date(trade_date, 1)
        print dates
        print prices
        dates = pd.to_datetime(dates, format="%m/%d/%Y")
        plt.plot(dates, prices, color='red', label='Actual')
        plt.plot(dates, predicted_prices, color='blue', label='Predicted')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
        plt.title('Linear Regression Model')
        plt.ylabel('Price')
        plt.show()
        plt.show(block=True)
        plt.interactive(False)


    def get_trade_action(self, current_date, stock):
        pred_price = self.get_predict_price(current_date, stock)
        prev_date = Util.add_date(current_date, -1)
        prev_price = stock.get_closing_price(prev_date)
        action = Common.TradeAction.HOLD
        if pred_price > prev_price:
            action = Common.TradeAction.BUY
        elif pred_price < prev_price:
            action = Common.TradeAction.SELL
        return action

    def get_name(self):
        return 'Linear Regression'


