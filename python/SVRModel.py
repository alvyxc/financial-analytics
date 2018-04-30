import numpy as np
import Util
import Common
from sklearn.svm import SVR
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd

class SVRModel:
    global_config = None
    num_day = -42
    kernel = 'rbf'
    num_days_to_pred = 1

    def __init__(self, config, kernel, num_days_to_pred=1):
        self.global_config = config
        self.kernel = kernel
        self.num_days_to_pred = num_days_to_pred

    def getSVR(self):
        svr = None
        if self.kernel == 'rbf':
            svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
        elif self.kernel == 'linear':
            svr = SVR(kernel='linear', C=1e3)
        elif self.kernel == 'poly':
            svr = SVR(kernel='poly', C=1e3, degree=2)
        else:
            raise ValueError('Kernel is not supported')
        return svr

    def get_predict_model(self, dates, prices):

        # converting to matrix of n X 1
        # normalize training dates
        dates = np.reshape(dates, (len(dates), 1))
        dates = preprocessing.scale(dates)
        svr = self.getSVR()
        svr.fit(dates, prices)
        return svr

        pred_price = svr.predict(pred_date_normalized)[0]
        return pred_price

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

        dates = pd.to_datetime(dates, format="%m/%d/%Y")
        plt.plot(dates, prices, color='red', label='Actual')
        plt.plot(dates, predicted_prices, color='blue', label='Predicted')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
        title = "SV Regression Model " + self.kernel
        plt.title(title)
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
        name = "SVR kernel = " + self.kernel
        return name