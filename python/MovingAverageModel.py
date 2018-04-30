import pandas as pd
import numpy as np
import Util
import Common


class MovingAverageModel:
    global_config = None

    def __init__(self, config):
        self.global_config = config

    def get_trade_action(self, current_date, stock):
        day42 = Util.add_date(current_date, -42)
        d42_average_df = Util.filter_stock_df(stock.stock_df, day42, current_date, self.global_config.close_tag)
        d42_average = d42_average_df.mean()
        day252 = Util.add_date(current_date, -252)
        d252_average_df = Util.filter_stock_df(stock.stock_df, day252, current_date, self.global_config.close_tag)
        d252_average = d252_average_df.mean()
        action = Common.TradeAction.HOLD
        if d42_average > d252_average:
            action = Common.TradeAction.BUY
        elif d42_average < d252_average:
            action = Common.TradeAction.SELL
        return action

    def get_name(self):
        return 'Moving Average Strategy'