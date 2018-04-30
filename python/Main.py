import Common
import MPT
import Util
import os, time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np

# Initialize configuration
global_config = Common.ConfigGlobal()
global_config.data_source = "morningstar"
global_config.close_tag = "Close"

start = '06/11/2013'
start = datetime.strptime(start, "%m/%d/%Y") #string to date
end = start + timedelta(days=-10) # date - days
end_str = end.strftime('%m/%d/%Y')

symbols = ['AAPL','MRK']
test = symbols
test.append('XYZ')
print symbols
print test
#Util.write_stocks_to_cvs(symbols, '01/01/2010', '02/28/2018', global_config)

#df = Util.read_stock('AAPL', '01/01/2011', '02/28/2015')
#print df

#source = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
#Util.write_spy500_list_to_csv(global_config.spy_500_list_url, global_config.spy_500_csv_file)

#global_config.spy_500_list = Util.get_spy500_list(global_config.spy_500_csv_file)
#Util.write_stocks_to_cvs('01/01/2016', '04/01/2018', global_config)



#spy_500_list = Util.get_spy500_list(global_config.spy_500_filtered_csv_file)
#for stock in spy_500_list:
#    global_config.spy_500_df_dict.setdefault(stock, [])

#for stock in spy_500_list:
#    df = Util.read_stock(global_config, stock)
    #    global_config.spy_500_df_dict[stock].append(df)

#print global_config.spy_500_df_dict['MMM']

#df = Util.read_stock(global_config, 'MMM')

#plt.plot(df.index.get_level_values('Date'), df[global_config.close_tag])
#plt.title('GM Stock Price')
#plt.ylabel('Price ($)')
#plt.show()
#plt.show(block=True)
#plt.interactive(False)

#ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
#ts = ts.cumsum()
#ts.plot()
#print start,end

#print end_str

#st = MPT.get_stock_stat('SPY', '01/01/2010', '02/28/2018', global_config)
#index = MPT.get_stock_stat('AAPL', '01/01/2010', '02/28/2018', global_config)

#beta = MPT.get_beta(st, index)
#print beta

#print st.annual_closing
#print st.monthly_closing