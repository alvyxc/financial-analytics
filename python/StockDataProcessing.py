import Util
import Common

# Read Market data
#symbols = ['SPY']
#data_path = Common.DataPathConfig()
#Util.write_stocks_to_cvs('01/01/2006', '04/01/2018', symbols, data_path, False)

symbols = ['NVDA', 'S', 'BCS', 'TD', 'TMUS', 'HMC', 'TM', 'RAD']
data_path = Common.DataPathConfig()
Util.write_stocks_to_cvs('01/01/2006', '04/01/2018', symbols, data_path, False)

# Read S&P 500 stocks
#global_config = Common.ConfigGlobal()
#Util.write_stocks_to_cvs('01/01/2006', '04/01/2018', global_config.spy_500_list, data_path, False)


