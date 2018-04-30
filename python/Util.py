from datetime import datetime, timedelta
import pandas_datareader.data as web
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
import csv
from os import mkdir
from os.path import exists, join
import requests

def add_date(original_date, num_days):
    new_date = datetime.strptime(original_date, "%m/%d/%Y")
    new_date = new_date + timedelta(days=num_days)
    new_date_str = new_date.strftime('%m/%d/%Y')
    return new_date_str

def date_str_to_integer(date_str):
    dt_date = datetime.strptime(date_str, "%m/%d/%Y")
    return 10000*dt_date.year + 100*dt_date.month + dt_date.day

def date_dt_to_integer(dt_date):
    return 10000 * dt_date.year + 100 * dt_date.month + dt_date.day


def total_return_from_returns(returns):
    """Retuns the return between the first and last value of the DataFrame.
    Parameters
    ----------
    returns : pandas.Series or pandas.DataFrame
    Returns
    -------
    total_return : float or pandas.Series
        Depending on the input passed returns a float or a pandas.Series.
    """
    return (returns + 1).prod() - 1


def write_stocks_to_cvs(start_date, end_date, stock_list, data_path, update_list):
    filtered_symbols = stock_list
    for stock in stock_list:
        csv_file = data_path.local_stocks_data_path + stock + '.csv'
        print "Writing " + stock + " to " + csv_file
        try:
            web.DataReader(stock, data_source=data_path.data_source, start=start_date, \
                       end=end_date, retry_count=0).to_csv(data_path.local_stocks_data_path + stock + '.csv')
        except ValueError:
            filtered_symbols.remove(stock)
            print "failed to get data: " + stock

    if update_list:
        header = ['Symbol']
        csv_file = data_path.spy_500_filtered_csv_file
        writer = csv.writer(open(csv_file, 'w'), lineterminator='\n')
        writer.writerow(header)
        for stock in filtered_symbols:
            writer.writerow([stock])


def read_stock(config, stock, start_data, end_data):
    csv_file = config.local_stocks_data_path + stock + '.csv'
    df = pd.read_csv(csv_file, parse_dates=['Date'], index_col=['Symbol', 'Date'])
    df = df.loc[(df.index.get_level_values('Date') >= start_data) & (df.index.get_level_values('Date') <= end_data)]
    return df


def read_stock(data_path, stock):
    csv_file = data_path.local_stocks_data_path + stock + '.csv'
    print "reading stock: " + csv_file
    df = pd.read_csv(csv_file, parse_dates=['Date'], index_col=['Symbol', 'Date'])
    return df


def filter_stock_df(df, start_date, end_date, tag):
    filtered_df = df.loc[(df.index.get_level_values('Date') >= start_date) & (df.index.get_level_values('Date') < end_date)]
    filtered_df = filtered_df[tag]
    return filtered_df


def write_spy500_list_to_csv(spy_500_table_url, csv_file):
    spy500_response = requests.get(spy_500_table_url)
    soup = BeautifulSoup(spy500_response.content, 'html.parser')
    table = soup.find("table", { "class" : "wikitable sortable" })

    # Fail now if we haven't found the right table
    header = table.findAll('th')
    if header[0].string != "Ticker symbol" or header[1].string != "Security":
        raise Exception("Can't parse wikipedia's table!")

    # Retreive the values in the table
    records = []
    rows = table.findAll('tr')
    for row in rows:
        fields = row.findAll('td')
        if fields:
            symbol = fields[0].string
            # fix as now they have links to the companies on WP
            name = ' '.join(fields[1].stripped_strings)
            sector = fields[3].string
            records.append([symbol, name, sector])

    header = ['Symbol', 'Name', 'Sector']
    writer = csv.writer(open(csv_file, 'w'), lineterminator='\n')
    writer.writerow(header)
    # Sorting ensure easy tracking of modifications
    records.sort(key=lambda s: s[1].lower())
    writer.writerows(records)


def get_spy500_list(spy500_cvs):
    df = pd.read_csv(spy500_cvs)
    spy500_list = df['Symbol'].tolist()
    return spy500_list


def get_cov(l1, l2):
    if len(l1) < len(l2):
        adjust_len = len(l2) - len(l1)
        normalize_l2 = l2.iloc[adjust_len:]
        cov_val = np.cov(l1, normalize_l2)[0][1]
        return cov_val
    elif len(l2) < len(l1):
        adjust_len = len(l1) - len(l2)
        normalize_l1 = l1.iloc[adjust_len:]
        cov_val = np.cov(normalize_l1, l2)[0][1]
        return cov_val
    else:
        cov_val = np.cov(l1, l2)[0][1]
        return cov_val


def get_comp_val(v1, v2):
    if v2 >= v1:
        return 1
    else:
        return 0