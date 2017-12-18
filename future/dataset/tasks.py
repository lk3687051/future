import redis
import tushare as ts
r = redis.StrictRedis(host='localhost', port=6379, db=1)

def data_exist(stock, date):
    return r.exists(':'.join((stock, date)))
def get_stock_data(stock, date):
    if data_exist(stock, date):
        return
    
    df = ts.get_tick_data(stock,date=date, pause=3)
    if df is None:
        return None

    prices = {}
    for i in df.index:
        key = "%.2f" % (df.ix[i]['price'])
        if key in prices:
             prices[key] += df.ix[i]['volume']
        else:
            prices[key] = df.ix[i]['volume']

    if prices:
        store_prices(stock, date, prices)

def store_prices(stock, date, prices):
    if len(prices) == 0:
        return
    if 'nan' in prices:
        return

    r.hmset(':'.join((stock, date)),prices)

import tushare as ts

df = ts.get_tick_data('600848',date='2014-01-09', pause=6)
df.head(10)
