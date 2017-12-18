import tushare as ts
from pprint import pprint
import redis
from future.stock.stock import StockInfo, StockHistory
from rq import Queue
from tasks import get_stock_data
q = Queue(connection=redis.Redis())
def get_stock_list():
    stock_list = StockInfo.get().index.tolist()
    return stock_list

from datetime import date
from dateutil.rrule import rrule, DAILY

def get_dataset():
    stocks = get_stock_list()
    for stock in stocks:
        a = date(2014, 10, 27)
        b = date(2017, 10, 26)
        for dt in rrule(DAILY, dtstart=a, until=b):
            dt_str = dt.strftime("%Y-%m-%d")
            if dt.weekday() in [0,1,2,3,4]:
                print("Now we are download %s %s" % ('600848', dt_str))
                #prices = get_stock_data(stock, dt_str)
                result = q.enqueue(get_stock_data, stock, dt_str)
                #if prices:
                #    store_prices(stock, dt_str, prices)
get_dataset()
