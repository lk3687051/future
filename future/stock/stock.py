import pandas as pd
import datetime

from future.utils.config import get_config
import os.path
dataset_path = get_config('path', 'dateset')
buyed_list = []
def set_buy_list(buyed):
    global buyed_list
    buyed_list = buyed

def get_buy_list():
    return buyed_list

class StockHistory(object):
    store = pd.HDFStore(os.path.join(dataset_path, 'history.h5'))

    @classmethod
    def _gen_key(self, stock_id):
        return "/history" + "/id_" + str(stock_id)

    @classmethod
    def set(cls, stock_id, df):
        cls.store[cls._gen_key(stock_id)] = df

    @classmethod
    def get_history(cls, stock_id = None, start = None, end = None):
        key = cls._gen_key(stock_id)
        if type(start) is datetime.date:
            start = str(start)
            end = str(end)

        if key in cls.store:
            if start and end:
                return cls.store[key].loc[end:start]
            return cls.store[key]

    @classmethod
    def get_date(cls, date = None, stock_list = None):
        series = []
        if stock_list is not None:
            for s_id in stock_list:
                key = cls._gen_key(s_id)
                if key in cls.store:
                    df = cls.store[key]
                    if date in df.index:
                        series.append(df.loc[date])

        else:
            for key in cls.store.keys():
                df = cls.store[key]
                if date in df.index:
                    series.append(df.loc[date])
        df = pd.DataFrame(series)
        return df


class StockInfo(object):
    store = pd.HDFStore(os.path.join(dataset_path, 'history.h5'))
    @classmethod
    def get(cls):
        return cls.store[u'stockinfo']

    @classmethod
    def set(cls, data):
        cls.store[u'stockinfo'] = data

class StockFeature(object):
    @classmethod
    def set(cls, name, date):
        store = pd.HDFStore('C:\stock_data/features/%s.h5' % (name))
        store['/feature'] = date

    @classmethod
    def get(cls, name):
        store = pd.HDFStore('C:\stock_data/features/%s.h5' % (name))
        return store['/feature']
