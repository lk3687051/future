import tushare as ts
import time
from multiprocessing import Pool

def mp_worker(stock_id):
    for i in range(0,5):
        try:
            df = ts.get_hist_data(stock_id)
            return (stock_id, df)
        except:
            print("Can not get data, " + str(i))

if __name__ == '__main__':
    from stock.stock import StockInfo, StockHistory
    print("Start update stock info")
    time_s = time.clock()
    stocks = ts.get_stock_basics()
    StockInfo.set(stocks)
    time_cost = time.clock() - time_s
    print("End update stock info, used %d seconds" % time_cost)

    time_s = time.clock()
    ids = stocks.index.tolist()
    p = Pool(8)
    pool_result = p.map(mp_worker, ids)

    for (id, result) in pool_result:
        print("Set " + str(id))
        if result is None:
            print("Can not get history of " + id)
            continue
        StockHistory.set(id, result)
    time_cost = time.clock() - time_s
    p.close()
    p.join()
    print("End update stock history, used %d seconds" % time_cost)