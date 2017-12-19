from future.stock.stock import StockInfo, StockHistory
import datetime
import time
from future.dataset import dataset
update_time  = None
df = None
class history_feature(dataset):
    def __init__(self, date = None):
        self.date = date
        if date is None:
            self.date = str(datetime.date.today())
        self._get_stocks()
        super(history_feature, self).__init__()

    def _get_stocks(self):
        self.stock_info = StockInfo.get()

    def gen_train_dataset(self):
        print("Begin collect dataset of " + self.name)
        samples = {}
        for stock in self.stock_info.index.tolist():
            for (key, sample) in self._get_train_feature(stock):
                samples[key] = sample
        # Convert samples to dataframe structs
        df = self._samples_to_df(samples)
        if df is None or df.empty:
            print("Did not create samples, Please check again and again")
        train_df = df.sample(frac=0.9)
        test_df = df.loc[~df.index.isin(train_df.index)]
        self.save_dataset(train_df, test_df)

    def daily_feature(self):
        now = int(time.time())
        global df, update_time
        if df is not None and now - update_time < 120:
            return df
        end_day = (datetime.datetime.strptime(self.date,'%Y-%m-%d')  - datetime.timedelta(days=360)).strftime('%Y%m%d')
        samples = {}
        for stock in self.stock_info.index.tolist():
            if str(self.stock_info.loc[stock,'timeToMarket']) > end_day:
                continue
            feature = self._get_daily_featue(stock)
            if feature:
                samples[stock] = feature
        df = self._samples_to_df(samples)
        update_time = int(time.time())
        return df

    def _get_daily_featue(self, stock):
        index = 0
        begin_date = (datetime.datetime.strptime(self.date,'%Y-%m-%d')  - datetime.timedelta(days=120)).strftime('%Y-%m-%d')
        df = StockHistory.get_history(stock_id = stock)
        if df is None or df.empty:
            #print("Can not get history of stock " + stock)
            return None
        # If have date means we need preedict or eval
        if self.date in df.index:
            index = df.index.tolist().index(self.date)
            if len(df) < self.history_length + 1 or df.index[self.history_length + index] < begin_date:
                #print("Stock miss so many days " + stock)
                return None
        else:
            #print("Stock %s did not in index " % (self.date) + stock)
            return None
        # Why here is 61, because the last close
        pre_features = self.get_pre_process(df = df[:index + self.history_length + 1].copy())
        # Make sure the features is more than 60 days
        if pre_features is None or len(pre_features) < index + self.history_length:
            print("pre_feature is None or length is to small")
            return None
        sample = self.get_feature(pre_features[index : index + self.history_length])


        if sample['feature_c_change0'] >= 9.95:
            return None
        return sample

    def _get_train_feature(self, stock):
        df = StockHistory.get_history(stock_id = stock)
        if df is None or df.empty:
            return None
        pre_features = self.get_pre_process(df = df)
        # The train dataset just contains below 2017-07-01
        pre_features = pre_features['2017-07-01': ]
        # Make sure the features is more than 60 days
        if pre_features is None or len(pre_features) < self.history_length:
            return None
        for i in range(0, len(pre_features) - self.history_length , 5):
            date_time = pre_features.index[i]
            sample = self.get_feature(pre_features[i:i+self.history_length])
            # If today close price is bigger than 9.95 skip it. And I do not know Why.
            if sample['feature_c_change0'] >= 9.95:
                continue
            key = stock + '_' + date_time
            yield key, sample
