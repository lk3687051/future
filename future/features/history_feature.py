import pandas as pd
import numpy as np
import sys
from future.stock.stock import StockInfo, StockHistory, StockFeature
import datetime
import random
def get_stocks():
    s_list = StockInfo.get().index.tolist()
    return s_list

def get_stocks_with_outstanding():
    info = StockInfo.get()

    return info.index.tolist(), info['outstanding'].tolist()

def samples_to_df(samples, length):
    # convert list to dataframe
    keys = None
    d = np.zeros((len(samples), length))
    i = 0
    for (index, sample) in samples.items():
        if i is 0:
            keys = sample.keys()
        d[i] = list(sample.values())
        i = i + 1

    df = pd.DataFrame(data=d, index=samples.keys(), columns=keys)
    return df

def get_pre_process(df):
    feature_df = pd.DataFrame()
    # 预处理
    df['close1'] = df['close'].shift(-1)
    feature_df['feature_h_chang'] = (df['high'] - df['close1'] )/ df['close1']
    feature_df['feature_l_change'] = (df['low'] - df['close1'] )/ df['close1']
    feature_df['feature_c_change'] = (df['close'] - df['close1'] )/ df['close1']
    feature_df['feature_o_change'] = (df['open'] - df['close1'])/df['close1']

    feature_df['feature_turnover'] = df['turnover']
    # 获取预测值

    feature_df['target_price_change1'] =  df['p_change'].shift(1)
    feature_df['target_price_change3'] =  (df['p_change'].shift(1) + \
                                                            df['p_change'].shift(2) + \
                                                            df['p_change'].shift(3) ) / 3 \

    feature_df['target_price_change5'] =  (df['p_change'].shift(1) + \
                                                            df['p_change'].shift(2) + \
                                                            df['p_change'].shift(3) + \
                                                            df['p_change'].shift(4) + \
                                                            df['p_change'].shift(5) ) / 5

    return feature_df

def get_feature(pre_feature):
    sample = {}
    for j in range(0, 60):
        sample['feature_h_chang' + str(j)] = pre_feature.at[pre_feature.index[j], 'feature_h_chang'] * 100
        sample['feature_l_change' + str(j)] = pre_feature.at[pre_feature.index[j],'feature_l_change'] * 100
        sample['feature_c_change' + str(j)] = pre_feature.at[pre_feature.index[j], 'feature_c_change'] * 100
        sample['feature_o_change' + str(j)] = pre_feature.at[pre_feature.index[j], 'feature_o_change'] * 100
        sample['feature_turnover' + str(j)] = pre_feature.at[pre_feature.index[j], 'feature_turnover']

    sample['target_price_change1'] = pre_feature.at[pre_feature.index[0],'target_price_change1']
    sample['target_price_change3'] = pre_feature.at[pre_feature.index[0],'target_price_change3']
    sample['target_price_change5'] = pre_feature.at[pre_feature.index[0],'target_price_change5']
    return sample

def gen_train_feature():
    # Get stocks
    print("Begin gen train feature")
    stocks, outstandings = get_stocks_with_outstanding()
    samples = {}

    for (stock, outstanding) in zip(stocks, outstandings):
        # Get stock dataframe
        index = 0
        df = StockHistory.get_history(stock_id = stock)

        if df is None or df.empty:
            continue

        df = df['2017-08-25': ]
        if 'turnover' not in list(df.columns.values):
            df['turnover'] = df['volume'] / (outstanding * 10000)

        # When get all features and labels, first five did not have label
        pre_features = get_pre_process(df = df)[5:]

        # Make sure the features is more than 60 days
        if pre_features is None or len(pre_features) < index + 60:
            continue

        for i in range(0, len(pre_features) - 60 , 5):
            date_time = pre_features.index[i]
            samples[stock + '_' + date_time] = get_feature(pre_features[i:i+60])

    df = samples_to_df(samples, 303)
    train_df = df.sample(frac=0.9)
    test_df = df.loc[~df.index.isin(train_df.index)]
    StockFeature.set('history_train', train_df)
    StockFeature.set('history_test', test_df)
    print(train_df)
    print(test_df)
    return None

def daily_feature(date = None):
    # Get stocks
    stocks, outstandings = get_stocks_with_outstanding()
    samples = {}
    begin_date = (datetime.datetime.strptime(date,'%Y-%m-%d')  - datetime.timedelta(days=100)).strftime('%Y-%m-%d')

    for (stock, outstanding) in zip(stocks, outstandings):
        # Get stock dataframe
        index = 0
        df = StockHistory.get_history(stock_id = stock)

        if df is None or df.empty:
            continue

        # If have date means we need preedict or eval
        if date in df.index:
            index = df.index.tolist().index(date)
            if len(df) < 61 or df.index[60] < begin_date:
                continue
        else:
            continue

        if 'turnover' not in list(df.columns.values):
            df['turnover'] = df['volume'] / (outstanding * 10000)

        # Why here is 61, because the last close
        pre_features = get_pre_process(df = df[:index + 61].copy())

        # Make sure the features is more than 60 days
        if pre_features is None or len(pre_features) < index + 60:
            continue

        samples[stock] = get_feature(pre_features[index : index + 60])

    # convert list to dataframe
    return samples_to_df(samples, 303)

def get_predict_dateset(date):
    df = daily_feature(date)
    return df.iloc[:, 0: 300]

def get_eval_dataset(date):
    df = daily_feature(date)
    return df[:, 0: 300], df[:, 0: 303]

def get_train_dataset():
    df = StockFeature.get('history')
    return df.iloc[0:300000, 0:300], df.iloc[0:300000, 300:303]

def get_test_dateset():
    df = StockFeature.get('history')
    return df.iloc[300000:, 0:300], df.iloc[300000:, 300:303]

def get_test_dateset_daily(date):
    df = StockFeature.get('history')
    df_ = df[df.index.str.contains(date)]
    return df_.iloc[:, 0:300], df_.iloc[:, 300:303]

if __name__ == "__main__":
    TRAIN = True
    #df = feature()
    #StockFeature.set('history', df)
    #features, labels = get_test_dataset()
    print(df)
    # df.to_csv('C:\stock_data\\features\\history.csv')
