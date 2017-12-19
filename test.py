#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import datetime
import logging
import time
from future.model.cnn_three_mean import ThreeMeanStockModel
from future.model.cnn_one_mean import OneMeanStockModel
from future.model.cnn_one_radio import OneRadioStockModel
from future.model.cnn_one_radio_avg import OneRadioAvgStockModel
from future.stock.stock import set_buy_list, get_buy_list
from future.dataset.history_features_turnover import history_features_turnover
import pandas as pd
from future.utils.config import get_config
import os.path
import sys
from dateutil.rrule import rrule, DAILY

def predict(date):
    result = {}
    print("Predict Stock at " + date)
    # Model Three
    print("=========================================================")
    print("MODEL THREE HISTORY MEAN")
    model = ThreeMeanStockModel()
    for r in model.predict(date):
        if r:
            result.update(r)

    #Model 1
    print("=========================================================")
    print("MODEL ONE HISTORY MEAN")
    model = OneMeanStockModel()
    for r in model.predict(date):
        if r:
            result.update(r)

    print("=========================================================")
    print("MODEL THREE HISTORY RADIO")
    model = OneRadioStockModel()
    for r in model.predict(date):
        if r:
            result.update(r)

    print("=========================================================")
    print("CNN MODEL ONE HISTORY RADIO AVG")
    model = OneRadioAvgStockModel()
    for r in model.predict(date):
        if r:
            result.update(r)

    if not result:
        return

    df = pd.DataFrame.from_dict(result)

    down_cols = [col for col in df.columns if col.startswith('0')]
    df['down'] = df[down_cols].sum(axis=1)

    up_cols = [col for col in df.columns if col.startswith('2')]
    df['up'] = df[up_cols].sum(axis=1)
    if df.empty:
        return
    # Add results to it
    feature = history_features_turnover(date=date)
    target = feature.daily_feature().iloc[:, 300: 303]

    #a, target = get_eval_dataset('history', date)
    df = pd.concat([df, target], axis=1)
    result_path = get_config('path', 'results')
    df.to_csv(os.path.join(result_path, '%s.csv' % (date)), index=True)

if __name__ == "__main__":
    a = datetime.date(2017, 9, 1)
    b = datetime.date(2017, 12, 15)
    for dt in rrule(DAILY, dtstart=a, until=b):
        dt_str = dt.strftime("%Y-%m-%d")
        if dt.weekday() in [0,1,2,3,4]:
            print("Now we are predict %s " % (dt_str))
            predict(dt_str)
