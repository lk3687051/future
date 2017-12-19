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

def predictall(date):
    result = {}
    model = ThreeMeanStockModel()
    for r in model.predict(date):
        if r:
            result.update(r)

    model = OneMeanStockModel()
    for r in model.predict(date):
        if r:
            result.update(r)

    model = OneRadioStockModel()
    for r in model.predict(date):
        if r:
            result.update(r)

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
    df.drop(['1-20170913-cnn_three_mean', '1-20170927-cnn_one_mean'], axis=1)
    return df
