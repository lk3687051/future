import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import datetime
import logging
import time
from model.cnn_three_mean import ThreeMeanStockModel
from model.cnn_one_mean import OneMeanStockModel
from stock.stock import set_buy_list, get_buy_list
set_buy_list(['600569', '603559', '000546'])
buyed_list = get_buy_list()
date = '2017-09-20'

print("Predict Stock at " + date)
# Model Three
print("=========================================================")
print("MODEL THREE HISTORY MEAN")
model = ThreeMeanStockModel()
model.predict(date)

#Model 1
print("=========================================================")
print("MODEL ONE HISTORY MEAN")
model = OneMeanStockModel()
model.predict(date)
