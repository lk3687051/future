import datetime
import tensorflow as tf
from stock.stock import get_buy_list
from features import get_predict_dateset, get_train_dataset, get_test_datesets
import os
import numpy as np

class StockModel():
    def __init__(self):
        self.model_base_path = os.path.join("C:\production\\data\\models\\", self.name)
        self.mnist_classifier = tf.estimator.Estimator(
                model_fn=self.model_fn, model_dir=os.path.join(self.model_base_path, 'master'))

    def get_dataset(self, train = False, date = None):
        if train:
            self.features, self.labels = get_train_dataset(name = 'history')
            self.labels = self.process_labels(self.labels[self.target_label])
            self.test_features, self.test_labels = get_test_datesets(name = 'history')
            self.test_labels = self.process_labels(self.test_labels[self.target_label])
        else:
            self.dataset = get_predict_dateset(name = 'history', date = date)

    def get_versions(self):
        self.versions = [f for f in os.listdir(self.model_base_path) if os.path.isdir(os.path.join(self.model_base_path, f))]

    def set_stock_buy_list(stocks):
        cls.stock_buy = stocks

    def predict(self,  date):
        self.get_dataset(date=date)
        self.input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": self.dataset.values.astype(np.float32)},
                    y=None,
                    num_epochs=1,
                    shuffle=False)
        buyed_list = get_buy_list()
        print('Now we are predict')
        self.get_versions()
        for version in self.versions:
            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            model_path = os.path.join(self.model_base_path, version)
            self.mnist_classifier = tf.estimator.Estimator(
                    model_fn=self.model_fn, model_dir=model_path)
            print("Now we are predict Stock by model %s with version %s" %(self.name, version))
            results = self.mnist_classifier.predict(input_fn=self.input_fn)
            for (stock, result) in zip(self.dataset.index,  results):
                if stock in buyed_list or result['probabilities'][2] > 0.8:
                    print("---------------------------------------------------------------------------------------------------------")
                    print(stock)
                    print(result)

    def eval(self, date):
        pass
