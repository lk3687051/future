import datetime
import tensorflow as tf
from future.stock.stock import get_buy_list
import os
import numpy as np
from future.utils.config import get_config
import os.path
from future.dataset.history_feature_ratio import history_feature_ratio
model_path = get_config('path', 'model')

class StockModel():
    def __init__(self):
        self.model_base_path = os.path.join(model_path, self.name)
        self.mnist_classifier = tf.estimator.Estimator(
                model_fn=self.model_fn, model_dir=os.path.join(self.model_base_path, 'master'))

    def get_versions(self):
        self.versions = [f for f in os.listdir(self.model_base_path) if os.path.isdir(os.path.join(self.model_base_path, f))]

    def set_stock_buy_list(stocks):
        cls.stock_buy = stocks

    def predict(self,  date = None):
        self._get_daily_feature(date = date)
        if self.features is None:
            print("Did not get features")
            return None

        self.input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": self.features.values.astype(np.float32)},
                    y=None,
                    num_epochs=1,
                    shuffle=False)
        self.get_versions()
        for version in self.versions:
            # Master means this is in developing
            if version == 'master':
                continue
            print("Now we are predict Stock by model %s with version %s" %(self.name, version))
            label_num = 3
            keys = [str(i) + '-' + version + '-'+ self.name for i in range(0, label_num)]
            results = {}
            for i in range(0, label_num):
                results[keys[i]] = {}

            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            model_path = os.path.join(self.model_base_path, version)
            self.mnist_classifier = tf.estimator.Estimator(
                    model_fn=self.model_fn, model_dir=model_path)

            model_results = self.mnist_classifier.predict(input_fn=self.input_fn)
            for (stock, model_result) in zip(self.daily_dataset.index,  model_results):
                for i in range(0, label_num):
                    results[keys[i]][stock] = model_result['probabilities'][i]
            yield results

    def eval(self, date):
        pass
