from future.utils.config import get_config
import os.path
import numpy as np
import pandas as pd
dateset_path = get_config('path', 'dateset')

class dataset(object):
    def __init__(self):
        self.train_path = os.path.join(dateset_path, self.name + '_train.csv')
        self.test_path = os.path.join(dateset_path, self.name + '_test.csv')

    def _samples_to_df(self, samples):
        # convert list to dataframe
        keys = None
        d = np.zeros((len(samples), self.feature_num + self.label_num))
        i = 0
        for (index, sample) in samples.items():
            if i is 0:
                keys = sample.keys()
            d[i] = list(sample.values())
            i = i + 1
        df = pd.DataFrame(data=d, index=samples.keys(), columns=keys)
        return df

    def get_dataset(self):
        train_df = pd.DataFrame.from_csv(self.train_path)
        test_df = pd.DataFrame.from_csv(self.test_path)
        return train_df, test_df

    def save_dataset(self, train_df, test_df):
        train_df.to_csv(self.train_path)
        test_df.to_csv(self.test_path)
