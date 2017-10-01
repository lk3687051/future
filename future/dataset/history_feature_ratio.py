from future.stock.stock import StockInfo, StockHistory
from future.dataset.history_base import history_feature
import pandas as pd
class history_feature_ratio(history_feature):
    def __init__(self, date = None):
        self.name = 'history_feature_radio'
        self.feature_num = 300
        self.label_num = 3
        self.history_length = 60
        super(history_feature_ratio, self).__init__(date)
    def get_pre_process(self, df):
        feature_df = pd.DataFrame()
        # 预处理
        df['close1'] = df['close'].shift(-1)
        feature_df['feature_h_chang'] = (df['high'] - df['close1'] )/ df['close1']
        feature_df['feature_l_change'] = (df['low'] - df['close1'] )/ df['close1']
        feature_df['feature_c_change'] = (df['close'] - df['close1'] )/ df['close1']
        feature_df['feature_o_change'] = (df['open'] - df['close1'])/df['close1']
        feature_df['volume'] = df['volume']

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

    def get_feature(self, pre_feature):
        if len(pre_feature) != self.history_length:
            print("Error the pre_feature is not equre history_length")
        pre_feature['volume_ratio'] = pre_feature['volume'] / pre_feature['volume'].mean()

        sample = {}
        for j in range(0, 60):
            sample['feature_h_chang' + str(j)] = pre_feature.at[pre_feature.index[j], 'feature_h_chang'] * 100
            sample['feature_l_change' + str(j)] = pre_feature.at[pre_feature.index[j],'feature_l_change'] * 100
            sample['feature_c_change' + str(j)] = pre_feature.at[pre_feature.index[j], 'feature_c_change'] * 100
            sample['feature_o_change' + str(j)] = pre_feature.at[pre_feature.index[j], 'feature_o_change'] * 100
            sample['volume_ratio' + str(j)] = pre_feature.at[pre_feature.index[j], 'volume_ratio']

        sample['target_price_change1'] = pre_feature.at[pre_feature.index[0],'target_price_change1']
        sample['target_price_change3'] = pre_feature.at[pre_feature.index[0],'target_price_change3']
        sample['target_price_change5'] = pre_feature.at[pre_feature.index[0],'target_price_change5']
        return sample
