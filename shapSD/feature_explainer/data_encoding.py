"""
provide data encoding methods, supports label encoding, one-hot encoding, data discretization
author: Xiaoqi
date: 2019.06.24
"""

import pandas as pd
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler


class DataEncoder(object):

    def __init__(self, df_data):
        self.df_data = df_data

    def label_encoding(self):
        try:
            data = self.df_data.copy()
            cat_cols = data.select_dtypes(['category']).columns
            if len(cat_cols) == 0:
                cat_cols = data.select_dtypes(exclude=['number']).columns
                data[cat_cols] = data[cat_cols].astype('category')
            data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes)
            return data
        except:
            raise Exception('Label encoding error')

    def get_labels(self):
        label_dic = {}
        data = self.df_data.copy()
        cat_cols = data.select_dtypes(exclude=['number']).columns
        for col in cat_cols:
            label_dic[col] = dict(enumerate(data[col].astype('category').cat.categories))
        return label_dic

    def onehot_encoding(self):
        data = self.df_data.copy()
        return pd.get_dummies(data)

    def num_discretization(self, quantile=10, **kwargs):
        data = self.df_data.copy()
        num_cols = data.select_dtypes(['number']).columns
        data[num_cols] = data[num_cols].apply(lambda x: pd.qcut(x, quantile, duplicates='drop', **kwargs).astype(str))
        return data

    def data_scaling(self, scale_type='scale'):
        data = self.df_data.copy()
        try:
            if scale_type == 'scale':
                return pd.DataFrame(scale(data), columns=self.df_data.columns)
            if scale_type == 'min_max':
                scaler = MinMaxScaler()
                return pd.DataFrame(scaler.fit_transform(data), columns=self.df_data.columns)
            if scale_type == 'standard':
                scaler = StandardScaler()
                return pd.DataFrame(scaler.fit_transform(data), columns=self.df_data.columns)
        except Exception as err:
            print('Scaling type does not support')
            print(err)
