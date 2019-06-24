"""
provide data encoding methods, supports label encoding, one-hot encoding, data discretization
author: Xiaoqi
date: 2019.06.24
"""

import pandas as pd


class DataEncoder(object):

    def __init__(self, df_data, encoding_type='label', **kwargs):
        """
        Parameters
        -----------
        df_data: pandas dataframe, input data
        encoding_type: str
                    label: label encoding
                    onehot: onehot encoding
                    discretization: equal frequency discretization
        kwargs: optional, e.g. q=10 indicates quantile to cut

        Return:
        ----------
        data: dataframe after encoding
        """
        self.df_data = df_data
        if encoding_type == 'label':
            self.label_encoding()
        elif encoding_type == 'onehot':
            self.onehot_encoding()
        elif encoding_type == 'discretization':
            self.num_discretization(**kwargs)
        else:
            raise Exception('{} is not supported'.format(encoding_type))

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

    def onehot_encoding(self):
        data = self.df_data.copy()
        return pd.get_dummies(data)

    def num_discretization(self, quantile=10):
        data = self.df_data.copy()
        num_cols = data.select_dtypes(['number']).columns
        data[num_cols] = data[num_cols].apply(lambda x: pd.qcut(x, quantile, duplicates='drop').astype(str))
        return data
