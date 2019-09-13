"""
provide data pre-processing methods
author: Xiaoqi
date: 2019.06.24
"""

import pandas as pd
from matplotlib import colors
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class FeatureProcessing(object):

    def __init__(self, df_data):
        self.df_data = df_data
        self.scaler = None

    def calc_correlation(self):
        corr = np.round(spearmanr(self.df_data, nan_policy='omit').correlation, 4)
        df_corr = pd.DataFrame(data=corr, index=self.df_data.columns, columns=self.df_data.columns)
        return df_corr

    def show_corr_features(self, cut_off=0.5):
        df_corr = self.calc_correlation()
        corr_dic = {}
        for f in self.df_data.columns:
            indices = df_corr.loc[(df_corr[f] >= cut_off) & (df_corr[f] != 1)].index
            corr = df_corr.loc[(df_corr[f] >= cut_off) & (df_corr[f] != 1)][f]
            if len(indices) > 0:
                for i in range(len(corr)):
                    if not (indices[i], f) in corr_dic.keys():
                        corr_dic[(f, indices[i])] = corr[i]

        if len(corr_dic) == 0:
            print('There is no correlated features with coefficient larger than {} in this dataset'.format(cut_off))
        else:
            print('The following dataframe shows the correlated features in this dataset')
            corr_df = pd.DataFrame(corr_dic, index=['spearmanr_corr']).transpose()
            return corr_df.sort_values('spearmanr_corr', ascending=False)

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
    def vis_corr_dataframe(self):
        def background_gradient(s, m, M, cmap='PuBu', low=0, high=0):
            rng = M - m
            norm = colors.Normalize(m - (rng * low),
                                    M + (rng * high))
            normed = norm(s.values)
            c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
            return ['background-color: %s' % color for color in c]

        df_corr = self.calc_correlation()
        df_color = pd.DataFrame(df_corr, index=self.df_data.columns)
        df_color = df_color.transpose()
        return df_color.style.apply(background_gradient,
                                    cmap='PuBu',
                                    m=df_color.min().min(),
                                    M=df_color.max().max(),
                                    low=0,
                                    high=0.2)

    def data_scaling(self, scale_type='standard'):
        data = self.df_data.copy()
        try:
            # if scale_type == 'scale':
            #     return pd.DataFrame(scale(data), columns=self.df_data.columns)
            if scale_type == 'min_max':
                self.scaler = MinMaxScaler()
                return pd.DataFrame(self.scaler.fit_transform(data), columns=self.df_data.columns)
            if scale_type == 'standard':
                self.scaler = StandardScaler()
                return pd.DataFrame(self.scaler.fit_transform(data), columns=self.df_data.columns)
        except Exception as err:
            print('Scaling type does not support')
            print(err)

    def data_scaling_inverse(self, data, scale_type='standard'):
        try:
            if scale_type == 'min_max':
                return pd.DataFrame(self.scaler.inverse_transform(data), columns=data.columns)
            if scale_type == 'standard':
                return pd.DataFrame(self.scaler.inverse_transform(data), columns=data.columns)
        except Exception as err:
            print('Scaling type does not support')
            print(err)
