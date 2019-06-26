"""
various methods to calculate feature importance
author: Xiaoqi
date: 2019.06.24
"""
import eli5
import numpy as np
import pandas as pd
from eli5.sklearn import PermutationImportance
from sklearn.metrics import mean_squared_log_error

from .logging_custom import *


class FeatureImportance(object):

    def __init__(self, x_train, y_train, model):
        self.x_train = x_train
        self.y_train = y_train
        self.model = model

    @execution_time_logging
    def raw_perm_importance(self, iteration=10):
        imp = []
        imp_var = []
        try:
            if hasattr(self.model, 'score'):
                base_score = self.model.score(self.x_train, self.y_train)
                for col in self.x_train.columns:
                    scores = []
                    for m in range(iteration):
                        x = self.x_train.copy()
                        x[col] = np.random.permutation(x[col])
                        score = self.model.score(x, self.y_train)
                        scores.append(score)
                    score_drop_list = np.array(base_score) - np.array(scores)
                    variance = np.round(np.var(score_drop_list), 6)
                    score_mean_drop = np.round(np.mean(score_drop_list), 4)
                    imp.append(score_mean_drop)
                    imp_var.append('{}±{}'.format(score_mean_drop, variance))

                df_imp = pd.DataFrame(
                    {'Features': self.x_train.columns, 'Importance': imp, 'Importance weights': imp_var})
                df_imp = df_imp.sort_values('Importance', ascending=False)
                return df_imp[['Importance weights', 'Features']]
            else:
                base_score = mean_squared_log_error(self.model.predict(self.x_train), self.y_train)
                for col in self.x_train.columns:
                    scores = []
                    for m in range(iteration):
                        x = self.x_train.copy()
                        x[col] = np.random.permutation(x[col])
                        score = mean_squared_log_error(self.model.predict(x), self.y_train)
                        scores.append(score)
                    score_rmse_list = np.array(scores) - np.array(base_score)
                    variance = np.round(np.var(score_rmse_list), 6)
                    score_rmse_inc = np.round(np.mean(score_rmse_list), 4)
                    imp.append(score_rmse_inc)
                    imp_var.append('{}±{}'.format(score_rmse_inc, variance))

                df_imp = pd.DataFrame(
                    {'Features': self.x_train.columns, 'Importance': imp, 'Importance weights': imp_var})
                df_imp = df_imp.sort_values('Importance', ascending=False)
                return df_imp[['Importance weights', 'Features']]
        except Exception as err:
            print('Error: model is not supported')
            err_logging(err)
            raise Exception(err)

    @execution_time_logging
    def eli5_perm_importance(self, **kwargs):
        try:
            perm = PermutationImportance(self.model).fit(self.x_train, self.y_train)
            return eli5.show_weights(perm, feature_names=self.x_train.columns.tolist(), **kwargs)
        except AttributeError as err:
            err_logging(err)
            raise AttributeError(err)

    def eli5_weights_importance(self,  **kwargs):
        """
        Return
        -------------
        feature weights for each feature
        """

        try:
            weights = eli5.show_weights(self.model, feature_names=self.x_train.columns.tolist(), **kwargs)
            return weights
        except Exception as err:
            print('Error: model is not supported')
            err_logging(err)
            raise Exception(err)

    def eli5_instance_importance(self, instance, **kwargs):
        try:
            prediction = eli5.show_prediction(self.model, instance, show_feature_values=True,
                                              feature_names=list(self.x_train.columns), **kwargs)
            return prediction
        except Exception as err:
            print('Error: model is not supported')
            err_logging(err)
            raise Exception(err)
