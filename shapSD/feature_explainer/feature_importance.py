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
import matplotlib.pyplot as plt
import seaborn as sns
from .logging_custom import *


class FeatureImportance(object):

    def __init__(self, x_train, y_train, model):
        self.x_train = x_train
        self.y_train = y_train
        self.model = model

    # @execution_time_logging
    def permutation_importance(self):
        imp = []
        try:
            if hasattr(self.model, 'score'):
                base_score = self.model.score(self.x_train, self.y_train)
                for col in self.x_train.columns:
                    x = self.x_train.copy()
                    x[col] = np.random.permutation(x[col])
                    score = self.model.score(x, self.y_train)
                    imp.append(np.round(base_score - score, 4))

                df_imp = pd.DataFrame(
                    {'Features': self.x_train.columns, 'Importance': imp})
                df_imp = df_imp.sort_values('Importance', ascending=False)
                return df_imp
            else:
                base_score = mean_squared_log_error(self.model.predict(self.x_train), self.y_train)
                for col in self.x_train.columns:
                    x = self.x_train.copy()
                    x[col] = np.random.permutation(x[col])
                    score = mean_squared_log_error(self.model.predict(x), self.y_train)
                    imp.append(np.round(score - base_score, 4))

                df_imp = pd.DataFrame(
                    {'Features': self.x_train.columns, 'Importance': imp})
                df_imp = df_imp.sort_values('Importance', ascending=False)
                return df_imp
        except Exception as err:
            print('Error: model is not supported')
            err_logging(err)
            raise Exception(err)

    def model_importance(self):
        try:
            if hasattr(self.model, 'feature_importances_'):
                df_imp = pd.DataFrame(
                    {'Features': self.x_train.columns,
                     'Importance': self.model.feature_importances_})
                df_imp = df_imp.sort_values('Importance', ascending=False)
                return df_imp
        except Exception as err:
            print('Error: model is not supported')
            err_logging(err)
            raise Exception(err)

    @staticmethod
    def vis_perm_importance(df_imp):
        """
        visualize the feature importance plot
        Parameters:
        ____________
        df_imp: dataframe type, contains two columns
                first column: "Features", describing the features
                second column: "Importance": describing the feature importance score or weights
        """
        try:
            df_imp.columns = ['Features', 'Importance']
            plt.figure(figsize=(10, 6))
            sns.set(font_scale=1.5)
            sns.barplot(x="Importance", y="Features", data=df_imp)
            plt.title('Permutation Feature Importance Plot')
            plt.tight_layout()
            plt.show()
        except Exception as err:
            raise Exception('DataFrame should contains two columns, Features & Importance Score \n', err)

    # @execution_time_logging
    def eli5_perm_importance(self, **kwargs):
        try:
            perm = PermutationImportance(self.model).fit(self.x_train, self.y_train)
            return eli5.show_weights(perm, feature_names=self.x_train.columns.tolist(), **kwargs)
        except AttributeError as err:
            err_logging(err)
            raise AttributeError(err)

    def eli5_weights_importance(self, **kwargs):
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
