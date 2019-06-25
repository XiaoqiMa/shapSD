"""
Try to inspect variable influence, starting with flipping binary variable
author: Xiaoqi
date: 2019.06.24
"""
import shap
import numpy as np


class BinaryFlip(object):

    def __init__(self, x_train, y_train, model, flip_attr):
        self.x_train = x_train
        self.y_train = y_train
        self.model = model
        self.flip_attr = flip_attr

    def get_prediction(self, x_train):
        try:
            predictions = self.model.predict_proba(x_train)  # classification task
            return predictions[:, 0]
        except AttributeError:
            predictions = self.model.predict(x_train)  # regression task
            return predictions
        except Exception:
            raise Exception('Model does not support probability prediction')

    def calc_flip_effect(self):
        ori_prediction = self.get_prediction(self.x_train)
        df_flip_effect = self.x_train.copy()
        df_flip_effect[self.flip_attr] = df_flip_effect[self.flip_attr].apply(lambda x: x ^ 1)
        new_prediction = self.get_prediction(df_flip_effect)

        avg_effect = np.mean(np.abs(new_prediction - ori_prediction))
        attr_name = '{}_effect'.format(self.flip_attr)
        df_flip_effect[attr_name] = np.abs(new_prediction - ori_prediction) - avg_effect
        return df_flip_effect

    def calc_flip_shap_values(self):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.x_train)
        df_flip_shap = self.x_train.copy()
        attr_index = list(df_flip_shap.columns).index(self.flip_attr)
        attr_name = '{}_shap_values'.format(self.flip_attr)
        df_flip_shap[attr_name] = shap_values[:, attr_index]
        return df_flip_shap
