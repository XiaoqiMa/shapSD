"""
Try to inspect variable influence, starting with flipping binary variable
author: Xiaoqi
date: 2019.06.24
"""

from .shap_explain import *


class BinaryFlip(object):

    def __init__(self, x_train, y_train, model, flip_attr):
        self.x_train = x_train
        self.y_train = y_train
        self.model = model
        self.flip_attr = flip_attr

    def get_prediction(self, x_train):
        try:
            predictions = self.model.predict_proba(x_train)  # classification task
            return predictions[:, 1]
        except AttributeError:
            predictions = self.model.predict(x_train)  # regression task
            return predictions
        except Exception:
            raise Exception('Model does not support probability prediction')

    def calc_abs_flip_effect(self):
        ori_prediction = self.get_prediction(self.x_train)
        df_flip_effect = self.x_train.copy()
        df_flip_effect[self.flip_attr] = df_flip_effect[self.flip_attr].apply(lambda x: x ^ 1)
        new_prediction = self.get_prediction(df_flip_effect)

        attr_name = '{}_abs_effect'.format(self.flip_attr)
        df_flip_effect[attr_name] = np.abs(new_prediction - ori_prediction)
        return df_flip_effect

    def calc_flip_effect(self, reverse_direction=False):
        attr_val1, attr_val2 = self.x_train[self.flip_attr].unique()
        attr_index1 = self.x_train.loc[self.x_train[self.flip_attr] == attr_val1].index
        attr_index2 = self.x_train.loc[self.x_train[self.flip_attr] == attr_val2].index

        ori_prediction = self.get_prediction(self.x_train)
        df_flip_effect = self.x_train.copy()
        df_flip_effect[self.flip_attr] = df_flip_effect[self.flip_attr].apply(lambda x: x ^ 1)
        new_prediction = self.get_prediction(df_flip_effect)

        attr_name = '{}_effect'.format(self.flip_attr)
        if not reverse_direction:
            print('{} effect from "{}" to "{}"'.format(self.flip_attr, attr_val1, attr_val2))
            df_flip_effect[attr_name] = new_prediction - ori_prediction
            df_flip_effect[attr_name].iloc[attr_index1] = - df_flip_effect[attr_name]
        else:
            print('{} effect from "{}" to "{}"'.format(self.flip_attr, attr_val2, attr_val1))
            df_flip_effect[attr_name] = new_prediction - ori_prediction
            df_flip_effect[attr_name].iloc[attr_index2] = - df_flip_effect[attr_name]

        return df_flip_effect

    def calc_flip_shap_values(self, explainer_type='Tree'):
        shaper = ShapExplain(self.x_train, self.model, explainer_type=explainer_type)
        exp, shap_v, expected_v = shaper.calc_shap_values(attr=self.flip_attr)
        df_flip_shap = self.x_train.copy()
        attr_name = '{}_shap_values'.format(self.flip_attr)
        df_flip_shap[attr_name] = shap_v
        return df_flip_shap
