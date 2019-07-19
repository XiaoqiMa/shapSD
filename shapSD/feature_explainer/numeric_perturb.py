"""
Make perturbation on numeric variable, observe prediction changes
author: Xiaoqi
date: 2019.06.24
"""
from .shap_values import *


class NumericPerturb(object):

    def __init__(self, x_train, y_train, model, perturb_attr):
        self.x_train = x_train
        self.y_train = y_train
        self.model = model
        self.perturb_attr = perturb_attr

    def get_prediction(self, x_train):
        try:
            predictions = self.model.predict_proba(x_train)  # classification task
            return predictions[:, 0]
        except AttributeError:
            predictions = self.model.predict(x_train)  # regression task
            return predictions
        except Exception:
            raise Exception('Model does not support probability prediction')

    def calc_perturb_effect(self):
        ori_prediction = self.get_prediction(self.x_train)
        df_perturb = self.x_train.copy()
        df_perturb[self.perturb_attr] = np.random.permutation(df_perturb[self.perturb_attr].values)
        new_prediction = self.get_prediction(df_perturb)

        attr_name = '{}_change'.format(self.perturb_attr)
        pred_attr_name = '{}_prediction_change'.format(self.perturb_attr)
        df_perturb[attr_name] = np.abs(df_perturb[self.perturb_attr] - self.x_train[self.perturb_attr])
        df_perturb[pred_attr_name] = np.abs(new_prediction - ori_prediction)
        return df_perturb

    def calc_add_effect(self, add_value=10):
        ori_prediction = self.get_prediction(self.x_train)
        df_perturb = self.x_train.copy()
        df_perturb[self.perturb_attr] = df_perturb[self.perturb_attr].apply(lambda x: int(x + add_value))
        new_prediction = self.get_prediction(df_perturb)

        pred_attr_name = '{}_prediction_change'.format(self.perturb_attr)
        df_perturb[pred_attr_name] = new_prediction - ori_prediction
        return df_perturb

    def calc_perturb_shap_values(self, explainer_type='Tree'):
        shaper = ShapValues(self.x_train, self.model, explainer_type=explainer_type)
        exp, shap_v, expected_v = shaper.calc_shap_values(attr=self.perturb_attr)
        df_shap_perturb = self.x_train.copy()
        attr_name = '{}_shap_values'.format(self.perturb_attr)
        df_shap_perturb[attr_name] = shap_v
        return df_shap_perturb
