"""
Make perturbation on numeric variable, observe prediction changes
author: Xiaoqi
date: 2019.06.24
"""
import shap
import numpy as np


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
        x_perturb = self.x_train.copy()
        x_perturb[self.perturb_attr] = np.random.permutation(x_perturb[self.perturb_attr].values)
        new_prediction = self.get_prediction(x_perturb)

        attr_name = '{}_change'.format(self.perturb_attr)
        pred_attr_name = '{}_prediction_change'.format(self.perturb_attr)
        x_perturb[attr_name] = np.abs(x_perturb[self.perturb_attr] - self.x_train[self.perturb_attr])
        x_perturb[pred_attr_name] = np.abs(new_prediction - ori_prediction)
        return x_perturb

    def calc_perturb_shap_values(self):
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.x_train)
        x = self.x_train.copy()
        attr_index = list(x.columns).index(self.perturb_attr)
        attr_name = '{}_shap_values'.format(self.perturb_attr)
        x[attr_name] = shap_values[:, attr_index]
        return x

