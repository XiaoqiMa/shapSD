"""
calculate shapley values for single instance; plot shapley values
author: Xiaoqi
date: 2019.06.24
"""
import time
import shap
import numpy as np
from shapSD.utils import *
from shapSD.logging_custom import *


class ShapValues(object):

    def __init__(self, x_train, model, explainer=shap.TreeExplainer):
        self.x_train = x_train
        self.model = model
        self.explainer = explainer

    def calc_shap_values(self, attr=None, **kwargs):
        try:
            exp = self.explainer(self.model, **kwargs)
            shap_values = exp.shap_values(self.x_train, **kwargs)
            if isinstance(shap_values, list):
                shap_v = shap_values[1]
                if attr is not None:
                    attr_index = list(self.x_train.columns).index(attr)
                    shap_v = shap_v[:, attr_index]
                expected_v = exp.expected_value[1]
            else:
                shap_v = shap_values
                if attr is not None:
                    attr_index = list(self.x_train.columns).index(attr)
                    shap_v = shap_v[:, attr_index]
                expected_v = exp.expected_value
            return exp, shap_v, expected_v
        except Exception as err:
            print('Error: model is not supported by SHAP TreeExplainer')
            err_logging(err)
            raise Exception(err)

    def calc_shap_inter_values(self, **kwargs):
        try:
            exp = self.explainer(self.model, **kwargs)
            shap_inter_values = exp.shap_interaction_values(self.x_train, **kwargs)
            if isinstance(shap_inter_values, list):
                shap_inter = shap_inter_values[1]
                expected_v = exp.expected_value[1]
            else:
                shap_inter = shap_inter_values
                expected_v = exp.expected_value
            return exp, shap_inter, expected_v
        except Exception as err:
            print('Error: model is not supported by SHAP TreeExplainer')
            err_logging(err)
            raise Exception(err)

    @execution_time_logging
    def shap_force_plot(self, instance_ind=None, instance_interval=None, **kwargs):
        try:
            exp, shap_values, expected_value = self.calc_shap_values(**kwargs)
            if instance_ind is not None:
                return shap.force_plot(expected_value, shap_values[instance_ind],
                                       self.x_train.iloc[instance_ind], **kwargs)

            if isinstance(instance_interval, tuple):
                start = instance_interval[0]
                end = instance_interval[1]
                return shap.force_plot(expected_value, shap_values[start:end, :],
                                       self.x_train.iloc[start:end, :], **kwargs)

            return shap.force_plot(expected_value, shap_values, self.x_train, **kwargs)
        except Exception as err:
            print('Error: model is not supported by SHAP force plot')
            err_logging(err)
            raise Exception(err)

    @execution_time_logging
    def shap_summary_plot(self, plot_type='dot', interaction=False, **kwargs):
        try:
            if not interaction:
                exp, shap_values, expected_value = self.calc_shap_values(**kwargs)
                shap.summary_plot(shap_values, self.x_train, plot_type=plot_type, show=False, **kwargs)
                fig_id = str(time.time()).split('.')[0]
                save_fig('summary_plot_{}'.format(fig_id))
                return
            else:
                exp, shap_inter_values, expected_value = self.calc_shap_inter_values(**kwargs)
                shap.summary_plot(shap_inter_values, self.x_train, show=False, **kwargs)
                fig_id = str(time.time()).split('.')[0]
                save_fig('inter_summary_plot_{}'.format(fig_id))
                return
        except Exception as err:
            print('Error: model is not supported by SHAP summary plot')
            err_logging(err)
            raise Exception(err)

    @execution_time_logging
    def shap_dependence_plot(self, ind, interaction_index, interaction=False, **kwargs):
        try:
            if not interaction:
                explainer, shap_values, expected_value = self.calc_shap_values(**kwargs)
                shap.dependence_plot(ind=ind, interaction_index=interaction_index,
                                     shap_values=shap_values,
                                     features=self.x_train,
                                     display_features=self.x_train, show=False, **kwargs)
                fig_id = str(time.time()).split('.')[0]
                save_fig('dependence_plot_{}_{}'.format(ind, fig_id))
                return
            else:
                explainer, shap_inter_values, expected_value = self.calc_shap_inter_values(**kwargs)
                shap_inter_values = np.array(shap_inter_values)
                shap.dependence_plot((ind, interaction_index),
                                     shap_inter_values,
                                     features=self.x_train,
                                     display_features=self.x_train, show=False, **kwargs)
                fig_id = str(time.time()).split('.')[0]
                save_fig('inter_dependence_{}_{}'.format(ind, fig_id))
                return

        except Exception as err:
            print('Error: model is not supported by SHAP dependence plot')
            err_logging(err)
            raise Exception(err)
