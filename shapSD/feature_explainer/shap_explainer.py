"""
calculate shapley values for single instance; plot shapley values
author: Xiaoqi
date: 2019.06.24
"""
import shap
import numpy as np
import pandas as pd
from .utils import *
from .logging_custom import *


class ShapExplainer(object):
    def __init__(self, x_train, model, explainer_type='Tree'):
        """
        Parameters:
        --------------
        x_train: dataframe type; input training data
        model: model that has been fitted on training data
        explainer_type: str
                    'Tree': shap TreeExplainer
                    'Deep': shap DeepExplainer
                    'Kernel': shap KernelExplainer
        """
        self.x_train = x_train
        self.model = model
        self.explainer_type = explainer_type
        if self.explainer_type == 'Tree':
            self.explainer = shap.TreeExplainer
        elif self.explainer_type == 'Deep':
            self.explainer = shap.DeepExplainer
        elif self.explainer_type == 'Kernel':
            self.explainer = shap.KernelExplainer
        else:
            raise TypeError('Does not support {} Explainer'.format(self.explainer_type))

    def calc_shap_values(self, attr=None, background_sample=500, **kwargs):
        exp, shap_values = None, None
        shap_v, expected_v = None, None
        try:
            if self.explainer_type == 'Tree':
                exp = self.explainer(self.model, **kwargs)
                shap_values = exp.shap_values(self.x_train, **kwargs)

            if self.explainer_type == 'Deep':
                # background = self.x_train.iloc[
                #     np.random.choice(self.x_train.shape[0], background_sample, replace=False)]
                background = shap.kmeans(self.x_train, background_sample)
                exp = self.explainer(self.model, background, **kwargs)
                shap_values = exp.shap_values(self.x_train.values)

            if self.explainer_type == 'Kernel':
                # background = self.x_train.iloc[
                #     np.random.choice(self.x_train.shape[0], background_sample, replace=False)]
                background = shap.kmeans(self.x_train, background_sample)
                try:
                    exp = self.explainer(self.model.predict_proba, background, **kwargs)
                    shap_values = exp.shap_values(self.x_train.values)
                except AttributeError:
                    exp = self.explainer(self.model.predict, background, **kwargs)
                    shap_values = exp.shap_values(self.x_train.values)
                except Exception as err:
                    raise Exception(err)

            if isinstance(shap_values, list):
                if len(shap_values) == 1:
                    shap_v = shap_values[0]
                    expected_v = exp.expected_value[0]
                if len(shap_values) == 2:
                    shap_v = shap_values[1]
                    expected_v = exp.expected_value[1]
                if attr is not None:
                    attr_index = list(self.x_train.columns).index(attr)
                    shap_v = shap_v[:, attr_index]
            else:
                shap_v = shap_values
                if attr is not None:
                    attr_index = list(self.x_train.columns).index(attr)
                    shap_v = shap_v[:, attr_index]
                expected_v = exp.expected_value
            return exp, shap_v, expected_v
        except Exception as err:
            print('Error: model is not supported by SHAP {} Explainer'.format(self.explainer_type))
            err_logging(err)
            raise Exception(err)

    def calc_shap_inter_values(self, **kwargs):
        try:
            if self.explainer_type == 'Tree':
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

    def get_shap_values_as_df(self, instance_ind=None, instance_interval=None, background_sample=500):
        exp, shap_values, expected_value = self.calc_shap_values(attr=None, background_sample=background_sample)
        start = end = 0
        if instance_ind is not None:
            start = end = instance_ind

        if isinstance(instance_interval, tuple) or isinstance(instance_interval, list):
            start = instance_interval[0]
            end = instance_interval[1]

        df_exp = pd.DataFrame(columns=range(len(self.x_train.columns)))
        df_exp = pd.concat([df_exp, pd.DataFrame(shap_values[start:end + 1, :])])
        df_exp.columns = self.x_train.columns
        df_exp.reset_index(drop=True, inplace=True)
        return df_exp

    # @execution_time_logging
    def shap_force_plot(self, instance_ind=None, instance_interval=None, background_sample=500, **kwargs):
        try:
            shap.initjs()
            exp, shap_values, expected_value = self.calc_shap_values(attr=None, background_sample=background_sample,
                                                                     **kwargs)
            if instance_ind is not None:
                return shap.force_plot(expected_value, shap_values[instance_ind],
                                       self.x_train.iloc[instance_ind])

            if isinstance(instance_interval, tuple) or isinstance(instance_interval, list):
                start = instance_interval[0]
                end = instance_interval[1]
                return shap.force_plot(expected_value, shap_values[start:end + 1, :],
                                       self.x_train.iloc[start:end + 1, :])

            return shap.force_plot(expected_value, shap_values, self.x_train)
        except Exception as err:
            print('Error: model is not supported by SHAP force plot')
            err_logging(err)
            raise Exception(err)

    # @execution_time_logging
    def shap_summary_plot(self, plot_type='dot', interaction=False, background_sample=500, **kwargs):
        try:
            if not interaction:
                exp, shap_values, expected_value = self.calc_shap_values(attr=None, background_sample=background_sample,
                                                                         **kwargs)
                shap.summary_plot(shap_values, self.x_train, plot_type=plot_type, show=False, **kwargs)
                fig_id = str(time.time()).split('.')[0]
                path = save_fig('summary_plot_{}'.format(fig_id))
                return path
            else:
                exp, shap_inter_values, expected_value = self.calc_shap_inter_values(**kwargs)
                shap.summary_plot(shap_inter_values, self.x_train, show=False, **kwargs)
                fig_id = str(time.time()).split('.')[0]
                path = save_fig('inter_summary_plot_{}'.format(fig_id))
                return path
        except Exception as err:
            print('Error: model is not supported by SHAP summary plot')
            err_logging(err)
            raise Exception(err)

    # @execution_time_logging
    def shap_dependence_plot(self, ind, interaction_index, interaction=False, background_sample=500, **kwargs):
        try:
            if not interaction:
                explainer, shap_values, expected_value = self.calc_shap_values(attr=None,
                                                                               background_sample=background_sample,
                                                                               **kwargs)
                shap.dependence_plot(ind=ind, interaction_index=interaction_index,
                                     shap_values=shap_values,
                                     features=self.x_train,
                                     display_features=self.x_train, show=False, **kwargs)
                fig_id = str(time.time()).split('.')[0]
                path = save_fig('dependence_plot_{}_{}'.format(ind, fig_id))
                return path
                return
            else:
                explainer, shap_inter_values, expected_value = self.calc_shap_inter_values(**kwargs)
                shap_inter_values = np.array(shap_inter_values)
                shap.dependence_plot((ind, interaction_index),
                                     shap_inter_values,
                                     features=self.x_train,
                                     display_features=self.x_train, show=False, **kwargs)
                fig_id = str(time.time()).split('.')[0]
                path = save_fig('inter_dependence_{}_{}'.format(ind, fig_id))
                return path
                return

        except Exception as err:
            print('Error: model is not supported by SHAP dependence plot')
            err_logging(err)
            raise Exception(err)
