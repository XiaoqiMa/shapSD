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
    def __init__(self, x_train, model, explainer_type='Tree', background_sample=500):
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

        self.exp, self.shap_v, self.expected_v = self.calc_shap_values(background_sample=background_sample)

    def calc_shap_values(self, background_sample=500):
        exp, shap_values = None, None
        shap_v, expected_v = None, None
        try:
            if self.explainer_type == 'Tree':
                exp = self.explainer(self.model)
                shap_values = exp.shap_values(self.x_train)

            if self.explainer_type == 'Deep':
                background = self.x_train.iloc[
                    np.random.choice(self.x_train.shape[0], background_sample, replace=False)]
                exp = self.explainer(self.model, background)
                shap_values = exp.shap_values(self.x_train.values)

            if self.explainer_type == 'Kernel':
                # background = self.x_train.iloc[
                #     np.random.choice(self.x_train.shape[0], background_sample, replace=False)]
                background = shap.kmeans(self.x_train, background_sample)
                try:
                    exp = self.explainer(self.model.predict_proba, background)
                    shap_values = exp.shap_values(self.x_train.values)
                except AttributeError:
                    exp = self.explainer(self.model.predict, background)
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
            else:
                shap_v = shap_values
                expected_v = exp.expected_value
            return exp, shap_v, expected_v
        except Exception as err:
            print('Error: model is not supported by SHAP {} Explainer'.format(self.explainer_type))
            err_logging(err)
            raise Exception(err)

    def get_attr_shap_values(self, attr=None):
        # _, shap_v, _ = self.calc_shap_values(attr=attr)
        attr_index = list(self.x_train.columns).index(attr)
        shap_v = self.shap_v[:, attr_index]
        col_name = '{}_shap_values'.format(attr)
        x = self.x_train.copy()
        x[col_name] = shap_v
        return x

    def calc_shap_inter_values(self):
        try:
            if self.explainer_type == 'Tree':
                exp = self.explainer(self.model)
                shap_inter_values = exp.shap_interaction_values(self.x_train)
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

    def get_shap_values_as_df(self, instance_ind=None, instance_interval=None):
        # exp, shap_values, expected_value = self.calc_shap_values(attr=None, background_sample=background_sample)
        start = end = 0
        if instance_ind is not None:
            start = end = instance_ind

        if isinstance(instance_interval, tuple) or isinstance(instance_interval, list):
            start = instance_interval[0]
            end = instance_interval[1]

        df_exp = pd.DataFrame(columns=range(len(self.x_train.columns)))
        df_exp = pd.concat([df_exp, pd.DataFrame(self.shap_v[start:end + 1, :])])
        df_exp.columns = self.x_train.columns
        df_exp.reset_index(drop=True, inplace=True)
        return df_exp

    # @execution_time_logging
    def shap_force_plot(self, instance_ind=None, instance_interval=None, show_feature_value=True, feature_names=None):
        try:
            shap.initjs()
            # exp, shap_values, expected_value = self.calc_shap_values(attr=None, background_sample=background_sample,
            #                                                          )

            if instance_ind is not None:
                if show_feature_value:
                    features = self.x_train.iloc[instance_ind]
                else:
                    features = None
                if feature_names is None:
                    feature_names = self.x_train.columns
                return shap.force_plot(self.expected_v, self.shap_v[instance_ind],
                                       features=features,
                                       feature_names=feature_names)

            if isinstance(instance_interval, tuple) or isinstance(instance_interval, list):
                start = instance_interval[0]
                end = instance_interval[1]
                if show_feature_value:
                    features = self.x_train.iloc[start:end + 1, :]
                else:
                    features = None
                if feature_names is None:
                    feature_names = self.x_train.columns
                return shap.force_plot(self.expected_v, self.shap_v[start:end + 1, :],
                                       features=features, feature_names=feature_names)

            return shap.force_plot(self.expected_v, self.shap_v, self.x_train, feature_names=self.x_train.columns)
        except Exception as err:
            print('Error: model is not supported by SHAP force plot')
            err_logging(err)
            raise Exception(err)

    # @execution_time_logging
    def shap_summary_plot(self, plot_type='dot', interaction=False):
        try:
            if not interaction:
                shap.summary_plot(self.shap_v, self.x_train, plot_type=plot_type, show=False)
                fig_id = str(time.time()).split('.')[0]
                path = save_fig('summary_plot_{}'.format(fig_id))
                return path
            else:
                exp, shap_inter_values, expected_value = self.calc_shap_inter_values()
                shap.summary_plot(shap_inter_values, self.x_train, show=False)
                fig_id = str(time.time()).split('.')[0]
                path = save_fig('inter_summary_plot_{}'.format(fig_id))
                return path
        except Exception as err:
            print('Error: model is not supported by SHAP summary plot')
            err_logging(err)
            raise Exception(err)

    # @execution_time_logging
    def shap_dependence_plot(self, ind, interaction_index, interaction=False):
        try:
            if not interaction:
                # explainer, shap_values, expected_value = self.calc_shap_values(attr=None,
                #                                                                background_sample=background_sample,
                #                                                                )
                shap.dependence_plot(ind=ind, interaction_index=interaction_index,
                                     shap_values=self.shap_v,
                                     features=self.x_train,
                                     display_features=self.x_train, show=False)
                fig_id = str(time.time()).split('.')[0]
                path = save_fig('dependence_plot_{}_{}'.format(ind, fig_id))
                return path
                # return
            else:
                explainer, shap_inter_values, expected_value = self.calc_shap_inter_values()
                shap_inter_values = np.array(shap_inter_values)
                shap.dependence_plot((ind, interaction_index),
                                     shap_inter_values,
                                     features=self.x_train,
                                     display_features=self.x_train, show=False)
                fig_id = str(time.time()).split('.')[0]
                path = save_fig('inter_dependence_{}_{}'.format(ind, fig_id))
                return path
                # return

        except Exception as err:
            print('Error: model is not supported by SHAP dependence plot')
            err_logging(err)
            raise Exception(err)
