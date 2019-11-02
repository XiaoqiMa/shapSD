"""
provide global explanation methods
author: Xiaoqi
date: 2019.10.29
"""

from ..pysubgroup import *


class PatternExplainer(object):

    def __init__(self, x_origin, x_train, attribute, model, local_exp):
        """
        Initialize a feature pattern explainer
        :param x_origin: the original input data
        :param x_train: the encoded input data
        :param attribute: the selected feature to be inspected
        :param model: the underlying black-box model to be interpreted
        :param local_exp: the local explainer
        :param method: the method applied to estimate the local feature influence
                    'SHAP': by default, using the KernelSHAP method
                    'LIME': using the LIME method (very slow)
                    'binary': using the binary flip method, only for binary features
                    'numeric': using the numeric perturb method, only for numeric features
        """
        self.x_origin = x_origin
        self.x_train = x_train
        self.attr = attribute
        self.model = model
        self.local_exp = local_exp
        self.effect_name = None

    def calc_local_feature_influence(self, method='SHAP', shap_explainer_type='Tree'):
        if method == 'binary':
            df_flip_effect = self.local_exp.binary_flip_explanation(self.attr)
            df_flip = self.x_origin.copy()
            # df_flip.reset_index(drop=True, inplace=True)
            self.effect_name = df_flip_effect.columns.tolist()[-1]
            df_flip[self.effect_name] = df_flip_effect[self.effect_name]
            return df_flip
        if method == 'numeric':
            df_perturb_effect = self.local_exp.numeric_perturb_explanation(self.attr)
            df_perturb = self.x_origin.copy()
            # df_perturb.reset_index(drop=True, inplace=True)
            self.effect_name = df_perturb_effect.columns.tolist()[-1]
            df_perturb[self.effect_name] = df_perturb_effect[self.effect_name]
            return df_perturb
        if method == 'LIME':
            df_lime_effect = self.local_exp.lime_explanation_as_df(instance_interval=(0, len(self.x_train)))
            df_lime = self.x_origin.copy()
            # df_lime.reset_index(drop=True, inplace=True)
            self.effect_name = '{}_lime_weight'.format(self.attr)
            df_lime[self.effect_name] = df_lime_effect[self.attr]
            return df_lime
        if method == 'SHAP':
            df_shap_effect = self.local_exp.shap_values_with_attr(self.attr, explainer_type=shap_explainer_type)
            df_shap = self.x_origin.copy()
            # df_shap.reset_index(drop=True, inplace=True)
            self.effect_name = df_shap_effect.columns.tolist()[-1]
            df_shap[self.effect_name] = df_shap_effect[self.effect_name]
            return df_shap

    def subgroup_discovery(self, df_effect, measure=GAStandardQFNumeric(0.8), ignore_labels=[], inverse_effect=False,
                           statistic_is_positive=True):
        target = NumericTarget(self.effect_name)
        ignore_labels.extend([self.effect_name, self.attr])
        search_space = create_selectors(df_effect, nbins=10, ignore=ignore_labels)
        task = SubgroupDiscoveryTask(df_effect, target, search_space, qf=measure, result_set_size=10)
        result = BeamSearch().execute(task, inverse_effect, statistic_is_positive)
        df_result = as_df(df_effect, result, statistics_to_show=all_statistics_numeric)
        return df_result[['quality', 'subgroup', 'size_sg', 'mean_sg', 'mean_dataset', 'mean_lift']]
