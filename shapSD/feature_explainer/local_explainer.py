"""
provide local explanation methods
author: Xiaoqi
date: 2019.07.30
"""
import pandas as pd
from .shap_explainer import ShapExplainer
from .lime_explainer import LimeExplainer
from .binary_flip import BinaryFlip
from .numeric_perturb import NumericPerturb
from .text_explainer import text_explanation_with_lime, text_shap_explainer, text_explanation_with_shap


class LocalExplainer(object):

    def __init__(self, x_train, model):
        """
        Initialize a feature local explainer
        :param x_train: input data
        :param model: the underlying black-box model to be interpreted
        """
        self.x_train = x_train
        self.model = model

    def shap_explanation(self, instance_ind=None, instance_interval=None,
                         background_sample=500, explainer_type='Tree', show_feature_value=True, feature_names=None):
        """
        Local variable influence measured by KernelSHAP method
        :param instance_ind: int, the selected instance to be explained
        :param instance_interval: tuple or list, a list of instances to be explained
        :param background_sample: int, number background data, it is used during the permutation process
        :param explainer_type: str, SHAP explainer type
                            'Tree': SHAP TreeExplainer
                            'Deep': SHAP DeepExplainer
                            'Kernel': SHAP KernelExplainer
        :param show_feature_value: bool, False: does not show the feature value in the shap force plot
        :param feature_names: the column names will be passed by default
        :return: SHAP force plot to visualize the local feature influence
        """
        explainer = ShapExplainer(self.x_train, self.model, explainer_type, background_sample)
        return explainer.shap_force_plot(instance_ind, instance_interval, show_feature_value, feature_names)

    def shap_values_as_df(self, instance_ind=None, instance_interval=None,
                          background_sample=500, explainer_type='Tree'):
        """
        Local feature influence for the selected instances, encapsulated in a dataframe
        :param instance_ind: int, the selected instance to be explained
        :param instance_interval: tuple or list, a list of instances to be explained
        :param background_sample: int, number background data, it is used during the permutation process
        :param explainer_type: str, SHAP explainer type
                            'Tree': SHAP TreeExplainer
                            'Deep': SHAP DeepExplainer
                            'Kernel': SHAP KernelExplainer
        :return: a dataframe which contains feature influence for chosen instances, measured by SHAP
        """
        explainer = ShapExplainer(self.x_train, self.model, explainer_type, background_sample)
        return explainer.get_shap_values_as_df(instance_ind, instance_interval)

    def shap_values_with_attr(self, attr, background_sample=500, explainer_type='Tree'):
        """
        Calculate the local influence for the selected attribute
        :param attr: str, the selected attribute to be investigated
        :param background_sample: int, number background data, it is used during the permutation process
        :param explainer_type: str, SHAP explainer type
                            'Tree': SHAP TreeExplainer
                            'Deep': SHAP DeepExplainer
                            'Kernel': SHAP KernelExplainer
        :return: a dataframe, contains the original dataset, and one additional column is the 
                local influence for the selected attribute
        """
        explainer = ShapExplainer(self.x_train, self.model, explainer_type, background_sample)
        return explainer.get_attr_shap_values(attr)

    def shap_explanation_text(self, instance_ind, explainer_type='Tree'):
        """
        Local text explainer, to show the weights of each feature (word), implemented using SHAP
        :param instance_ind: int, the selected instance to be explained
        :param explainer_type: str, by default SHAP TreeExplainer is applied
        :return: SHAP force plot to show the weights of each word in a text
        """
        shap_explainer = text_shap_explainer(self.x_train, self.model, explainer_type=explainer_type)
        return text_explanation_with_shap(shap_explainer, instance_ind)

    def lime_explanation(self, instance_ind, num_features=10, explainer_type='tabular', class_names=None):
        """
        Local variable influence interpretation using LIME
        :param instance_ind: int, the selected instance to be explained
        :param num_features: int, the number of features to show in the plot
        :param explainer_type: str, by default only the tabular data is supported, but 
                                it can be extended to image data
        :param class_names: list, the labels for the target to show in the plot
        :return: LIME visualization to show the feature weights (by fitting a local model
                which is considered as the approximate model to the black-box model)
        """
        explainer = LimeExplainer(self.x_train, self.model, explainer_type=explainer_type, class_names=class_names)
        return explainer.show_lime_instance_explanation(instance_ind, num_features)

    def lime_explanation_text(self, instance_ind, class_names=None):
        """
        Local text explainer, to show the weights of each feature (word), implemented using LIME
        :param instance_ind: int, the selected instance to be explained
        :param class_names: list, the labels for the target to show in the plot
        :return: LIME visualization to show the weights of each word in a text
        """
        return text_explanation_with_lime(self.x_train, instance_ind, self.model, class_name=class_names)

    def lime_explanation_as_df(self, instance_ind=None, instance_interval=None,
                               explainer_type='tabular', class_names=None):
        """
        Local feature influence for the selected instances, encapsulated in a dataframe
        :param instance_ind: int, the selected instance to be explained
        :param instance_interval: tuple or list, a list of instances to be explained
        :param explainer_type: str, by default only the tabular data is supported, but 
                                it can be extended to image data
        :param class_names: list, the labels for the target to show in the plot
        :return: a dataframe which contains feature influence for chosen instances, measured by LIME
        """
        explainer = LimeExplainer(self.x_train, self.model, explainer_type=explainer_type, class_names=class_names)
        return explainer.get_explanation_as_df(instance_ind, instance_interval)

    def binary_flip_explanation(self, flip_attr, has_direction=False, instance_ind=None):
        """
        Calculate the local influence for the selected binary attribute
        :param flip_attr: str, the selected attribute to flip
        :param has_direction: bool, False: calculate the absolute effect, True: calculate the
                              effect with one directions, e.g. effect from 'male' to 'female'
        :param instance_ind: int, the selected instance to be explained
        :return: a dataframe, contains the original dataset, and one additional column is the
                local influence for the selected attribute measured by binary flip method
        """
        explainer = BinaryFlip(self.x_train, self.model, flip_attr)
        if not has_direction:
            df_effect = explainer.calc_abs_flip_effect()
        else:
            df_effect = explainer.calc_flip_effect()

        if instance_ind is not None:
            return pd.DataFrame([df_effect.iloc[instance_ind]])
        else:
            return df_effect

    def numeric_perturb_explanation(self, perturb_attr, value_change=10, instance_ind=None):
        """
        Calculate the local influence for the selected numeric attribute
        :param perturb_attr: str, the selected attribute to perturb
        :param value_change: int, the magnitude of value to change
        :param instance_ind: int, the selected instance to be explained
        :return:  a dataframe, contains the original dataset, and one additional column is the
                local influence for the selected attribute measured by numeric perturb method
        """
        explainer = NumericPerturb(self.x_train, self.model, perturb_attr)
        df_effect = explainer.calc_change_effect(value_change)
        if instance_ind is not None:
            return pd.DataFrame([df_effect.iloc[instance_ind]])
        else:
            return df_effect
