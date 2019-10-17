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
        self.x_train = x_train
        self.model = model

    def shap_explanation(self, instance_ind=None, instance_interval=None,
                         background_sample=500, explainer_type='Tree', show_feature_value=True, feature_names=None):
        explainer = ShapExplainer(self.x_train, self.model, explainer_type, background_sample)
        return explainer.shap_force_plot(instance_ind, instance_interval, show_feature_value, feature_names)

    def shap_values_as_df(self, instance_ind=None, instance_interval=None,
                          background_sample=500, explainer_type='Tree'):
        explainer = ShapExplainer(self.x_train, self.model, explainer_type, background_sample)
        return explainer.get_shap_values_as_df(instance_ind, instance_interval)

    def shap_values_with_attr(self, attr, background_sample=500, explainer_type='Tree'):
        explainer = ShapExplainer(self.x_train, self.model, explainer_type, background_sample)
        return explainer.get_attr_shap_values(attr)

    def shap_explanation_text(self, instance_ind, explainer_type='Tree'):
        shap_explainer = text_shap_explainer(self.x_train, self.model, explainer_type=explainer_type)
        return text_explanation_with_shap(shap_explainer, instance_ind)

    def lime_explanation(self, instance_ind, num_features=10, explainer_type='tabular', class_names=None):
        explainer = LimeExplainer(self.x_train, self.model, explainer_type=explainer_type, class_names=class_names)
        return explainer.show_lime_instance_explanation(instance_ind, num_features)

    def lime_explanation_text(self, instance_ind, class_names=None):
        return text_explanation_with_lime(self.x_train, instance_ind, self.model, class_name=class_names)

    def lime_explanation_as_df(self, instance_ind=None, instance_interval=None,
                               explainer_type='tabular', class_names=None):
        explainer = LimeExplainer(self.x_train, self.model, explainer_type=explainer_type, class_names=class_names)
        return explainer.get_explanation_as_df(instance_ind, instance_interval)

    def binary_flip_explanation(self, flip_attr, has_direction=False, instance_ind=None):
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
        explainer = NumericPerturb(self.x_train, self.model, perturb_attr)
        df_effect = explainer.calc_change_effect(value_change)
        if instance_ind is not None:
            return pd.DataFrame([df_effect.iloc[instance_ind]])
        else:
            return df_effect
