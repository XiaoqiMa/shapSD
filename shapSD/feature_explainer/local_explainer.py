"""
provide local explanation methods
author: Xiaoqi
date: 2019.07.30
"""

from .shap_explainer import ShapExplainer
from .lime_explainer import LimeExplainer
from .binary_flip import BinaryFlip
from .numeric_perturb import NumericPerturb


class LocalExplainer(object):

    def __init__(self, x_train, model):
        self.x_train = x_train
        self.model = model

    def shap_explanation(self, instance_ind=None, instance_interval=None,
                         background_sample=500, explainer_type='Tree'):
        explainer = ShapExplainer(self.x_train, self.model, explainer_type)
        return explainer.shap_force_plot(instance_ind, instance_interval, background_sample)

    def shap_values_as_df(self, instance_ind=None, instance_interval=None,
                          background_sample=500, explainer_type='Tree'):
        explainer = ShapExplainer(self.x_train, self.model, explainer_type)
        return explainer.get_shap_values_as_df(instance_ind, instance_interval,
                                               background_sample)

    def lime_explanation(self, instance_ind, num_features=10, explainer_type='tabular'):
        explainer = LimeExplainer(self.x_train, self.model, explainer_type=explainer_type)
        return explainer.lime_explain_instance(instance_ind, num_features)

    def lime_explanation_as_df(self, instance_ind=None, instance_interval=None,
                               explainer_type='tabular'):
        explainer = LimeExplainer(self.x_train, self.model, explainer_type=explainer_type)
        return explainer.get_explanation_as_df(instance_ind, instance_interval)

    def binary_flip_explanation(self, flip_attr, has_direction=False):
        explainer = BinaryFlip(self.x_train, self.model, flip_attr)
        if not has_direction:
            return explainer.calc_abs_flip_effect()
        else:
            return explainer.calc_flip_effect()

    def numeric_perturb_explanation(self, perturb_attr, value_change=None):
        explainer = NumericPerturb(self.x_train, self.model, perturb_attr)
        if value_change is not None:
            return explainer.calc_change_effect(value_change)
        else:
            return explainer.calc_perturb_effect()
