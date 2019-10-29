"""
provide global explanation methods
author: Xiaoqi
date: 2019.10.29
"""
from .feature_importance import *
from .shap_explainer import ShapExplainer


class GlobalExplainer(object):

    def __init__(self, x_train, y_train, model):
        """
        Initialize a feature global explainer
        :param x_train: input data
        :param y_train: output data
        :param model: the underlying black-box model to be interpreted
        """
        self.x_train = x_train
        self.y_train = y_train
        self.model = model

    def permutation_importance(self, use_eli5=False):
        """
        Global variable influence measured by permutation importance
        :param use_eli5: bool, if True, use the ELI5 implementation, otherwise the raw implementation
        :return: feature importance ranking plot
        """
        feature_imp = FeatureImportance(self.x_train, self.y_train, self.model)
        if use_eli5:
            return feature_imp.eli5_perm_importance()
        else:
            imp = feature_imp.permutation_importance()
            return feature_imp.vis_perm_importance(imp)

    def weights_importance(self):
        """
        Global variable influence measured by feature weights
        :return: an explanation of estimator parameters (weights)
        """
        feature_imp = FeatureImportance(self.x_train, self.y_train, self.model)
        return feature_imp.eli5_weights_importance(show=['feature_importances', 'target', 'description'])

    def shap_feature_importance(self):
        """
        Global variable influence measured by SHAP feature importance (average absolute marginal
        effect of each feature)
        :return: a summary plot visualized using SHAP
        """
        tree_shap = ShapExplainer(self.x_train, self.model, explainer_type='Tree')
        return tree_shap.shap_summary_plot(plot_type='bar')
