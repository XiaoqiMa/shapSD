from .shap_explainer import ShapExplainer
from .lime_explainer import LimeExplainer


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
