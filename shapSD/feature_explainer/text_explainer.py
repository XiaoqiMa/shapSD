"""
show word influence in a sentence; using LIME and SHAP visualizations
author: Xiaoqi
date: 2019.09.12
"""

import pandas as pd
from .shap_explainer import ShapExplainer
from lime.lime_text import LimeTextExplainer


def text_explanation_with_lime(x_train, instance_ind, model, class_name=None):
    try:
        instance = x_train.iloc[instance_ind]
        explainer = LimeTextExplainer(class_names=class_name)
        exp = explainer.explain_instance(instance, model.predict_proba)
        return exp.show_in_notebook(text=instance)
    except Exception as e:
        print('Model is not supported by LimeTextExplainer')
        print(e)


def text_shap_explainer(x_train_transform, model, explainer_type='Tree'):
    if isinstance(x_train_transform, pd.core.series.Series):
        x_train_transform = pd.DataFrame(x_train_transform)
    if not isinstance(x_train_transform, pd.core.frame.DataFrame):
        raise TypeError('Input should be pandas DataFrame format')
    shap_explainer = ShapExplainer(x_train_transform, model, explainer_type=explainer_type)
    return shap_explainer


def text_explanation_with_shap(shap_explainer, instance_ind):
    return shap_explainer.shap_force_plot(instance_ind=instance_ind, show_feature_value=False)
