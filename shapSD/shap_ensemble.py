import shap
import numpy as np
from shapSD.utils import *
from shapSD.logging_custom import *
import time

def calc_shap_values(X_train, model, **kwargs):
    try:
        explainer = shap.TreeExplainer(model, **kwargs)
        shap_values = explainer.shap_values(X_train, **kwargs)
        if isinstance(shap_values, list):
            shap_v = shap_values[1]
            expected_v = explainer.expected_value[1]
        else:
            shap_v = shap_values
            expected_v = explainer.expected_value
        return explainer, shap_v, expected_v
    except Exception as err:
        print('Error: model is not supported by SHAP TreeExplainer')
        err_logging(err)
        raise Exception(err)


def calc_shap_inter_values(X_train, model, **kwargs):
    try:
        explainer = shap.TreeExplainer(model, **kwargs)
        shap_inter_values = explainer.shap_interaction_values(X_train, **kwargs)
        if isinstance(shap_inter_values, list):
            shap_inter = shap_inter_values[1]
            expected_v = explainer.expected_value[1]
        else:
            shap_inter = shap_inter_values
            expected_v = explainer.expected_value
        return explainer, shap_inter, expected_v
    except Exception as err:
        print('Error: model is not supported by SHAP TreeExplainer')
        err_logging(err)
        raise Exception(err)


@execution_time_logging
def shap_force_plot(X_trian, model, instance_ind=None, instance_interval=None, **kwargs):
    try:
        explainer, shap_values, expected_value = calc_shap_values(X_trian, model, **kwargs)
        if instance_ind:
            return shap.force_plot(expected_value, shap_values[instance_ind],
                                   X_trian.iloc[instance_ind], **kwargs)

        if isinstance(instance_interval, tuple):
            start = instance_interval[0]
            end = instance_interval[1]
            return shap.force_plot(expected_value, shap_values[start:end, :],
                                   X_trian.iloc[start:end, :], **kwargs)

        return shap.force_plot(expected_value, shap_values, X_trian, **kwargs)
    except Exception as err:
        print('Error: model is not supported by SHAP force plot')
        err_logging(err)
        raise Exception(err)


@execution_time_logging
def shap_summary_plot(X_train, model, plot_type='dot', interaction=False, **kwargs):
    try:
        if not interaction:
            explainer, shap_values, expected_vlaue = calc_shap_values(X_train, model, **kwargs)
            shap.summary_plot(shap_values, X_train, plot_type=plot_type, show=False, **kwargs)
            save_fig('summary_plot_{}'.format(str(time.time()).split('.')[0]))
            return
        else:
            explainer, shap_inter_values, expected_vlaue = calc_shap_inter_values(X_train, model, **kwargs)
            shap.summary_plot(shap_inter_values, X_train, show=False, **kwargs)
            save_fig('inter_summary_plot_{}'.format(str(time.time()).split('.')[0]))
            return
    except Exception as err:
        print('Error: model is not supported by SHAP summary plot')
        err_logging(err)
        raise Exception(err)


@execution_time_logging
def shap_dependence_plot(X_train, model, ind, interaction_index, interaction=False, **kwargs):
    try:
        if not interaction:
            explainer, shap_values, expected_value = calc_shap_values(X_train, model, **kwargs)
            shap.dependence_plot(ind=ind, interaction_index=interaction_index,
                                 shap_values=shap_values,
                                 features=X_train,
                                 display_features=X_train, show=False, **kwargs)
            save_fig('dependence_plot_{}_{}'.format(ind, str(time.time()).split('.')[0]))
            return
        else:
            explainer, shap_inter_values, expected_value = calc_shap_inter_values(X_train, model, **kwargs)
            shap_inter_values = np.array(shap_inter_values)
            shap.dependence_plot((ind, interaction_index),
                                 shap_inter_values,
                                 features=X_train,
                                 display_features=X_train, show=False, **kwargs)
            save_fig('inter_dependence_{}_{}'.format(ind, str(time.time()).split('.')[0]))
            return

    except Exception as err:
        print('Error: model is not supported by SHAP dependence plot')
        err_logging(err)
        raise Exception(err)

