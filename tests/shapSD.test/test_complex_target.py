import shapSD.pysubgroup as ps
from shapSD.feature_explainer.utils import save_dataframe
from shapSD.feature_explainer.data_encoding import DataEncoder
from shapSD.feature_explainer.model_init import InitializeModel
from shapSD.feature_explainer.shap_explainer import ShapExplainer
from shapSD.feature_explainer.logging_custom import execution_time_logging
import pandas as pd
import time


@execution_time_logging
def test_complex_target(attr):
    adult = pd.read_csv('../../data/adult.csv', index_col=0)
    df_adult = DataEncoder(adult).label_encoding()
    x_train = df_adult.drop('income', axis=1)
    y_train = df_adult['income']

    model = InitializeModel(x_train, y_train).lgb_clf_model()
    shaper = ShapExplainer(x_train, model)
    exp, shap_v, expected_v = shaper.calc_shap_values(attr=attr)

    new_adult = adult.copy()[:2000]
    attr_shap_name = '{}_shap_values'.format(attr)
    new_adult[attr_shap_name] = shap_v[:2000]
    assert len(new_adult) == 2000
    save_dataframe(new_adult.head(), 'shap_subgroup.csv', description='adult dataset example')

    target = ps.ComplexTarget((attr, attr_shap_name))
    search_space = ps.create_nominal_selectors(new_adult, ignore=[attr, attr_shap_name, 'income'])
    task = ps.SubgroupDiscoveryTask(new_adult, target, search_space, qf=ps.CorrelationQF('entropy'))
    result = ps.BeamSearch().execute(task)
    # result = ps.overlap_filter(result, new_adult, similarity_level=0.9)

    df = ps.results_as_df(new_adult, result, statistics_to_show=ps.complex_statistics, complex_target=True)
    description = '{}: use {} feature value and its shapley value to conduct ' \
                  'subgroup discover'.format(time.ctime(), attr)
    save_dataframe(df, 'shap_subgroup.csv', description=description)


test_complex_target('education-num')
