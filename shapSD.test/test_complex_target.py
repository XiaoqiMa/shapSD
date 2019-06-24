from shapSD.model_init import *
import lightgbm as lgb
import shap
import numpy as np
from shapSD.logging_custom import *
import pysubgroup.pysubgroup as ps
from shapSD.utils import save_dataframe

np.warnings.filterwarnings('ignore')


@execution_time_logging
def test_complex_target(attr):
    file_path = '../data/adult.csv'
    X_train, y = get_data(file_path)
    params = {
        "max_bin": 512,
        "learning_rate": 0.05,
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 10,
        "verbose": -1,
        "min_data": 100,
        "boost_from_average": True
    }
    d_train = lgb.Dataset(X_train, label=y)
    model = lgb.train(params, d_train, 100, verbose_eval=1000)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    X = pd.read_csv(file_path, index_col=0).drop('income', axis=1)
    attr_index = list(X.columns).index(attr)
    X['shap_values'] = shap_values[:, attr_index]

    target = ps.ComplexTarget((attr, 'shap_values'))
    search_space = ps.create_nominal_selectors(X, ignore=[attr, 'shap_values', 'native-country'])
    # task = ps.SubgroupDiscoveryTask(X, target, search_space, qf=ps.CorrelationQF('significance_test'))
    task = ps.SubgroupDiscoveryTask(X, target, search_space, qf=ps.CorrelationQF('entropy'))
    result = ps.BeamSearch().execute(task)
    result = ps.overlap_filter(result, X, similarity_level=0.9)

    df = ps.results_as_df(X, result, statistics_to_show=ps.complex_statistics, complex_target=True)
    save_dataframe(df, 'shap_subgroup.csv', description='education-num & education-num shapley values as target')
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(df)


@execution_time_logging
def test_shap_values(attr=None):
    file_path = '../data/adult.csv'
    X_train, y = get_data(file_path)
    params = {
        "max_bin": 512,
        "learning_rate": 0.05,
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "binary_logloss",
        "num_leaves": 10,
        "verbose": -1,
        "min_data": 100,
        "boost_from_average": True
    }
    d_train = lgb.Dataset(X_train, label=y)
    model = lgb.train(params, d_train, 100, verbose_eval=1000)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    X = pd.read_csv(file_path, index_col=0).drop('income', axis=1)
    if attr is None:
        # default use the sum of shapley values of the single instance
        X['shap_values'] = np.sum(shap_values, axis=1)
    else:
        attr_index = list(X.columns).index(attr)
        X['shap_values'] = shap_values[:, attr_index]

    target = ps.NumericTarget('shap_values')
    # search_space = ps.create_nominal_selectors(X, ignore=['shap_values'])
    search_space = ps.create_selectors(X, ignore=['shap_values'])
    task = ps.SubgroupDiscoveryTask(X, target, search_space, qf=ps.StandardQFNumeric(1), result_set_size=20)
    result = ps.BeamSearch(beam_width=30).execute(task)
    result = ps.overlap_filter(result, X, similarity_level=0.8)

    df = ps.results_as_df(X, result, statistics_to_show=ps.all_statistics_numeric)
    save_dataframe(df, 'shap_subgroup.csv', description='education-num shapley values as target')
    # for (q, sg) in result:
    #     print(str(q) + ":\t" + str(sg.subgroup_description))
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(df)



test_complex_target('education-num')
