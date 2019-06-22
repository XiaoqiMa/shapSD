from shapSD.model_init import *
import lightgbm as lgb
import shap
import numpy as np
from shapSD.logging_custom import *
import pysubgroup.pysubgroup as ps

np.warnings.filterwarnings('ignore')


@execution_time_logging
def test_complex():
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
    X['shap_values'] = shap_values[:, 4]

    target = ps.ComplexTarget(('education-num', 'shap_values'))
    search_space = ps.create_nominal_selectors(X, ignore=['education-num', 'shap_values', 'native-country'])
    # task = ps.SubgroupDiscoveryTask(X, target, search_space, qf=ps.CorrelationQF())
    task = ps.SubgroupDiscoveryTask(X, target, search_space, qf=ps.CorrelationQF('entropy'))
    result = ps.BeamSearch().execute(task)
    result = ps.overlap_filter(result, X, similarity_level=0.7)
    for (q, sg) in result:
        print(str(q) + ":\t" + str(sg.subgroup_description))

    df = ps.results_as_df(X, result, statistics_to_show=ps.complex_statistics, complex_target=True)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

def test_shap_values():
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
    X['shap_values'] = shap_values[:, 4]

    target = ps.NumericTarget('shap_values')
    search_space = ps.create_nominal_selectors(X, ignore=['shap_values'])
    task = ps.SubgroupDiscoveryTask(X, target, search_space, qf=ps.StandardQFNumeric(1))
    # task = ps.SubgroupDiscoveryTask(X, target, search_space, qf=ps.CorrelationQF('entropy'))
    result = ps.BeamSearch().execute(task)
    result = ps.overlap_filter(result, X, similarity_level=0.7)
    for (q, sg) in result:
        print(str(q) + ":\t" + str(sg.subgroup_description))

    df = ps.results_as_df(X, result, statistics_to_show=ps.all_statistics_numeric)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)

test_shap_values()