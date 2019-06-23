import unittest
from shapSD.numeric_perturb import *
from shapSD.model_init import *
import pysubgroup.pysubgroup as ps
import lightgbm as lgb

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.warnings.filterwarnings('ignore')


class TestNumericPerturb(unittest.TestCase):

    def model_init(self):
        file_path = '../data/adult.csv'
        X_train, y = get_data(file_path)
        model = RandomForestClassifier(random_state=0, n_estimators=10, max_depth=10)
        model.fit(X_train, y)
        return X_train, y, model

    def lgb_model_init(self):
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
        file_path = '../data/adult.csv'
        X_train, y = get_data(file_path)
        d_train = lgb.Dataset(X_train, label=y)
        model = lgb.train(params, d_train, 100, verbose_eval=1000)
        return X_train, y, model

    def test_get_prediction(self):
        X_train, y, model = self.model_init()
        prediction = get_prediction(X_train, model)
        self.assertEqual(len(prediction), len(y))

    def test_perturb_flip_effect(self):
        # X_train, y, model = self.model_init()
        X_train, y, model = self.lgb_model_init()
        perturb_attr = 'age'
        df_effect = calc_perturb_effect(X_train, perturb_attr, model)
        self.assertEqual(len(df_effect.columns), len(X_train.columns) + 2)

    def test_subgroup_discovery(self):
        np.random.seed(0)
        adult = pd.read_csv('../data/adult.csv', index_col=0)
        # X_train, y, model = self.model_init()
        X_train, y, model = self.lgb_model_init()
        perturb_attr = 'age'
        df_effect = calc_perturb_effect(X_train, perturb_attr, model)
        new_adult = adult.drop('income', axis=1)
        new_adult['attr_change'] = df_effect['attr_change']
        new_adult['pred_change'] = df_effect['pred_change']

        target = ps.ComplexTarget(('attr_change', 'pred_change'))
        search_space = ps.create_nominal_selectors(new_adult, ignore=['attr_change', 'pred_change'])
        task = ps.SubgroupDiscoveryTask(new_adult, target, search_space, qf=ps.CorrelationQF('entropy'))
        result = ps.BeamSearch().execute(task)
        # result = ps.overlap_filter(result, new_adult, similarity_level=0.7)
        df_result = ps.results_as_df(new_adult, result, statistics_to_show=ps.complex_statistics, complex_target=True)
        print(df_result)
