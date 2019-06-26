import unittest
from shapSD.model_init import *
import pysubgroup.pysubgroup as ps
from shapSD.utils import save_dataframe
from shapSD.data_encoding import DataEncoder
from shapSD.numeric_perturb import NumericPerturb
import warnings
import pandas as pd
import time

warnings.filterwarnings('ignore')


class TestNumericPerturb(unittest.TestCase):

    @staticmethod
    def read_data():
        file_path = '../data/adult.csv'
        return pd.read_csv(file_path, index_col=0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        adult = self.read_data()
        df_adult = DataEncoder(adult).label_encoding()
        self.x_train = df_adult.drop('income', axis=1)
        self.y_train = df_adult['income']
        self.perturb_attr = 'age'
        self.model = InitializeModel(self.x_train, self.y_train).lgb_clf_model()
        self.n_perturb = NumericPerturb(self.x_train, self.y_train, self.model, self.perturb_attr)

    def test_get_prediction(self):
        prediction = self.n_perturb.get_prediction(self.x_train)
        self.assertEqual(len(prediction), len(self.y_train))

    def test_calc_perturb_effect(self):
        df_perturb = self.n_perturb.calc_perturb_effect()
        self.assertEqual(len(df_perturb.columns), len(self.x_train.columns) + 2)

    def test_subgroup_discovery(self):
        new_adult = self.read_data()
        df_perturb = self.n_perturb.calc_perturb_effect()

        attr_name = '{}_change'.format(self.perturb_attr)
        pred_attr_name = '{}_prediction_change'.format(self.perturb_attr)
        self.assertEqual(attr_name, 'age_change')
        self.assertEqual(pred_attr_name, 'age_prediction_change')
        new_adult[attr_name] = df_perturb[attr_name]
        new_adult[pred_attr_name] = df_perturb[pred_attr_name]

        target = ps.ComplexTarget((attr_name, pred_attr_name))
        search_space = ps.create_nominal_selectors(new_adult, ignore=[attr_name, pred_attr_name, 'income'])
        task = ps.SubgroupDiscoveryTask(new_adult, target, search_space, qf=ps.CorrelationQF('entropy'))
        result = ps.BeamSearch().execute(task)
        df_result = ps.results_as_df(new_adult, result, statistics_to_show=ps.complex_statistics, complex_target=True)
        self.assertIsNotNone(df_result)

        description = '{}: perturb {} value, use perturb effect to conduct ' \
                      'subgroup discover'.format(time.ctime(), self.perturb_attr)
        save_dataframe(df_result, 'numeric_perturb.csv', description=description)


if __name__ == "__main__":
    unittest.main()
