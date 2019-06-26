import unittest
import pandas as pd
from shapSD.feature_explainer.binary_flip import BinaryFlip
from shapSD.feature_explainer.model_init import InitializeModel
from shapSD.feature_explainer.data_encoding import DataEncoder
from shapSD.feature_explainer.utils import save_dataframe
import shapSD.pysubgroup as ps
import time


class TestBinaryFlip(unittest.TestCase):

    @staticmethod
    def read_data():
        file_path = '../../data/adult.csv'
        return pd.read_csv(file_path, index_col=0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        adult = self.read_data()
        df_adult = DataEncoder(adult).label_encoding()
        self.x_train = df_adult.drop('income', axis=1)
        self.y_train = df_adult['income']
        self.flip_attr = 'sex'
        self.model = InitializeModel(self.x_train, self.y_train).lgb_clf_model()
        self.b_flip = BinaryFlip(self.x_train, self.y_train, self.model, self.flip_attr)

    def test_get_prediction(self):
        prediction = self.b_flip.get_prediction(self.x_train)
        self.assertEqual(len(prediction), len(self.y_train))

    def test_calc_flip_effect(self):
        df_flip_effect = self.b_flip.calc_flip_effect()
        self.assertEqual(len(df_flip_effect.columns), len(self.x_train.columns) + 1)

    def test_calc_flip_shap_values(self):
        df_flip_shap = self.b_flip.calc_flip_shap_values()
        self.assertEqual(len(df_flip_shap.columns), len(self.x_train.columns) + 1)

    def test_effect_discovery(self):
        df_flip_effect = self.b_flip.calc_flip_effect()[:1000]
        new_adult = self.read_data()[:1000]
        self.assertEqual(df_flip_effect.shape, new_adult.shape)
        attr_name = '{}_effect'.format(self.flip_attr)
        new_adult[attr_name] = df_flip_effect[attr_name]
        self.assertEqual(attr_name, 'sex_effect')

        target = ps.NumericTarget(attr_name)
        search_space = ps.create_selectors(new_adult, ignore=[attr_name, self.flip_attr, 'income'])
        task = ps.SubgroupDiscoveryTask(new_adult, target, search_space, qf=ps.StandardQFNumeric(1), result_set_size=10)
        result = ps.BeamSearch().execute(task)
        df_result = ps.results_as_df(new_adult, result, statistics_to_show=ps.all_statistics_numeric)

        self.assertIsNotNone(df_result)
        description = '{}: flip {} value, use flip effect to conduct ' \
                      'subgroup discover'.format(time.ctime(), self.flip_attr)
        save_dataframe(df_result, 'binary_subgroup.csv', description=description)

    def test_shap_discovery(self):
        df_shap_effect = self.b_flip.calc_flip_shap_values()[:1000]
        new_adult = self.read_data()[:1000]
        attr_name = '{}_shap_values'.format(self.flip_attr)
        new_adult[attr_name] = df_shap_effect[attr_name]
        self.assertEqual(attr_name, 'sex_shap_values')

        target = ps.NumericTarget(attr_name)
        search_space = ps.create_selectors(new_adult, ignore=[attr_name, self.flip_attr, 'income'])
        task = ps.SubgroupDiscoveryTask(new_adult, target, search_space, qf=ps.StandardQFNumeric(1), result_set_size=10)
        result = ps.BeamSearch().execute(task)
        df_result = ps.results_as_df(new_adult, result, statistics_to_show=ps.all_statistics_numeric)

        self.assertIsNotNone(df_result)
        description = '{}: shap {} value, use shap values to conduct ' \
                      'subgroup discover'.format(time.ctime(), self.flip_attr)
        save_dataframe(df_result, 'binary_subgroup.csv', description=description)


if __name__ == '__main__':
    unittest.main()
