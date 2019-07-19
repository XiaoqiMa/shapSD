import unittest
from shapSD.feature_explainer.model_init import *
from shapSD.feature_explainer.data_encoding import DataEncoder
from shapSD.feature_explainer.shap_values import ShapValues
import warnings
import pandas as pd

warnings.filterwarnings('ignore')


class TestShapValues(unittest.TestCase):

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
        self.model = InitializeModel(self.x_train, self.y_train).lgb_clf_model()
        self.shaper = ShapValues(self.x_train, self.model)

    def test_calc_shap_values(self):
        exp, shap_v, expected_v = self.shaper.calc_shap_values()
        self.assertEqual(len(shap_v), len(self.x_train))

    def test_calc_shap_inter_values(self):
        exp, shap_inter, expected_v = self.shaper.calc_shap_inter_values()
        self.assertEqual(len(shap_inter), len(self.x_train))

    def test_shap_force_plot(self):
        self.shaper.shap_force_plot(instance_ind=0)

    def test_shap_summary_plot(self):
        self.shaper.shap_summary_plot()

    def test_shap_dependence_plot(self):
        self.shaper.shap_dependence_plot('education-num', 'education-num')


if __name__ == "__main__":
    unittest.main()
