import unittest
import pandas as pd
from shapSD.feature_explainer.data_encoding import DataEncoder
import warnings

warnings.filterwarnings('ignore')


class TestDataEncoding(unittest.TestCase):

    @staticmethod
    def read_data():
        file_path = '../../data/adult.csv'
        adult = pd.read_csv(file_path, index_col=0)
        return adult

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = DataEncoder(self.read_data())
        self.assertTrue(isinstance(self.encoder, DataEncoder))

    def test_label_encoding(self):
        data = self.encoder.label_encoding()
        # make sure all columns are numeric type
        num_col_len = len(data.select_dtypes(['number']).columns)
        self.assertEqual(num_col_len, len(data.columns))

    def test_num_discretization(self):
        data = self.encoder.num_discretization()
        # make sure all columns are str type
        cat_col_len = len(data.select_dtypes(exclude=['number']).columns)
        self.assertEqual(cat_col_len, len(data.columns))


if __name__ == '__main__':
    unittest.main()
