import unittest
from shapSD.encoding import *

class TestEncoding(unittest.TestCase):

    @staticmethod
    def read_data():
        file_path = '../data/adult.csv'
        adult = pd.read_csv(file_path, index_col=0)
        return adult

    def test_label_encoding(self):
        adult_data = self.read_data()
        data = label_encoding(adult_data)
        # make sure all columns are numeric type
        num_col_len = len(data.select_dtypes(['number']).columns)
        self.assertEqual(num_col_len, len(data.columns))

    def test_num_discretization(self):
        adult_data = self.read_data()
        data = num_discretization(adult_data)
        # make sure all columns are str type
        cat_col_len = len(data.select_dtypes(exclude=['number']).columns)
        self.assertEqual(cat_col_len, len(data.columns))


if __name__ == '__main__':
    unittest.main()
