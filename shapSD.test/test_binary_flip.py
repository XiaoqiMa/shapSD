import unittest
from sklearn.ensemble import RandomForestClassifier
from shapSD.binary_flip import *
from shapSD.encoding import *


class TestBinaryFlip(unittest.TestCase):

    @staticmethod
    def read_data():
        file_path = '../data/adult.csv'
        adult = pd.read_csv(file_path, index_col=0)
        return adult

    def model_init(self):
        adult = self.read_data()
        data = label_encoding(adult)
        X_train = data.drop('income', axis=1)
        y = data['income']
        model = RandomForestClassifier(random_state=0, n_estimators=10, max_depth=10)
        model.fit(X_train, y)
        return X_train, y, model

    def test_get_prediction(self):
        X_train, y, model = self.model_init()
        prediction = get_prediction(X_train, model)
        self.assertEqual(len(prediction), len(y))

    def test_calc_flip_effect(self):
        X_train, y, model = self.model_init()
        flip_attr = 'sex'
        df_effect = calc_flip_effect(X_train, flip_attr, model)
        self.assertEqual(len(df_effect.columns), len(X_train.columns) + 1)
