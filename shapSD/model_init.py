"""
provides methods to initialize models, support RandomForest, LightGBM, XGBoost and etc.
author: Xiaoqi
date: 2019.06.24
"""
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb


class InitializeModel(object):

    def __init__(self, x_train, y_train, model_type, pred_type='classification', **kwargs):
        """
        Parameters
        -----------
        x_train: pandas dataframe, input train data
        y_train: pandas dataframe, input train target
        model_type: str
                    rf: random forest
                    lightgbm: a gradient boosting framework
                    xgboost: an optimized distributed gradient boosting model
        pred_type: str, prediction type
                    classification: classification tasks for prediction
                    regression: regression tasks for prediction
        kwargs: optional

        Return:
        ----------
        model: classification or regression model
        """
        self.x_train = x_train
        self.y_train = y_train
        if pred_type == 'classification':
            if model_type == 'rf':
                self.rf_clf_model(**kwargs)
            elif model_type == 'lightgbm':
                self.lgb_clf_model(**kwargs)
            elif model_type == 'xgboost':
                self.xgb_clf_model(**kwargs)
            else:
                raise Exception('Cannot initialize {} classification model'.format(model_type))
        elif pred_type == 'regression':
            if model_type == 'rf':
                self.rf_reg_model(**kwargs)
            elif model_type == 'lightgbm':
                self.lgb_reg_model(**kwargs)
            elif model_type == 'xgboost':
                self.xgb_reg_model(**kwargs)
            else:
                raise Exception('Cannot initialize {} regression model'.format(model_type))
        else:
            raise Exception('Does not support model initialization')

    def rf_clf_model(self, **kwargs):
        model = RandomForestClassifier(random_state=0, n_estimators=30, **kwargs)
        model.fit(self.x_train, self.y_train)
        return model

    def rf_reg_model(self, **kwargs):
        model = RandomForestRegressor(random_state=0, n_estimators=30, **kwargs)
        model.fit(self.x_train, self.y_train)
        return model

    def lgb_clf_model(self, **kwargs):
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
        d_train = lgb.Dataset(self.x_train, label=self.y_train)
        model = lgb.train(params, d_train, 100, verbose_eval=1000, **kwargs)
        return model

    def lgb_reg_model(self, **kwargs):
        params = {
            "max_bin": 512,
            "learning_rate": 0.05,
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "binary_logloss",
            "num_leaves": 10,
            "verbose": -1,
            "min_data": 100,
            "boost_from_average": True
        }
        d_train = lgb.Dataset(self.x_train, label=self.y_train)
        model = lgb.train(params, d_train, 100, verbose_eval=1000, **kwargs)
        return model

    def xgb_clf_model(self, **kwargs):
        xgc = xgb.XGBClassifier(n_estimators=50, max_depth=20, base_score=0.5,
                                objective='binary:logistic', random_state=42, **kwargs)
        xgc.fit(self.x_train, self.y_train)
        return xgc

    def xgb_reg_model(self, **kwargs):
        xgr = xgb.XGBRegressor(n_estimators=50, max_depth=20, base_score=0.5,
                               objective='reg:logistic', random_state=42, **kwargs)
        xgr.fit(self.x_train, self.y_train)
        return xgr
