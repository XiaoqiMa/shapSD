from shapSD.encoding import *
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# import xgboost as xgb
import lightgbm as lgb


def read_data(file_path):
    data = pd.read_csv(file_path, index_col=0)
    return data


def get_data(file_path):
    df = read_data(file_path)
    data = label_encoding(df)
    target_col = data.columns[len(data.columns) - 1]
    X_train = data.drop(target_col, axis=1)
    y = data[target_col]
    return X_train, y


def rf_clf_model(X_train, y, **kwargs):
    model = RandomForestClassifier(random_state=0, n_estimators=20, **kwargs)
    model.fit(X_train, y)
    return model


def rf_reg_model(X_train, y, **kwargs):
    model = RandomForestRegressor(random_state=0, n_estimators=20, **kwargs)
    model.fit(X_train, y)
    return model


def lgb_clf_model(X_train, y, **kwargs):
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
    return model


def lgb_reg_model(X_train, y, **kwargs):
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
    d_train = lgb.Dataset(X_train, label=y)
    model = lgb.train(params, d_train, 100, verbose_eval=1000)
    return model

# def xgb_clf_model(X_train, y, **kwargs):
#     xgc = xgb.XGBClassifier(n_estimators=50, max_depth=20, base_score=0.5,
#                             objective='binary:logistic', random_state=42, **kwargs)
#     xgc.fit(X_train, y)
#     return xgc
#
#
# def xgb_reg_model(X_train, y, **kwargs):
#     xgr = xgb.XGBRegressor(n_estimators=50, max_depth=20, base_score=0.5, random_state=42)
#     xgr.fit(X_train, y)
#     return xgr
