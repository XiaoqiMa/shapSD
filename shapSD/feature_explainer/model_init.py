"""
provides methods to initialize models, support RandomForest, LightGBM, XGBoost and etc.
author: Xiaoqi
date: 2019.06.24
"""
# import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import warnings
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l2
from keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')


class InitializeModel(object):

    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def rf_clf_model(self, **kwargs):
        model = RandomForestClassifier(random_state=0, n_estimators=30, **kwargs)
        model.fit(self.x_train, self.y_train)
        return model

    def rf_reg_model(self, **kwargs):
        model = RandomForestRegressor(random_state=0, n_estimators=30, **kwargs)
        model.fit(self.x_train, self.y_train)
        return model

    def lgb_clf_model(self, **kwargs):
        model = lgb.LGBMRegressor(max_bin=512,
                                  learning_rate=0.01,
                                  n_estimators=500,
                                  boosting_type="gbdt",
                                  objective="binary",
                                  num_leaves=10,
                                  verbose=-1,
                                  random_state=42)
        model.fit(self.x_train, self.y_train, **kwargs)
        return model

    def lgb_reg_model(self, **kwargs):
        model = lgb.LGBMRegressor(max_bin=512,
                                  learning_rate=0.01,
                                  n_estimators=500,
                                  boosting_type="gbdt",
                                  objective="regression",
                                  num_leaves=10,
                                  verbose=-1,
                                  random_state=42)
        model.fit(self.x_train, self.y_train, **kwargs)
        return model

    def keras_nn_model(self, **kwargs):
        model = Sequential()
        model.add(Dense(100, activation='relu', input_dim=len(self.x_train), kernel_regularizer=l2(0.01)))
        model.add(Dense(100, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
        model.fit(self.x_train, self.y_train, epochs=100, batch_size=50, verbose=0,
                  callbacks=[es], **kwargs)
        return model

    # def xgb_clf_model(self, **kwargs):
    #     xgc = xgb.XGBClassifier(n_estimators=50, max_depth=20, base_score=0.5,
    #                             objective='binary:logistic', random_state=42, **kwargs)
    #     xgc.fit(self.x_train, self.y_train)
    #     return xgc
    #
    # def xgb_reg_model(self, **kwargs):
    #     xgr = xgb.XGBRegressor(n_estimators=50, max_depth=20, base_score=0.5,
    #                            objective='reg:logistic', random_state=42, **kwargs)
    #     xgr.fit(self.x_train, self.y_train)
    #     return xgr
