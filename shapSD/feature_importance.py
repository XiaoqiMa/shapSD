import numpy as np
import pandas as pd
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.metrics import mean_squared_error

from shapSD.log_execution_time import *
from shapSD.model_init import *

@func_logging
def raw_perm_importance(X_train, y_train, model, is_classification=True, iteration=10):
    imp = []
    imp_var = []
    if is_classification:
        base_score = model.score(X_train, y_train)
        for col in X_train.columns:
            scores = []
            for m in range(iteration):
                x = X_train.copy()
                x[col] = np.random.permutation(x[col])
                score = model.score(x, y_train)
                scores.append(score)
            score_drop_list = np.array(base_score) - np.array(scores)
            variance = np.round(np.var(score_drop_list), 6)
            score_mean_drop = np.round(np.mean(score_drop_list), 4)
            imp.append(score_mean_drop)
            imp_var.append('{}±{}'.format(score_mean_drop, variance))
    else:
        y_pred = model.predict(X_train)
        base_rmse = mean_squared_error(y_train, y_pred)
        for col in X_train.columns:
            scores = []
            for m in range(iteration):
                x = X_train.copy()
                x[col] = np.random.permutation(x[col])
                pred = model.predict(x)
                rmse_score = mean_squared_error(y_train, pred)
                scores.append(rmse_score)
            rmse_score_list = np.array(base_rmse) - np.array(scores)
            rmse_variance = np.round(np.var(rmse_score_list), 6)
            rmse_mean = np.round(np.mean(rmse_score_list), 4)
            imp.append(rmse_mean)
            imp_var.append('{}±{}'.format(rmse_mean, rmse_variance))

    df_imp = pd.DataFrame({'Features': X_train.columns, 'Importance': imp, 'Importance weights': imp_var})
    df_imp = df_imp.sort_values('Importance', ascending=False)
    return df_imp[['Importance weights', 'Features']]

@func_logging
def eli5_perm_importance(X_train, y_train, model, is_classification=True):
    if is_classification:
        perm = PermutationImportance(model, random_state=0).fit(X_train, y_train)
        return eli5.show_weights(perm, feature_names=X_train.columns.tolist())

file_path = '../data/adult.csv'
X_train, y = get_data(file_path)
model = train_rf_model(X_train, y)
# print(raw_perm_importance(X_train, y, model))
print(eli5_perm_importance(X_train, y, model))