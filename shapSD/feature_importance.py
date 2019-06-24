import numpy as np
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.metrics import mean_squared_log_error
from shapSD.logging_custom import *
from shapSD.model_init import *
from shapSD.utils import *


@execution_time_logging
def raw_perm_importance(X_train, y_train, model, iteration=10):
    imp = []
    imp_var = []
    try:
        if hasattr(model, 'score'):
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

            df_imp = pd.DataFrame({'Features': X_train.columns, 'Importance': imp, 'Importance weights': imp_var})
            df_imp = df_imp.sort_values('Importance', ascending=False)
            return df_imp[['Importance weights', 'Features']]
        else:
            base_score = mean_squared_log_error(model.predict(X_train), y_train)
            for col in X_train.columns:
                scores = []
                for m in range(iteration):
                    x = X_train.copy()
                    x[col] = np.random.permutation(x[col])
                    score = mean_squared_log_error(model.predict(x), y_train)
                    scores.append(score)
                score_rmse_list = np.array(scores) - np.array(base_score)
                variance = np.round(np.var(score_rmse_list), 6)
                score_rmse_inc = np.round(np.mean(score_rmse_list), 4)
                imp.append(score_rmse_inc)
                imp_var.append('{}±{}'.format(score_rmse_inc, variance))

            df_imp = pd.DataFrame({'Features': X_train.columns, 'Importance': imp, 'Importance weights': imp_var})
            df_imp = df_imp.sort_values('Importance', ascending=False)
            return df_imp[['Importance weights', 'Features']]
    except Exception as err:
        print('Error: model is not supported')
        err_logging(err)
        raise Exception(err)

@execution_time_logging
def eli5_perm_importance(X_train, y_train, model, **kwargs):
    perm = PermutationImportance(model).fit(X_train, y_train)
    return eli5.show_weights(perm, feature_names=X_train.columns.tolist(), **kwargs)


def eli5_weights_importance(X_train, model, **kwargs):
    """By default, “gain” is used, that is the average gain of the feature when it is used in trees.
    Other types are “weight” - the number of times a feature is used to split the data,
    “cover” - the average coverage of the feature.
    You can pass it with `importance_type` argument"""

    try:
        weights = eli5.show_weights(model, feature_names=list(X_train.columns), **kwargs)
        return weights
    except Exception as err:
        print('Error: model is not supported')
        err_logging(err)
        raise Exception(err)


def eli5_instance_importance(X_train, instance, model, **kwargs):
    try:
        prediction = eli5.show_prediction(model, instance, show_feature_values=True,
                                          feature_names=list(X_train.columns), **kwargs)
        return prediction
    except Exception as err:
        print('Error: model is not supported')
        err_logging(err)
        raise Exception(err)

# if __name__ == '__main__':
#     file_path = '../data/adult.csv'
#     X_train, y = get_data(file_path)
#     model = rf_clf_model(X_train, y)
#     # raw_perm_importance(X_train, y, model)
#     imp = eli5_perm_importance(X_train, y, model)
#     save_img_file(imp.data, 'eli5_importance.html')
