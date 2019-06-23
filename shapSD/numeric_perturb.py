import numpy as np


def get_prediction(X_train, model):
    try:
        predictions = model.predict_proba(X_train)  # classification task
        return predictions[:, 0]
    except AttributeError:
        predictions = model.predict(X_train)  # regression task
        return predictions
    except Exception:
        raise Exception('Model does not support probability prediction')


def calc_perturb_effect(X_train, perturb_attr, model):
    ori_prediction = get_prediction(X_train, model)
    X_perturb = X_train.copy()
    X_perturb[perturb_attr] = np.random.permutation(X_perturb[perturb_attr].values)
    new_prediction = get_prediction(X_perturb, model)

    X_perturb['attr_change'] = np.abs(X_perturb[perturb_attr] - X_train[perturb_attr])
    X_perturb['pred_change'] = np.abs(new_prediction - ori_prediction)

    return X_perturb
