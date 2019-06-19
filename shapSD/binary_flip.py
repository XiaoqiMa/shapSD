import numpy as np


def get_prediction(X_train, model, is_classification=True):
    try:
        if is_classification:
            predictions = model.predict_proba(X_train)[:, 0]  # classification task
        else:
            predictions = model.predict(X_train)  # regression task
        return predictions

    except:
        raise Exception('Model does not support probability prediction')


def calc_flip_effect(X_train, flip_attr, model, is_classification=True):
    ori_prediction = get_prediction(X_train, model, is_classification)
    X_flip = X_train.copy()
    X_flip[flip_attr] = X_flip[flip_attr].apply(lambda x: x ^ 1)
    new_prediction = get_prediction(X_flip, model, is_classification)

    avg_effect = np.mean(np.abs(new_prediction - ori_prediction))
    X_flip['effect'] = np.abs(new_prediction - ori_prediction) - avg_effect
    return X_flip
