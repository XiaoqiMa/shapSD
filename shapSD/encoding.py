import pandas as pd


def label_encoding(df_data):
    try:
        data = df_data.copy()
        cat_cols = data.select_dtypes(['category']).columns
        if len(cat_cols) == 0:
            cat_cols = data.select_dtypes(exclude=['number']).columns
            data[cat_cols] = data[cat_cols].astype('category')
        data[cat_cols] = data[cat_cols].apply(lambda x: x.cat.codes)
        return data
    except:
        raise Exception('Label encoding error')


def onehot_encoding(df_data):
    data = df_data.copy()
    return pd.get_dummies(data)


def num_discretization(df_data, quantiles=10):
    data = df_data.copy()
    num_cols = data.select_dtypes(['number']).columns
    data[num_cols] = data[num_cols].apply(lambda x: pd.qcut(x, quantiles, duplicates='drop').astype(str))
    return data
