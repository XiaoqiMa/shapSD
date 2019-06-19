from shapSD.encoding import *
from sklearn.ensemble import RandomForestClassifier


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


def train_rf_model(X_train, y):
    model = RandomForestClassifier(random_state=0, n_estimators=10)
    model.fit(X_train, y)
    return model
