import pandas as pd


def load_data(basepath=""):
    path1 = os.path.join(basepath, "train_data.csv")
    path2 = os.path.join(basepath, "test_data.csv")
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)
    data = pd.concat([data1, data2])
    data.drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last", inplace=True
    )
    data = feature_engineering(data)
    return data


def separate_data(data=load_data()):
    train_data = data[data.answerCode >= 0]
    test_data = data[data.answerCode < 0]
    return train_data, test_data


def xgboost_preprocessing():
    train, test = separate_data()
    for col in train.columns:
        train[col] = train[col].astype(float)
        test[col] = test[col].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        train.drop("answerCode", axis=1), train["answerCode"], test_size=0.2
    )
    return X_train, X_test, y_train, y_test
