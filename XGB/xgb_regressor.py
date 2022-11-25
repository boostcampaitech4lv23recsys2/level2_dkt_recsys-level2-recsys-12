import pandas as pd
import os
import sys

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import accuracy_score, roc_auc_score

from xgboost import XGBRegressor

from datetime import datetime
import numpy as np

sys.path.append(r"../")
from feature_engineering import feature_engineering
IS_CUSTOM = True


def load_xgb_data(basepath="../../data/"):
    """Load data for xgboost"""
    path1 = os.path.join(basepath, "train_data.csv")
    path2 = os.path.join(basepath, "test_data.csv")
    if IS_CUSTOM:
        print("Load custom...")
        path2 = os.path.join(basepath, "custom_test_data.csv")
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)
    data = pd.concat([data1, data2])
    data = data.sort_values(["userID", "Timestamp"])
    data.drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last", inplace=True
    )
    data = feature_engineering(data)
    return data


xgb_data = load_xgb_data()

drops = ["year", "day", "minute", "second"]


xgb_data = xgb_data.drop(drops, axis=1)
xgb_data.head()

def split_train_valid_test(data):
    train_df = data[data["answerCode"]!=-1]
    test_df = data[data["answerCode"]==-1]
    valid = data[data["answerCode"] != -1]
    valid = valid[valid["userID"] != valid["userID"].shift(-1)]
    train = pd.merge(
        train_df, valid, how='outer', indicator=True).query(
            '_merge == "right_only"'
        ).drop(columns=["_merge"])
    return train_df, valid, test_df


train, valid, test = split_train_valid_test(xgb_data)

x_train = train.drop(["answerCode"], axis=1)
y_train = train["answerCode"]
x_valid = valid.drop(["answerCode"], axis=1)
y_valid = valid["answerCode"]


DROPS = ["assessmentItemID", "testId", "Timestamp"]


def xgb_preprocessing(data):
    data = data.drop(DROPS, axis=1)
    for col in data.columns:
        data[col] = data[col].astype(float)
    return data


x_train = xgb_preprocessing(x_train)
x_valid = xgb_preprocessing(x_valid)
test = xgb_preprocessing(test)


model = XGBRegressor(tree_method="gpu_hist", gpu_id=0, early_stopping_rounds=100)

param_grid = {
    "booster": ["gbtree"],
    "colsample_bylevel": [0.8],
    "colsample_bytree": [0.8],
    "gamma": [3],
    "max_depth": [9],
    "min_child_weight": [3],
    "n_estimators": [100],
    "nthread": [4],
    "objective": ["binary:logistic"],
    "random_state": [42],
    "verbosity": [1],
}
cv = KFold(n_splits=5, random_state=42, shuffle=True)

gcv = GridSearchCV(
    model,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=4,
    pre_dispatch=8,
    verbose=1,
)


gcv.fit(
    x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)], verbose=True
)
print("final params", gcv.best_params_)
print("best score", gcv.best_score_)


# SAVE OUTPUT
model = gcv.best_estimator_
sub = pd.read_csv("/opt/ml/input/data/sample_submission.csv")
sub["prediction"] = model.predict(test.drop("answerCode", axis=1))

output_dir = "./output/"
file_name = f"XGB_grid_kfold_custom_submission_{datetime.now().microsecond}.csv"
write_path = os.path.join(output_dir, file_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(write_path, "w", encoding="utf8") as w:
    print("writing prediction : {}".format(write_path))
    w.write("id,prediction\n")
    for id, p in enumerate(sub["prediction"]):
        w.write("{},{}\n".format(id, p))

def get_accuracy(PRED_PATH=file_name):
    threshold = 0.5
    ANSWER_PATH = "../../data/custom_answer.csv"

    submission_result = pd.read_csv(PRED_PATH)
    answer = pd.read_csv(ANSWER_PATH)

    y_pred, y = submission_result["prediction"], answer["prediction"]

    return f"accuracy_score: {accuracy_score(y,y_pred.apply(lambda x: 1 if x > threshold else 0))}\nroc  auc_score: {roc_auc_score(y,y_pred)}"

if IS_CUSTOM:
    print(get_accuracy("output/" + file_name))

