#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(r"../")
from datetime import datetime

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

from data_loader import ctb_data_loader

# In[2]:


IS_CUSTOM = True
USE_VALID = True
DROPS = [
    "assessmentItemID",
    "testId",
    "Timestamp",
    "year",
    "day",
    "minute",
    "second",
    # 'userID',
    # 'KnowledgeTag',
    # 'userAnswerRate',
    # 'tagAnswerRate',
    # 'itemAnswerRate',
    # 'testAnswerRate',
    # 'timeConcentrationRate',
    # 'monthAnswerRate',
    "userSolvedLen",
    "testSolvedLen",
    "tagSolvedLen",
    "itemSolvedLen",
    "timeConcentrationCount",
    "monthSolvedCount",
    # 'userSolvedSum',
    # 'itemSolvedSum',
    # 'testSolvedSum',
    # 'tagSolvedSum',
    # 'timeConcentraionSum',
    # 'monthSolvedSum',
    # 'testSolvedVar',
    # 'userSolvedVar',
    # 'tagSolvedVar',
    # 'itemSolvedVar',
    # 'timeConcentrationVar',
    # 'monthSolvedVar',
    # 'timeConcentrationLevel',
    # 'month',
    # 'hour',
    # 'first3',
    # 'mid3',
    # 'last3',
    # 'user_correct_answer',
    # 'user_total_answer',
    # 'user_acc',
]


# # CTB preprocessing

# In[3]:


x_train, x_valid, y_train, y_valid, test = ctb_data_loader(
    IS_CUSTOM=IS_CUSTOM, USE_VALID=USE_VALID, DROPS=DROPS
)
cat_features = x_train.columns[x_train.nunique() < 200].to_list()


# # CatBoostClassifier

# In[4]:


model = CatBoostClassifier(
    cat_features=cat_features,
    task_type="GPU",
    leaf_estimation_iterations=10,
    od_type="Iter",
    logging_level="Silent",
)
# , logging_level="Silent","Info"
if USE_VALID:
    model = CatBoostClassifier(
        cat_features=cat_features,
        task_type="GPU",
        use_best_model=True,
        leaf_estimation_iterations=10,
        od_type="Iter",
        early_stopping_rounds=100,
        logging_level="Silent",
    )

predict = None
param_grid = {
    "iterations": [200],
    "depth": [10],
    "learning_rate": [0.1],
    # 'custom_loss': ['TotalF1'],
    # 'eval_metric': ['TotalF1'],
    # 'random_seed': [42],
    # 'min_data_in_leaf': [3,4,5],
    # 'max_leaves': [200],
    # 'l2_leaf_reg': [100],
    # 'border_count': [100],
    # 'bagging_temperature': [500],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# cv = KFold(n_splits=6, random_state=42, shuffle=True)

gcv = GridSearchCV(
    model,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=1,
    pre_dispatch=8,
    # verbose=1,
)


# In[5]:


show = False
if USE_VALID:
    gcv.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], plot=show, verbose=show)
else:
    gcv.fit(x_train, y_train, plot=show, verbose=show)
# , verbose=2, silent=False
print("final params", gcv.best_params_)
print("best score", gcv.best_score_)


# In[6]:


# SAVE OUTPUT
model = gcv.best_estimator_
sub = pd.read_csv("/opt/ml/input/data/sample_submission.csv")
sub["prediction"] = model.predict(test.drop("answerCode", axis=1))

output_dir = "./output/"
now = datetime.now()
now_str = "{:%Y-%m-%d_%H:%M:%S}".format(now)
file_name = f"CTB_C_grid_kfold_custom_submission_{now_str}.csv"
write_path = os.path.join(output_dir, file_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(write_path, "w", encoding="utf8") as w:
    print("writing prediction : {}".format(write_path))
    w.write("id,prediction\n")
    for id, p in enumerate(sub["prediction"]):
        w.write("{},{}\n".format(id, p))


# In[7]:


def get_accuracy(PRED_PATH=file_name):
    threshold = 0.5
    ANSWER_PATH = "../../data/custom_answer.csv"

    submission_result = pd.read_csv(PRED_PATH)
    answer = pd.read_csv(ANSWER_PATH)

    y_pred, y = submission_result["prediction"], answer["prediction"]

    return f"accuracy_score: {accuracy_score(y,y_pred.apply(lambda x: 1 if x > threshold else 0))}\nroc  auc_score: {roc_auc_score(y,y_pred)}"


# In[8]:


if IS_CUSTOM:
    print(get_accuracy("output/" + file_name))


# In[9]:


ft_importance_values = model.feature_importances_

# 정렬과 시각화를 쉽게 하기 위해 series 전환
ft_series = pd.Series(ft_importance_values, index=x_train.columns)
ft_top20 = ft_series.sort_values(ascending=False)[:20]

# 시각화
plt.figure(figsize=(8, 6))
plt.title("Feature Importance Top 20")
sns.barplot(x=ft_top20, y=ft_top20.index)
plt.show()


# In[ ]:
