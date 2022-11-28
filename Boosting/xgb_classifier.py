#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append(r"../")
from data_loader import xgb_data_loader
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score


from xgboost import XGBClassifier

from datetime import datetime


# # Load Preprocessed data

# In[2]:


IS_CUSTOM = True
USE_VALID = True
DROPS = [
    'assessmentItemID','testId','Timestamp','year','day','minute','second',
    # 'userID',
    # 'KnowledgeTag',
    
    # 'userAnswerRate',
    # 'tagAnswerRate',
    # 'itemAnswerRate',
    # 'testAnswerRate',
    # 'timeConcentrationRate',
    # 'monthAnswerRate',

    'userSolvedLen',
    'testSolvedLen',
    'tagSolvedLen',
    'itemSolvedLen',
    'timeConcentrationCount',
    'monthSolvedCount',

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
x_train, x_valid, y_train, y_valid, test = xgb_data_loader(IS_CUSTOM=IS_CUSTOM,USE_VALID=USE_VALID,DROPS=DROPS)


# # XGBClassifier

# In[3]:


model = XGBClassifier(tree_method="gpu_hist", gpu_id=0)

if USE_VALID:
    model = XGBClassifier(tree_method="gpu_hist", gpu_id=0, early_stopping_rounds=100)

"""
final params {'alpha': 10, 'booster': 'gbtree', 'colsample_bylevel': 0.9,
 'colsample_bytree': 0.8, 'eta': 0.05, 'gamma': 3, 'lambda': 0, 'max_depth': 9,
  'n_estimators': 200, 'nthread': 4, 'objective': 'binary:logistic', 'random_state': 42, 'verbosity': 1}
"""
param_grid = {
    "booster": ["gbtree"],
    "colsample_bylevel": [0.9],
    "colsample_bytree": [0.8],
    "gamma": [3],
    "lambda": [0],
    "alpha": [10],
    "eta": [0.05],
    "max_depth": [9],
    # "min_child_weight": [3],
    # "nrounds":[100000],
    "n_estimators": [300],
    "nthread": [4],
    "objective": ["binary:logistic"],
    "random_state": [42],
    "verbosity": [1],
}
# cv = KFold(n_splits=6, random_state=42, shuffle=True)
cv = StratifiedKFold(n_splits=6, random_state=42, shuffle=True)

gcv = GridSearchCV(
    model,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=4,
    pre_dispatch=8,
    verbose=1,
)


# In[4]:


if USE_VALID:
    gcv.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=True)
else:
    gcv.fit(x_train, y_train, verbose=True)

print("final params", gcv.best_params_)
print("best score", gcv.best_score_)


# In[5]:


# SAVE OUTPUT
model = gcv.best_estimator_
sub = pd.read_csv("/opt/ml/input/data/sample_submission.csv")
sub["prediction"] = model.predict(test.drop("answerCode", axis=1))

output_dir = "./output/"
now = datetime.now()
now_str = "{:%Y-%m-%d_%H:%M:%S}".format(now)
file_name = f"XGB_C_grid_kfold_custom_submission_{now_str}.csv"
write_path = os.path.join(output_dir, file_name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(write_path, "w", encoding="utf8") as w:
    print("writing prediction : {}".format(write_path))
    w.write("id,prediction\n")
    for id, p in enumerate(sub["prediction"]):
        w.write("{},{}\n".format(id, p))


# In[6]:


def get_accuracy(PRED_PATH=file_name):
    threshold = 0.5
    ANSWER_PATH = "../../data/custom_answer.csv"

    submission_result = pd.read_csv(PRED_PATH)
    answer = pd.read_csv(ANSWER_PATH)

    y_pred, y = submission_result["prediction"], answer["prediction"]

    return f"accuracy_score: {accuracy_score(y,y_pred.apply(lambda x: 1 if x > threshold else 0))}\nroc  auc_score: {roc_auc_score(y,y_pred)}"


# In[7]:


if IS_CUSTOM:
    print(get_accuracy("output/" + file_name))


# In[8]:


from xgboost import plot_importance
plot_importance(model)


# In[9]:


ft_importance_values = model.feature_importances_

# 정렬과 시각화를 쉽게 하기 위해 series 전환
ft_series = pd.Series(ft_importance_values, index = x_train.columns)
ft_top20 = ft_series.sort_values(ascending=False)[:20]

# 시각화
plt.figure(figsize=(8,6))
plt.title('Feature Importance Top 20')
sns.barplot(x=ft_top20, y=ft_top20.index)
plt.show()


# In[ ]:




