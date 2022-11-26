#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import pandas as pd
import numpy as np
import warnings
import random
import os
warnings.filterwarnings(action="ignore")

from catboost import CatBoostRegressor, CatBoostClassifier, Pool

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing

from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold

# output csv에 시간 지정해주기 위함
from datetime import datetime

import sys
sys.path.append("../")
from feature_engineering import *

# 시간이 오래 걸리는 부분들에 대한 연산 시간 구하기 위함
import time

# 태훈님의 dataloader.py import해서 preprocessing
from dataloader import *


# In[2]:


def to_categories(df):
    # 카테고리형 feature
# 여기에 범주형 feature들 이름을 추가해주세요!
    categories = [
                "userID",
                "KnowledgeTag",
                "first3", 
                "year",
                "month", 
                "day",
                "hour",
                "minute",
                "second",
                "timeConcentrationCount",
                "timeConcentrationLevel",
                "monthSolvedCount"
                ]
    # 카테고리형 feature들에 label encoding 수행하는 작업
    le = preprocessing.LabelEncoder()
    for category in categories:
        if category in df.columns and df[category].dtypes != "int":
            df[category] = le.fit_transform(df[category])
        df[category] = df[category].astype("category")
    return df


# In[3]:


# 데이터 준비
train_data, test_data = load_data(IS_CUSTOM=False)
data = get_entire_data(train_data, test_data)
data = get_features(data)
data = to_categories(data)


# In[4]:


train, valid, test = split_train_valid_test_categorical(data)


# In[5]:


FEATS = [
        "userID",
        # "answerCode",
        # "KnowledgeTag",
        "userAnswerRate",
        "userSolvedLen",
        "testAnswerRate",
        "testSolvedLen",
        "testAnswerSum",
        "tagAnswerRate",
        "tagSolvedLen",
        "tagAnswerSum",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "first3",
        "timeConcentrationRate",
        "timeConcentrationCount",
        "timeConcentrationLevel",
        "user_correct_answer",
        "user_total_answer",
        "user_acc",
        "monthAnswerRate",
        "monthSolvedCount",
        ]


# In[6]:


X_train = train.drop(["answerCode"], axis=1)
y_train = train["answerCode"]

X_valid = valid.drop(["answerCode"], axis=1)
y_valid = valid["answerCode"]


# In[7]:


X_valid


# In[8]:


n_est = 2000
seed = 42
n_fold = 10

skfold = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

folds=[]
for train_idx, valid_idx in skfold.split(X_train, y_train):
    folds.append((train_idx, valid_idx))


# In[9]:


cat_models = []

cat_features = X_train[FEATS].columns[X_train[FEATS].dtypes == "category"].to_list()
cat_features


# In[10]:


for fold in range(n_fold):
    print(f"\n----------------- Fold {fold} -----------------\n")
    start_time = time.time()

    # CatBoostRegressor 사용
    params = {
        "iterations": 150,
        "learning_rate": 0.1,  # 0.1
        "eval_metric": "AUC",
        "random_seed": 42,
        # "logging_level": "Verbose", # 매 epoch마다 로그를 찍고 싶으면 "logging_level": "Verbose"로 변경
        "early_stopping_rounds": 100,
        "task_type": "GPU",
        "depth":12,
        "verbose":100
    }
    
    model = CatBoostRegressor(
        **params,
        cat_features=cat_features,
        allow_writing_files=False,
    )
    
    train_idx, valid_idx = folds[fold]
    X_train_fold, X_valid_fold, y_train_fold, y_valid_fold = X_train[FEATS].iloc[train_idx], X_train[FEATS].iloc[valid_idx], y_train.iloc[train_idx], y_train.iloc[valid_idx]

    train_data = Pool(data=X_train_fold, label=y_train_fold, cat_features=cat_features)
    valid_data = Pool(data=X_valid_fold, label=y_valid_fold, cat_features=cat_features)

    model.fit(
        train_data,
        eval_set=valid_data,
        # plot=True, # plot 찍고 싶으면 주석 제거
        use_best_model=True,
    )

    # 여기서 AUC 계산에 사용되는 X_valid는 X_valid_fold랑 다름. 
    # 최종 성능 평가는 더 위에서 생성한 X_valid라는 별도의 고정된 데이터로 한다고 생각하면 됩니다.
    cat_models.append(model)
    preds = model.predict(X_valid[FEATS])
    acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_valid, preds)

    print(f"VALID AUC : {auc} ACC : {acc}\n")

    print(f"elapsed: {time.time() - start_time: .3f}")


# In[11]:


# K-fold 없이 진행하고 싶으면
normal = False
# 일반 성능: VALID AUC : 0.7499030745484736 ACC : 0.6841397849462365
if normal:
    start_time = time.time()

    # CatBoostRegressor 사용
    params = {
        "iterations": 150,
        "learning_rate": 0.1,  # 0.1
        "eval_metric": "AUC",
        "random_seed": 42,
        "logging_level": "Silent", # 매 epoch마다 로그를 찍고 싶으면 "logging_level": "Verbose"로 변경
        "early_stopping_rounds": 100,
        "task_type": "GPU",
        "depth":12
    }

    model = CatBoostRegressor(
        **params,
        cat_features=cat_features,
        allow_writing_files=False,
    )

    model.fit(
        train[FEATS],
        y_train,
        eval_set=[(valid[FEATS], y_valid)],
        # plot=True, # plot 찍고 싶으면 주석 제거
    )

    preds = model.predict(valid[FEATS])
    acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_valid, preds)

    print(f"VALID AUC : {auc} ACC : {acc}\n")

    print(f"elapsed: {time.time() - start_time: .3f}")


# In[12]:


# valid AUC, valid ACC, fold AUC 모두 고려해서 best를 저장해주세요
best_fold_idx = 1
model = cat_models[best_fold_idx]


# In[13]:


start_time = time.time()

result = permutation_importance(model, X_valid[FEATS], y_valid, scoring = "roc_auc", n_repeats=30, random_state=42)
sorted_result = result.importances_mean.argsort()
fig = plt.figure(figsize=(12, 6))
plt.barh(range(len(FEATS)), result.importances_mean[sorted_result], align="center")
plt.yticks(range(len(FEATS)), np.array(FEATS)[sorted_result])
plt.title("permutation_importance")

print(f"elapsed: {time.time() - start_time: .3f} sec")


# In[14]:


start_time = time.time()

feature_importance = model.feature_importances_
sorted_idx = np.argsort(feature_importance)
fig = plt.figure(figsize=(12, 6))
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align="center")
plt.yticks(range(len(sorted_idx)), np.array(FEATS)[sorted_idx])
plt.title("Feature Importance")

print(f"elapsed: {time.time() - start_time: .3f} sec")


# In[15]:


start_time = time.time()

X_test = test.drop(["answerCode"], axis=1)
total_preds = model.predict(X_test[FEATS])

print(f"elapsed: {time.time() - start_time: .3f} sec")


# In[16]:


# SAVE OUTPUT
output_dir = "/opt/ml/input/CatBoost_output"
write_path = os.path.join(output_dir, f"CatBoost_submission_{datetime.now().microsecond}.csv")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(write_path, "w", encoding="utf8") as w:
    print("writing prediction : {}".format(write_path))
    w.write("id,prediction\n")
    for id, p in enumerate(total_preds):
        w.write("{},{}\n".format(id, p))

