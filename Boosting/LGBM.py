#!/usr/bin/env python
# coding: utf-8

# # LGBM을 활용한 베이스라인

# In[140]:


import pandas as pd
import os
import random
import optuna
from optuna import Trial
from optuna.samplers import TPESampler


# In[141]:


import warnings 
warnings.filterwarnings("ignore")


# In[142]:


# !pip install optuna


# ## 1. 데이터 로딩

# In[143]:


data_dir = '/opt/ml/input/data/' # 경로는 상황에 맞춰서 수정해주세요!
csv_file_path = os.path.join(data_dir, 'train_data.csv') # 데이터는 대회홈페이지에서 받아주세요 :)
df = pd.read_csv(csv_file_path)


# ## 2. Feature Engineering

# In[144]:


def feature_engineering(df):
    
    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=['userID','Timestamp'], inplace=True)
    
    #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    df['user_acc'] = df['user_correct_answer']/df['user_total_answer']

    # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
    # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
    correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
    correct_t.columns = ["test_mean", 'test_sum']
    correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
    correct_k.columns = ["tag_mean", 'tag_sum']

    df = pd.merge(df, correct_t, on=['testId'], how="left")
    df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
    
    return df


# In[145]:


df = feature_engineering(df)
df.head()


# ## 3. Train/Test 데이터 셋 분리

# In[146]:


# train과 test 데이터셋은 사용자 별로 묶어서 분리를 해주어야함
random.seed(42)
def custom_train_test_split(df, ratio=0.7, split=True):
    
    users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
    random.shuffle(users)
    
    max_train_data_len = ratio*len(df)
    sum_of_train_data = 0
    user_ids =[]

    for user_id, count in users:
        sum_of_train_data += count
        if max_train_data_len < sum_of_train_data:
            break
        user_ids.append(user_id)


    train = df[df['userID'].isin(user_ids)]
    test = df[df['userID'].isin(user_ids) == False]

    #test데이터셋은 각 유저의 마지막 interaction만 추출
    test = test[test['userID'] != test['userID'].shift(-1)]
    return train, test


# In[147]:


# 유저별 분리
train, test = custom_train_test_split(df)

# 사용할 Feature 설정
FEATS = ['KnowledgeTag', 'user_correct_answer', 'user_total_answer', 
         'user_acc', 'test_mean', 'test_sum', 'tag_mean','tag_sum']

# X, y 값 분리
y_train = train['answerCode']
train = train.drop(['answerCode'], axis=1)

y_test = test['answerCode']
test = test.drop(['answerCode'], axis=1)


# In[148]:


# !pip install lightgbm


# In[149]:


import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np


# In[150]:


lgb_train = lgb.Dataset(train[FEATS], y_train)
lgb_test = lgb.Dataset(test[FEATS], y_test)


# ## 하이퍼파라미터 튜닝

# ## 4. 훈련 및 검증 (+ 하이퍼파라미터 튜닝(optuna))

# In[152]:


def objective(trial: Trial):
    params = {
        'objective': 'binary',
        'bagging_fraction': trial.suggest_float("bagging_fraction", 0.5, 0.6, step=0.01),
        'bagging_seed': 11, ##
        'learning_rate': trial.suggest_categorical("lr", [0.001, 0.005, 0.01, 0.05, 0.1]),
        'num_iterations': trial.suggest_int("n_iter", 100, 1000, 100),
        'max_depth': trial.suggest_categorical('max_depth', [-1, 1]), # need to consider
        'boosting': 'gbdt',
        'early_stopping': trial.suggest_categorical('patience', [0, 5, 10, 15, 20])

    }
    model = lgb.train(
        params, 
        lgb_train,
        valid_sets=[lgb_train, lgb_test],
        verbose_eval=100,
        num_boost_round=500,
        early_stopping_rounds=100
    )

    preds = model.predict(test[FEATS])
    acc = accuracy_score(y_test, np.where(preds >= 0.5, 1, 0))
    auc = roc_auc_score(y_test, preds)

    return auc

sampler = TPESampler(seed=42)
study = optuna.create_study(
    study_name="lgbm_parameter_opt",
    direction="maximize",
    sampler=sampler,
)
study.optimize(objective, n_trials=10)
print("Best Score:", study.best_value)
print("Best trial:", study.best_trial.params)


# In[153]:


# INSTALL MATPLOTLIB IN ADVANCE
# _ = lgb.plot_importance(model)


# ## 5. Inference

# In[154]:


# LOAD TESTDATA
test_csv_file_path = os.path.join(data_dir, 'test_data.csv')
test_df = pd.read_csv(test_csv_file_path)

# FEATURE ENGINEERING
test_df = feature_engineering(test_df)

# LEAVE LAST INTERACTION ONLY
test_df = test_df[test_df['userID'] != test_df['userID'].shift(-1)]

# DROP ANSWERCODE
test_df = test_df.drop(['answerCode'], axis=1)


# In[155]:


# MAKE PREDICTION
total_preds = model.predict(test_df[FEATS])


# In[156]:


# SAVE OUTPUT
output_dir = 'output/'
write_path = os.path.join(output_dir, "submission.csv")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(write_path, 'w', encoding='utf8') as w:
    print("writing prediction : {}".format(write_path))
    w.write("id,prediction\n")
    for id, p in enumerate(total_preds):
        w.write('{},{}\n'.format(id,p))


# ###**콘텐츠 라이선스**
# 
# <font color='red'><b>**WARNING**</b></font> : **본 교육 콘텐츠의 지식재산권은 재단법인 네이버커넥트에 귀속됩니다. 본 콘텐츠를 어떠한 경로로든 외부로 유출 및 수정하는 행위를 엄격히 금합니다.** 다만, 비영리적 교육 및 연구활동에 한정되어 사용할 수 있으나 재단의 허락을 받아야 합니다. 이를 위반하는 경우, 관련 법률에 따라 책임을 질 수 있습니다.
# 
# 
