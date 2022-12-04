#!/usr/bin/env python
# coding: utf-8

# ## Settings

# In[34]:


# import libraries
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from statsmodels.stats.outliers_influence import variance_inflation_factor
import feature_engineering as fe
import pandas as pd
import numpy as np


# In[35]:


# load data
train_data = pd.read_csv("../data/train_data.csv")


# In[36]:


# preprocess data
pd.set_option('display.max_columns', None)
original_data = fe.feature_engineering(train_data)
original_data


# In[37]:


# get continuous data
continuous_data = original_data.drop(['userID', 'assessmentItemID', 'testId', 'answerCode', 'Timestamp', 'KnowledgeTag', 'year', 'day', 'minute','second', 'first3', 'mid3', 'last3', 'hour_answerCode_Level'], axis=1)


# * 범주형 변수와 유의미한 영향을 주지 못할 것으로 판단되는 변수는 제외
# * 연속형 변수만을 사용하여 평가함
# * 범주형 변수: userID, assessmentID, testId, answerCode, KnowledgeTag, first3, hour_answerCode_Level
# * 무의미한 변수: Timestamp, year, day, minute, second, mid3, last3
# * 변수의 유의미성에 대한 평가는 매우 주관적임

# ## 1. 분산 임계 (Variance Threshold)
# * 분산값이 작은(변별력이 낮은) 변수를 제외하는 작업

# In[38]:


# set features: input 52 continuous features
X = continuous_data.copy()

selector = VarianceThreshold(threshold=2)
selector.fit_transform(X).shape


# * 분산 2 기준,
# * 52개의 연속형 변수 중, 28개의 feature가 선택되었고, 24개의 feature가 제외되었다

# In[39]:


# 선택된 feature의 정보
X.columns[selector.get_support()]


# In[40]:


selector.variances_[selector.get_support()]


# ## 2. 카이제곱 독립검정 (Chi-squared Test)
# * output(answerCode)과의 상관관계가 높은 input feature들을 찾는 작업

# In[41]:


# set features: input 28 features
X = continuous_data.copy()
X = X[['userID_answerCode_count', 'userID_answerCode_sum',
       'testId_answerCode_count', 'testId_answerCode_sum',
       'assessmentItemID_answerCode_count', 'assessmentItemID_answerCode_sum',
       'KnowledgeTag_answerCode_count', 'KnowledgeTag_answerCode_sum', 'month',
       'hour', 'dayofweek', 'dayofweek_answerCode_count',
       'dayofweek_answerCode_sum', 'userID_first3_answerCode_count',
       'userID_first3_answerCode_sum', 'hour_answerCode_count',
       'hour_answerCode_sum', 'month_answerCode_count', 'month_answerCode_sum',
       'elapsedTime', 'userID_elapsedTime_median',
       'KnowledgeTag_elapsedTime_median',
       'assessmentItemID_elapsedTime_median', 'testId_elapsedTime_median',
       'userID_answerCode_elapsedTime_median',
       'KnowledgeTag_answerCode_elapsedTime_median',
       'assessmentItemID_answerCode_elapsedTime_median',
       'testId_answerCode_elapsedTime_median']]
X


# In[42]:


# scaling data in 0 to 1
for col in X.columns:
    X[col] = X[col] / X[col].max()
X


# In[43]:


y = original_data.copy()['answerCode']
y


# In[44]:


selector = SelectKBest(chi2, k=19) # 선택하고자 하는 변수의 개수를 k 값으로 지정해줄 수 있음
selector.fit_transform(X, y).shape


# In[45]:


selector.scores_


# In[46]:


# 선택된 feature의 정보
X.columns[selector.get_support()]


# ## 결론
# * 분산 2를 기준으로 변별력이 낮은 변수는 제외
# * 카이제곱 독립성 검정을 통해, 상관관계가 높은 변수들을 선별

# * 선택된 변수
# * 범주형: 'userID', 'assessmentID', 'testId', 'KnowledgeTag', 'first3', 'hour_answerCode_Level' / 'answerCode' (target)
# * 연속형: 'userID_answerCode_count', 'userID_answerCode_sum',
#        'testId_answerCode_count', 'testId_answerCode_sum',
#        'assessmentItemID_answerCode_sum', 'KnowledgeTag_answerCode_count',
#        'KnowledgeTag_answerCode_sum', 'userID_first3_answerCode_count',
#        'userID_first3_answerCode_sum', 'month_answerCode_count',
#        'month_answerCode_sum', 'elapsedTime', 'userID_elapsedTime_median',
#        'KnowledgeTag_elapsedTime_median', 'testId_elapsedTime_median',
#        'userID_answerCode_elapsedTime_median',
#        'KnowledgeTag_answerCode_elapsedTime_median',
#        'assessmentItemID_answerCode_elapsedTime_median',
#        'testId_answerCode_elapsedTime_median'

# * 사용하고자 하는 연속형 변수의 수를 줄이고 싶다면, 카이제곱 독립성 검정에 사용하는 k 값을 낮춰주면 됩니다

# In[ ]:




