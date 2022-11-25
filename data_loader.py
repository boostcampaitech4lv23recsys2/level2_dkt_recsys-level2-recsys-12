"""
load_data
get_entire_data
get_features
split_train_valid_test_categorical
"""
import sys
sys.path.append("/opt/ml/input/code")

import feature_engineering as fe
import pandas as pd
import numpy as np


def load_data(path="/opt/ml/input/data", IS_CUSTOM=False):
    """path : 데이터가 존재하는 파일의 경로를 넣어주세요."""
    test_name = "/custom_test_data.csv" if IS_CUSTOM else "/test_data.csv"
    train, test = pd.read_csv(path+"/train_data.csv"), pd.read_csv(path+test_name)
    return train, test


def get_entire_data(data1, data2):
    """data1,data2 : train, test data를 넣어주세요."""
    df = pd.concat([data1, data2])
    data = df.sort_values(["userID", "Timestamp"])
    data.drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last", inplace=True
    )
    return data

def get_features(data):
    """data : feature_engineering을 진행 할 데이터셋을 넣어주세요."""
    return fe.feature_engineering(data)


def split_train_valid_test_categorical(df):
    """
    카테고리형 모델에 적용할 수 있게 train, test, valid를 분리합니다.
    input df : 전체 데이터셋
    """
    idx = df["answerCode"]==-1
    idx_ndarr = idx.values
    new = np.append(idx_ndarr,[False])[1:]
    valid = df[new]
    data = df.drop([idx for idx, i in enumerate(new) if i])
    train = data[data["answerCode"]!=-1]
    test = data[data["answerCode"]==-1]
    return train, valid, test

    