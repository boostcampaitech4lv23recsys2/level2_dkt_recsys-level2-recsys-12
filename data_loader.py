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

################################# XGBoost #################################


def xgb_preprocessing(data):
    """
    input : dataframe
    output : dataframe
    preprocessing numerical data
    """
    for col in data.columns:
        if col.endswith("Rate") or col.endswith("Count") or col.endswith("Len"):
            _min, _max = data[col].min(), data[col].max()
            data[col]=(data[col]-_min)/(_max-_min)
        if data[col].dtype == object:
            data[col]=data[col].astype(int)
    return data


def xgb_data_loader(IS_CUSTOM=False,USE_VALID=True, DROPS=[]):
    """
    Load and preprocess data to use xgboost

    input params : IS_CUSTOM, USE_VALID, DROPS
    output : x_train, x_valid, y_train, y_valid, test

    if USE_VALID=False, x_valid and y_valid is empty dataframe
    """
    _train, _test = load_data(IS_CUSTOM=IS_CUSTOM)
    entire_data = get_entire_data(_train, _test)
    df = get_features(entire_data).drop(DROPS, axis=1)
    train, valid, test = split_train_valid_test_categorical(df)
    if not USE_VALID:
        train = pd.concat([train,valid])
        valid = valid.drop([val for val in valid.index], axis=0)
    x_train = train.drop(["answerCode"], axis=1)
    y_train = train["answerCode"]
    x_valid = valid.drop(["answerCode"], axis=1)
    y_valid = valid["answerCode"]
    x_train = xgb_preprocessing(x_train)
    x_valid = xgb_preprocessing(x_valid)
    test = xgb_preprocessing(test)
    return x_train, x_valid, y_train, y_valid, test

    
################################# CatBoost #################################

def ctb_preprocessing(data):
    """
    input : dataframe
    output : dataframe
    preprocessing numerical data
    """
    for col in data.columns:
        if data[col].dtype == int:
            print(col)
            continue
        data[col] = data[col].fillna(-1).astype(str)
    return data

def ctb_data_loader(IS_CUSTOM=False,USE_VALID=True, DROPS=[]):
    """
    Load and preprocess data to use xgboost

    input params : IS_CUSTOM, USE_VALID, DROPS
    output : x_train, x_valid, y_train, y_valid, test

    if USE_VALID=False, x_valid and y_valid is empty dataframe
    """
    _train, _test = load_data(IS_CUSTOM=IS_CUSTOM)
    entire_data = get_entire_data(_train, _test)
    df = get_features(entire_data).drop(DROPS, axis=1)
    train, valid, test = split_train_valid_test_categorical(df)
    if not USE_VALID:
        train = pd.concat([train,valid])
        valid = valid.drop([val for val in valid.index], axis=0)
    x_train = train.drop(["answerCode"], axis=1)
    y_train = train["answerCode"]
    x_valid = valid.drop(["answerCode"], axis=1)
    y_valid = valid["answerCode"]
    x_train = ctb_preprocessing(x_train)
    x_valid = ctb_preprocessing(x_valid)
    test = ctb_preprocessing(test)
    return x_train, x_valid, y_train, y_valid, test



