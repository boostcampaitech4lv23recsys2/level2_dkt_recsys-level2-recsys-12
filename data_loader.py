"""
load_data
get_entire_data
get_features
split_train_valid_test_categorical
"""
import sys
sys.path.append("/opt/ml/input/code")

import feature_engineering as fe

from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from collections import Counter

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


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


def split_train_valid_test_categorical(df, valid_len=3):
    """
    카테고리형 모델에 적용할 수 있게 train, test, valid를 분리합니다.
    input df : 전체 데이터셋
    """
    idx = (df["answerCode"]==-1).values
    test = df[idx]
    val_idx = df["answerCode"].isna().values
    for i in range(valid_len):
        idx = np.append(idx, False)[1:]
        val_idx = val_idx | idx
    valid = df[val_idx]
    train = df[~(val_idx|(df["answerCode"]==-1))]
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


def xgb_data_loader(IS_CUSTOM=False,USE_VALID=True, DROPS=[], valid_len=3):
    """
    Load and preprocess data to use xgboost

    input params : IS_CUSTOM, USE_VALID, DROPS
    output : x_train, x_valid, y_train, y_valid, test

    if USE_VALID=False, x_valid and y_valid is empty dataframe
    """
    _train, _test = load_data(IS_CUSTOM=IS_CUSTOM)
    entire_data = get_entire_data(_train, _test)
    df = get_features(entire_data).drop(DROPS, axis=1)
    train, valid, test = split_train_valid_test_categorical(df,valid_len=valid_len)
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


def get_pca_data(ss_data, n_components = 2):
    pca = PCA(n_components = n_components)
    pca.fit(ss_data)
    return pca.transform(ss_data), pca

def get_pd_from_pca(pca_data, col_num):
    cols = ["pca_"+str(n) for n in range(col_num)]
    return pd.DataFrame(pca_data, columns=cols)

def print_variance_ratio(pca, only_sum = False):
    if only_sum == False:
        print('variance_ratio :', pca.explained_variance_ratio_)
    print('sum of variance_ratio: ', np.sum(pca.explained_variance_ratio_))


def xgb_PCA_data_loader(IS_CUSTOM=False,USE_VALID=True, DROPS=[], n_components=5, valid_len=3):
    """
    Load and preprocess data to use xgboost

    input params : IS_CUSTOM, USE_VALID, DROPS
    output : x_train, x_valid, y_train, y_valid, test

    if USE_VALID=False, x_valid and y_valid is empty dataframe
    """
    while "answerCode" in DROPS:
        print("answerCode는 DROP하지 마세요. 추가 후 진행합니다.")
        DROPS.remove("answerCode")
    print("Load data..........................................")
    _train, _test = load_data(IS_CUSTOM=IS_CUSTOM)
    entire_data = get_entire_data(_train, _test)
    df = get_features(entire_data).drop(DROPS, axis=1).dropna()


    print("Split data..........................................")

    train, valid, test = split_train_valid_test_categorical(df,valid_len=valid_len)
    if not USE_VALID:
        train = pd.concat([train,valid])
        valid = valid.drop([val for val in valid.index], axis=0)

    x_train = train.drop(["answerCode"], axis=1)
    y_train = train["answerCode"]
    x_valid = valid.drop(["answerCode"], axis=1)
    y_valid = valid["answerCode"]
    ans = test["answerCode"].values
    test = test.drop(["answerCode"], axis=1)
    
    print("Standard Scaling....................................")
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_valid = scaler.transform(x_valid)
    test = scaler.transform(test)
    
    print("Find PCA from data..........................................")
    pca_data, pca = get_pca_data(x_train, n_components=n_components)

    pca_x_train = get_pd_from_pca(pca_data, n_components)
    pca_x_valid = get_pd_from_pca(pca.fit_transform(x_valid), n_components)
    pca_test = get_pd_from_pca(pca.fit_transform(test), n_components)
    pca_test["answerCode"]=ans

    return pca_x_train, pca_x_valid, y_train, y_valid, pca_test
    
################################# CatBoost #################################

def ctb_preprocessing(data):
    """
    input : dataframe
    output : dataframe
    preprocessing numerical data
    """
    for col in data.columns:
        if data[col].dtype == int:
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


################################# LGBM #################################
def lgbm_data_loader(IS_CUSTOM=False,USE_VALID=True, DROPS=[], valid_len=3):
    _train, _test = load_data(IS_CUSTOM=IS_CUSTOM)
    _df = get_entire_data(_train, _test)
    df = get_features(_df).drop(DROPS, axis=1)
    for col in df.columns:
        if df[col].dtype=="object":
            df[col]=df[col].astype(float)
    train, valid, test = split_train_valid_test_categorical(df, valid_len=5)
    if not USE_VALID:
        train = pd.concat([train,valid])
        valid = valid.drop([val for val in valid.index], axis=0)
    y_train = train["answerCode"]
    train = train.drop(["answerCode"], axis=1)
    y_valid = valid["answerCode"]
    valid = valid.drop(["answerCode"], axis=1)
    return train, valid, y_train, y_valid, test

################################# Tabnet #################################
def show_process(func):
    def wrapFunc(*args, **kargs):
        print("Start", func.__name__)
        func(*args, **kargs)
        print("End", func.__name__)
    return wrapFunc
    
class DataLoader:
    def __init__(self, path="../data", IS_CUSTOM=True):
        self.load_data(path=path, IS_CUSTOM=IS_CUSTOM)
        self.entire_df = pd.concat([self.raw_train, self.raw_test]).drop_duplicates().sort_values(["userID","Timestamp"])
        self.preprocessing(self.entire_df)
        self.train_test_split(self.preprocessed_df)
    @show_process    
    def load_data(self, path="../data", IS_CUSTOM=True):
        self.raw_train = pd.read_csv(path+"/train_data.csv")
        self.raw_test = pd.read_csv(path+"/test_data.csv") if IS_CUSTOM else pd.read_csv(path+"/custom_test_data.csv")
    @show_process
    def train_test_split(self, data):
        self.train_df = data[data["answerCode"] != -1]
        self.test_df = data[data["answerCode"] == -1]
    @show_process
    def preprocessing(self, data):
        self.preprocessed_df = fe.feature_engineering(data)

class TabnetDataLoader(DataLoader):
    def __init__(self, IS_CUSTOM=True, test_size=0.2, USE_VALID=True, DROPS=[], path="../data"):
        super().__init__(IS_CUSTOM=True, path=path)
        self.test_size = test_size
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None
        self.first3_knowledgeTag_clustering()

        if not USE_VALID:
            self.test_size=-1

        self.train_df.drop(DROPS, axis=1, inplace=True)
        self.test_df.drop(DROPS, axis=1, inplace=True)
        self.X_test = self.test_df.drop("answerCode")
        self.y_test = self.test_df.answerCode if IS_CUSTOM else None

        self.train_valid_split(self.test_size)

    @show_process
    def first3_knowledgeTag_clustering(self):
        cluster = KMeans(n_clusters=44)
        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(self.train_df[["KnowledgeTag","first3"]])
        minmax_scaled_train = minmax_scaler.transform(self.train_df[["KnowledgeTag","first3"]])
        minmax_scaled_test = minmax_scaler.transform(self.test_df[["KnowledgeTag","first3"]])
        cluster.fit(minmax_scaled_train)
        self.train_df["tag_first3_cluster"] = cluster.predict(minmax_scaled_train)
        self.test_df["tag_first3_cluster"] = cluster.predict(minmax_scaled_test)
        self.labeling()

    # @show_process
    def binning(self, col, n_bins):
        binner = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="kmeans")
        binner.fit(self.train_df[col].values.reshape(-1,1))
        self.train_df[col] = binner.transform(self.train_df[col].values.reshape(-1,1)).astype(int)
        self.test_df[col] = binner.transform(self.test_df[col].values.reshape(-1,1)).astype(int)

    # @show_process
    def label_encoding(self,col):
        encoder = LabelEncoder()
        encoder.fit(pd.concat([self.train_df[col], self.test_df[col]]))
        self.train_df[col] = encoder.transform(self.train_df[col].copy())
        self.test_df[col] = encoder.transform(self.test_df[col].copy())

    @show_process
    def labeling(self):
        for col in self.train_df.columns:
            if col.split("_")[-1] in ("mean", "count", "var", "median"):
                n_bin = self.train_df[col].nunique()//50
                if n_bin > 3:
                    self.binning(col, n_bin)
                else:
                    self.label_encoding(col)
            elif col == "elapsedTime":
                self.binning(col, 10)
            elif col in ["assessmentItemID", "testId"]:
                self.label_encoding(col)
                
    @show_process
    def train_valid_split(self,test_size):
        if test_size <= 0:
            self.X_train = self.train_df.drop("answerCode",axis=1)
            self.y_train = self.train_df.answerCode
            return
        train_idx = np.array([])
        offset = 0
        for key, nunique in Counter(self.train_df.userID).items():
            data = np.arange(nunique).reshape(-1,1) + offset
            tidx, _, _, _ = train_test_split(data,data,test_size=test_size, random_state=42)
            train_idx = np.append(train_idx, tidx)
            offset += nunique
        idx = np.array([False]*len(self.train_df))
        idx[train_idx.astype(int)]=True

        self.X_train = self.train_df[idx].drop("answerCode",axis=1)
        self.y_train = self.train_df[idx].answerCode
        self.X_valid = self.train_df[~idx].drop("answerCode",axis=1)
        self.y_valid = self.train_df[~idx].answerCode
        print(f"X_train:{self.X_train.shape}\ny_train:{self.y_train.shape}\nX_valid:{self.X_valid.shape}\ny_valid:{self.y_valid.shape}")