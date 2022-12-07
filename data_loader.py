"""
load_data
get_entire_data
get_features
split_train_valid_test_categorical
"""
import sys
sys.path.append("/opt/ml/input/code")

import feature_engineering as fe

from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from tqdm import tqdm
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
    def __init__(self, IS_CUSTOM=True, test_size=0.2, USE_VALID=True, DROPS=[], path="../data", binning=False, pca=True, n_components=30):
        super().__init__(IS_CUSTOM=True, path=path)
        self.n_components = 30
        
        self.test_size = test_size
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None
        self.other_features = [
            "answerCode",
            "Timestamp",
            
        ]
        self.cat_features = [
            "userID",
            "assessmentItemID",
            "testId",
            "KnowledgeTag",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "dayofweek",
            "first3",
            "mid3",
            "last3",
            "hour_answerCode_Level",
            
        ]
        self.cont_features = [
            "userID_answerCode_mean",
            "userID_answerCode_count",
            "userID_answerCode_sum",
            "userID_answerCode_var",
            "userID_answerCode_median",
            "testId_answerCode_mean",
            "testId_answerCode_count",
            "testId_answerCode_sum",
            "testId_answerCode_var",
            "testId_answerCode_median",
            "assessmentItemID_answerCode_mean",
            "assessmentItemID_answerCode_count",
            "assessmentItemID_answerCode_sum",
            "assessmentItemID_answerCode_var",
            "assessmentItemID_answerCode_median",
            "KnowledgeTag_answerCode_mean",
            "KnowledgeTag_answerCode_count",
            "KnowledgeTag_answerCode_sum",
            "KnowledgeTag_answerCode_var",
            "KnowledgeTag_answerCode_median",
            "dayofweek_answerCode_mean",
            "dayofweek_answerCode_count",
            "dayofweek_answerCode_sum",
            "dayofweek_answerCode_var",
            "dayofweek_answerCode_median",
            "userID_first3_answerCode_mean",
            "userID_first3_answerCode_count",
            "userID_first3_answerCode_sum",
            "userID_first3_answerCode_var",
            "userID_first3_answerCode_median",
            "hour_answerCode_mean",
            "hour_answerCode_count",
            "hour_answerCode_sum",
            "hour_answerCode_var",
            "hour_answerCode_median",
            "month_answerCode_mean",
            "month_answerCode_count",
            "month_answerCode_sum",
            "month_answerCode_var",
            "month_answerCode_median",
            "user_acc",
            "assessmentItemID_elo_pred",
            "testId_elo_pred",
            "KnowledgeTag_elo_pred",
            "feature_ensemble_elo_pred",
            "userID_elapsedTime_median",
            "KnowledgeTag_elapsedTime_median",
            "assessmentItemID_elapsedTime_median",
            "testId_elapsedTime_median",
            "userID_answerCode_elapsedTime_median",
            "KnowledgeTag_answerCode_elapsedTime_median",
            "assessmentItemID_answerCode_elapsedTime_median",
            "elapsedTime",
            "testId_answerCode_elapsedTime_median",
            "user_correct_answer",
            "user_total_answer",
        ]
        self.important_cont_features = [
            'assessmentItemID_elo_pred',
            'testId_elo_pred',
            'KnowledgeTag_elo_pred',
            'feature_ensemble_elo_pred',
        ]
        self.first3_knowledgeTag_clustering()

        if not USE_VALID:
            self.test_size=-1

        self.train_df.drop(DROPS, axis=1, inplace=True)
        self.test_df.drop(DROPS, axis=1, inplace=True)
        self.X_test = self.test_df.drop("answerCode",axis=1)
        self.y_test = self.test_df.answerCode if IS_CUSTOM else None

        if pca:
            self.pca_and_labeling()
        elif binning:
            self.labeling()

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
                n_bin = self.train_df[col].nunique()//20
                if n_bin > 4:
                    self.binning(col, n_bin)
                else:
                    self.label_encoding(col)
            elif col == "elapsedTime":
                self.binning(col, 10)
            elif col in ["assessmentItemID", "testId"]:
                self.label_encoding(col)
            else:
                continue
                
    @show_process
    def train_valid_split(self,test_size):
        if test_size <= 0:
            self.X_train = self.train_df.drop("answerCode",axis=1)
            self.y_train = self.train_df.answerCode
            return
        train_idx = np.array([])
        offset = 0
        for key, nunique in tqdm(Counter(self.train_df.userID).items(),"split..."):
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
    
    @show_process
    def pca_and_labeling(self):
        cont = self.train_df[self.cont_features]
        cat = self.train_df[self.cat_features]
        self.cat_train_df = pd.DataFrame(columns=self.cat_features)
        self.cat_test_df = pd.DataFrame(columns=self.cat_features)
        
        ### Label encoding ###
        for col in self.cat_features:
            label_encoder = LabelEncoder()
            label_encoder.fit(cat[col])
            self.cat_train_df[col] = label_encoder.transform(self.train_df[col])
            self.cat_test_df[col] = label_encoder.transform(self.test_df[col])
    
        ### Scaling ###
        scaler = StandardScaler()
        scaler.fit(self.train_df[self.cont_features])
        train_cont = pd.DataFrame(scaler.transform(self.train_df[self.cont_features]), columns=self.cont_features)
        test_cont = pd.DataFrame(scaler.transform(self.test_df[self.cont_features]), columns=self.cont_features)
        train_cont = train_cont.fillna(train_cont.mean())
        test_cont = test_cont.fillna(test_cont.mean())
        self.pca_train_data, pca_func = get_pca_data(train_cont, n_components=self.n_components)        
        self.pca_test_data = pca_func.transform(test_cont)
        print_variance_ratio(pca_func)

        ### Important Features -> Scaling ###
        imp_scaler = StandardScaler()
        imp_scaler.fit(self.train_df[self.important_cont_features])
        self.important_train_cont = pd.DataFrame(imp_scaler.transform(self.train_df[self.important_cont_features]), columns=self.important_cont_features)
        self.important_test_cont = pd.DataFrame(imp_scaler.transform(self.test_df[self.important_cont_features]), columns=self.important_cont_features)

        self.train_df = pd.concat([self.cat_train_df, get_pd_from_pca(self.pca_train_data,self.n_components), self.important_train_cont, self.train_df["answerCode"]], axis=1)
        self.test_df = pd.concat([self.cat_test_df, get_pd_from_pca(self.pca_test_data,self.n_components), self.important_test_cont, self.test_df["answerCode"]], axis=1)


class Preprocessed_data_loader:
    def __init__(self, path="../data",IS_CUSTOM=False):

        self.other_features = [
            "answerCode",
            "Timestamp",
        ]
        self.cat_features = [
            "userID",
            "assessmentItemID",
            "testId",
            "KnowledgeTag",
            "year",
            "month",
            "day",
            "hour",
            "minute",
            "second",
            "dayofweek",
            "first3",
            "mid3",
            "last3",
            "hour_answerCode_Level",
            
        ]
        self.cont_features = [
            "userID_answerCode_mean",
            "userID_answerCode_count",
            "userID_answerCode_sum",
            "userID_answerCode_var",
            "userID_answerCode_median",
            "testId_answerCode_mean",
            "testId_answerCode_count",
            "testId_answerCode_sum",
            "testId_answerCode_var",
            "testId_answerCode_median",
            "assessmentItemID_answerCode_mean",
            "assessmentItemID_answerCode_count",
            "assessmentItemID_answerCode_sum",
            "assessmentItemID_answerCode_var",
            "assessmentItemID_answerCode_median",
            "KnowledgeTag_answerCode_mean",
            "KnowledgeTag_answerCode_count",
            "KnowledgeTag_answerCode_sum",
            "KnowledgeTag_answerCode_var",
            "KnowledgeTag_answerCode_median",
            "dayofweek_answerCode_mean",
            "dayofweek_answerCode_count",
            "dayofweek_answerCode_sum",
            "dayofweek_answerCode_var",
            "dayofweek_answerCode_median",
            "userID_first3_answerCode_mean",
            "userID_first3_answerCode_count",
            "userID_first3_answerCode_sum",
            "userID_first3_answerCode_var",
            "userID_first3_answerCode_median",
            "hour_answerCode_mean",
            "hour_answerCode_count",
            "hour_answerCode_sum",
            "hour_answerCode_var",
            "hour_answerCode_median",
            "month_answerCode_mean",
            "month_answerCode_count",
            "month_answerCode_sum",
            "month_answerCode_var",
            "month_answerCode_median",
            "user_acc",
            "assessmentItemID_elo_pred",
            "testId_elo_pred",
            "KnowledgeTag_elo_pred",
            "feature_ensemble_elo_pred",
            "userID_elapsedTime_median",
            "KnowledgeTag_elapsedTime_median",
            "assessmentItemID_elapsedTime_median",
            "testId_elapsedTime_median",
            "userID_answerCode_elapsedTime_median",
            "KnowledgeTag_answerCode_elapsedTime_median",
            "assessmentItemID_answerCode_elapsedTime_median",
            "elapsedTime",
            "testId_answerCode_elapsedTime_median",
            "user_correct_answer",
            "user_total_answer",
        ]
        self.data_path = path
        train_name = "/preprocessed_custom_train_data.csv" if IS_CUSTOM else "/preprocessed_train_data.csv"
        test_name = "/preprocessed_custom_test_data.csv" if IS_CUSTOM else "/preprocessed_test_data.csv"
        self.train_df = pd.read_csv(path+train_name)
        self.test_df = pd.read_csv(path+test_name)









if __name__ == "__main__":
    data = Preprocessed_data_loader(path="../data", IS_CUSTOM=True)



