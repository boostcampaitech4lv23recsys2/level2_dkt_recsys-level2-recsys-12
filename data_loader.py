"""
load_data
get_entire_data
get_features
split_train_valid_test_categorical
"""
import pandas as pd
import numpy as np
import sys
sys.path.append("/opt/ml/input/code")

from feature_engineering import cluster_two_features, feature_engineering

from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, LabelEncoder
from sklearn.decomposition import PCA
from tqdm import tqdm


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
    return feature_engineering(data)


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


def get_low_importance_pca(df, cols, n_components):
    df[cols] = df[cols].fillna(df[cols].mean())
    print("Scaling init...")
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[cols])
    print("PCA init...")
    pca = PCA(n_components=n_components)
    df_pca = pd.DataFrame(pca.fit_transform(df_scaled))
    df_pca.columns = ["pca_"+str(1+i) for i in range(n_components)]
    print("done!")
    return df_pca


def ctb_data_loader(IS_CUSTOM=False,USE_VALID=True, DROPS=[], low_importance=[], n_components=10, valid_len=3):
    """
    Load and preprocess data to use xgboost

    input params : IS_CUSTOM, USE_VALID, DROPS
    output : x_train, x_valid, y_train, y_valid, test

    if USE_VALID=False, x_valid and y_valid is empty dataframe
    """
    _train, _test = load_data(IS_CUSTOM=IS_CUSTOM)
    entire_data = get_entire_data(_train, _test)
    df = get_features(entire_data) #.drop(DROPS, axis=1)
    df["KnowledgeTag_first3_clust"] = cluster_two_features(df, "KnowledgeTag", "first3")
    pca_df = get_low_importance_pca(df,low_importance, n_components)
    for i in range(n_components):
        df["pca_"+str(i+1)] = pca_df["pca_"+str(i+1)]
    df = df.drop(DROPS, axis=1)
    train, valid, test = split_train_valid_test_categorical(df, valid_len=valid_len)
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
"""

"""

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



################################# category #################################
def get_binning_data(df, col, n_bins=10, encode='ordinal', strategy='quantile'):
    """
    #1. strategy = 'uniform'
    #2. strategy = 'quantile'
    #3. strategy = 'kmeans'
    """
    train_pt = pd.DataFrame(df[col])
    est_uni = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy=strategy)
    est_uni.fit(train_pt)
    Xt_uni=est_uni.transform(train_pt)
    unique, counts = np.unique(Xt_uni, return_counts=True)
    return Xt_uni.squeeze(1)
    
def category_data_loader(IS_CUSTOM=False, USE_VALID=True, encode='ordinal', strategy='kmeans', DROPS=[], valid_len=3, PCA=[], n_components=3, USE_PCA = False):
    """
    #1. strategy = 'uniform'
    #2. strategy = 'quantile'
    #3. strategy = 'kmeans'
    """
    _train, _test = load_data(IS_CUSTOM=IS_CUSTOM)
    entire_data = get_entire_data(_train, _test)
    df = get_features(entire_data) #.drop(DROPS, axis=1)
    df["KnowledgeTag_first3_clust"] = cluster_two_features(df, "KnowledgeTag", "first3")
    if USE_PCA:
        pca_df = get_low_importance_pca(df,PCA, n_components)
        for i in range(n_components):
            df["pca_"+str(i+1)] = pca_df["pca_"+str(i+1)]
        
    cat_features = []
    bin_list = []
    new_df = df.copy()

    for col in tqdm(df.columns,"Label Encoding..."):
        if (col in DROPS + ["answerCode"] + PCA) or ("pca" in col):
            continue
        if (df[col].dtype == int) or (df[col].dtype == float and df[col].nunique() < 100):
            new_df[col] = new_df[col].fillna(new_df[col].mean())
            le = LabelEncoder()
            new_df[col] = le.fit_transform(new_df[col])
            cat_features.append(col)
        elif df[col].dtype==float:
            bin_list.append(col)
            new_df[col] = new_df[col].fillna(new_df[col].mean())
            cat_features.append(col)
            continue
        else:
            le = LabelEncoder()
            new_df[col] = le.fit_transform(new_df[col])
            cat_features.append(col)
    
    print("Start binning")
    for col in tqdm(bin_list,"Binning..."):
        new_df[col] = get_binning_data(new_df, col, n_bins=int(df[col].nunique()**0.5), strategy="kmeans")
    new_df = new_df.drop(DROPS+PCA, axis=1)

    for col in new_df.columns:
        if "pca" in col:
            continue
        new_df[col] = new_df[col].astype(int)
    train, valid, test = split_train_valid_test_categorical(new_df, valid_len=valid_len)
    if not USE_VALID:
        train = pd.concat([train,valid])
        valid = valid.drop([val for val in valid.index], axis=0)
    y_train = train["answerCode"]
    train = train.drop(["answerCode"], axis=1)
    y_valid = valid["answerCode"]
    valid = valid.drop(["answerCode"], axis=1)
    return train, valid, y_train, y_valid, test, cat_features









if __name__ == "__main__":
    IS_CUSTOM = True
    USE_VALID = True
    valid_len = 5
    n_components = 5

    DROPS = [
        'Timestamp','year','day','minute','second','KnowledgeTag',
    ]
    PCA = [
        'hour_answerCode_sum',
        'userID_dayofweek_answerCode_count',
        'user_correct_answer',
        'user_total_answer',
        'hour_answerCode_var',
        'hour_answerCode_mean',
        'userID_first3_answerCode_count',
        'userID_month_answerCode_count',
        'KnowledgeTag_first3_answerCode_sum',
        'KnowledgeTag',
        'userID_answerCode_count',
        'userID_answerCode_sum',
        'testId_answerCode_sum',
        'KnowledgeTag_answerCode_count',
        'KnowledgeTag_answerCode_sum',
        'month',
        'hour',
        'dayofweek',
        'dayofweek_answerCode_mean',
        'dayofweek_answerCode_count',
        'dayofweek_answerCode_sum',
        'dayofweek_answerCode_var',
        'mid3',
        'KnowledgeTag_first3_answerCode_mean',
        'KnowledgeTag_first3_answerCode_count',
        'month_answerCode_var',
        'month_answerCode_count',
    ]
    train, valid, y_train, y_valid, test, cat_features = category_data_loader(IS_CUSTOM=IS_CUSTOM, USE_VALID=USE_VALID, encode='ordinal', strategy='quantile', DROPS=DROPS, valid_len=valid_len, PCA=PCA, n_components=n_components)
    
    print(f"train : {train.shape}")
    print(f"valid : {valid.shape}")
    print(f"test : {test.shape}")
    print(f"cat_features : {cat_features}")