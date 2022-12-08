import os
import sys

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.append("../../")
from feature_engineering import feature_engineering

using_feature = ["KnowledgeTag", "first3"]  # 그 범주에 따라서, 문제의 정답률이 상이하다 -> 넣으면 좋아.

# code/feature_engineering.py
def prepare_dataset(device, basepath, verbose=True, logger=None, isTrain=False):
    if isTrain:
        # train_data.csv와 test_data.csv를 concat
        data = load_data(basepath)
        # separate_data(data, isTrain=True):
        #   return x_train, test_data, x_valid
        train_data, test_data, valid_data = separate_data(data, isTrain=True)
        # id2index: {userID:인덱스, ..., assessmentItemID:인덱스} 형태의 dict
        id2index, feature_sorted_list, num_features_list = indexing_data(data)
        # train_data_proc: {"edge":[유저,문제], "label":문제 맞췄는지} 형태의 dict임
        train_data_proc = process_data(train_data, id2index, device)
        test_data_proc = process_data(test_data, id2index, device)
        valid_data_proc = process_data(valid_data, id2index, device)
        if verbose:
            # edge, label 정보 출력
            print_data_stat(train_data, "Train", logger=logger)
            print_data_stat(test_data, "Test", logger=logger)
            print_data_stat(valid_data, "Valid", logger=logger)
        return (
            train_data_proc,
            test_data_proc,
            valid_data_proc,
            feature_sorted_list,
            num_features_list,
            len(id2index),
        )
    else:
        data = load_data(basepath)
        train_data, test_data = separate_data(data)
        id2index, feature_sorted_list, num_features_list = indexing_data(data)
        train_data_proc = process_data(train_data, id2index, device)
        test_data_proc = process_data(test_data, id2index, device)
        if verbose:
            print_data_stat(train_data, "Train", logger=logger)
            print_data_stat(test_data, "Test", logger=logger)
        return (
            train_data_proc,
            test_data_proc,
            feature_sorted_list,
            num_features_list,
            len(id2index),
        )


def load_data(basepath):
    path1 = os.path.join(basepath, "train_data.csv")
    path2 = os.path.join(basepath, "test_data.csv")
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)
    data1 = feature_engineering(data1)
    data2 = feature_engineering(data2)
    data = pd.concat([data1, data2])
    data.drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last", inplace=True
    )

    return data


def separate_data(data, isTrain=False, test_size=0.2):
    train_data = data[data.answerCode >= 0]
    test_data = data[data.answerCode < 0]
    if isTrain:
        x_train, x_valid, y_train, y_valid = train_test_split(
            train_data.drop("answerCode", axis=1),
            train_data["answerCode"],
            test_size=test_size,
            random_state=42,
        )
        x_train["answerCode"] = y_train
        x_valid["answerCode"] = y_valid
        return x_train, test_data, x_valid
    return train_data, test_data


def indexing_data(data):
    # userid: 고유한 userID들이 정렬된 리스트
    # itemid: 고유한 assessmentItemID들이 정렬된 리스트
    userid, itemid = (
        sorted(list(set(data.userID))),
        sorted(list(set(data.assessmentItemID))),
    )
    n_user, n_item = len(userid), len(itemid)

    featureid = [
        sorted(list(set(data[feature_name]))) for feature_name in using_feature
    ]
    n_feature = [len(val) for val in featureid]

    # {7439: 7439, 7440: 7440, 7441: 7441} 형태의 dict
    userid_2_index = {v: i for i, v in enumerate(userid)}
    # {"A090074005": 16894, ..., "A090074006": 16895} 형태의 dict
    itemid_2_index = {v: i + n_user for i, v in enumerate(itemid)}
    # userid_2_index와 itemid_2_index 를 하나의 dict로 합치기
    id_2_index = dict(userid_2_index, **itemid_2_index)

    # feature 정보 사용하기
    feature_data = (
        data[["assessmentItemID"] + using_feature]
        .groupby("assessmentItemID")
        .head(1)
        .copy()
    )
    # feature 각각에 대한 dict를 저장하는 list
    feature_2_index_list = [
        {v: i for i, v in enumerate(_featureid)} for _featureid in featureid
    ]

    feature_data["assessmentItemID"] = feature_data["assessmentItemID"].map(
        itemid_2_index
    )

    for i, feature_name in enumerate(using_feature):
        feature_data[feature_name] = feature_data[feature_name].map(
            feature_2_index_list[i]
        )

    feature_data = feature_data.sort_values(["assessmentItemID"] + using_feature)

    feature_sorted_list = [
        feature_data[feature_name].tolist() for feature_name in using_feature
    ]
    num_features_list = [n_user, n_item] + n_feature

    return id_2_index, feature_sorted_list, num_features_list


"""
    data:
        x_train, test_data, x_valid 중 하나
    id_2_index:
        {userID:인덱스, ..., assessmentItemID:인덱스} 형태의 dict
"""


def process_data(data, id_2_index, device, isTrain=False):
    edge, label = [], []
    # edge: [[64, 8932], ..., [7340, 12539]] 형태
    # torch.LongTensor(edge).shape: torch.Size([len(data), 2])
    for user, item, acode in zip(data.userID, data.assessmentItemID, data.answerCode):
        uid, iid = id_2_index[user], id_2_index[item]
        edge.append([uid, iid])
        label.append(acode)
    # torch.LongTensor(edge).T.shape: torch.Size([2, len(data)])
    edge = torch.LongTensor(edge).T
    # torch.LongTensor(label).shape: torch.Size([len(data)])
    label = torch.LongTensor(label)
    # keys: ["edge", "label"]
    # values: [edge, label]
    return dict(edge=edge.to(device), label=label.to(device))


def print_data_stat(data, name, logger):
    userid, itemid = list(set(data.userID)), list(set(data.assessmentItemID))
    n_user, n_item = len(userid), len(itemid)

    logger.info(f"{name} Dataset Info")
    logger.info(f" * Num. Users    : {n_user}")
    logger.info(f" * Max. UserID   : {max(userid)}")
    logger.info(f" * Num. Items    : {n_item}")
    logger.info(f" * Num. Records  : {len(data)}")
