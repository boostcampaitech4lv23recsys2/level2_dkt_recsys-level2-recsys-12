import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from feature_engineering import lightgcn_feature_engineering

# code/feature_engineering.py
def prepare_dataset(device, basepath, verbose=True, logger=None, isTrain=False):
    if isTrain:
        data = load_data(basepath)
        train_data, test_data, valid_data = separate_data(data, isTrain=True)
        id2index = indexing_data(data)
        train_data_proc = process_data(train_data, id2index, device)
        test_data_proc = process_data(test_data, id2index, device)
        valid_data_proc = process_data(valid_data, id2index, device)
        if verbose:
            print_data_stat(train_data, "Train", logger=logger)
            print_data_stat(test_data, "Test", logger=logger)
            print_data_stat(valid_data, "Valid", logger=logger)
        return train_data_proc, test_data_proc, valid_data_proc, len(id2index)

    data = load_data(basepath)
    train_data, test_data = separate_data(data)
    id2index = indexing_data(data)
    train_data_proc = process_data(train_data, id2index, device)
    test_data_proc = process_data(test_data, id2index, device)
    if verbose:
        print_data_stat(train_data, "Train", logger=logger)
        print_data_stat(test_data, "Test", logger=logger)
    return train_data_proc, test_data_proc, len(id2index)


def load_data(basepath):
    path1 = os.path.join(basepath, "train_data.csv")
    path2 = os.path.join(basepath, "custom_test_data.csv")
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)
    data1 = lightgcn_feature_engineering(data1)
    data2 = lightgcn_feature_engineering(data2)
    data = pd.concat([data1, data2])
    data.drop_duplicates(
        subset=["userID", "assessmentItemID"], keep="last", inplace=True
    )

    return data


def separate_data(data, isTrain=False, test_size=0.2):
    train_data = data[data.answerCode >= 0]
    test_data = data[data.answerCode < 0]
    if isTrain:
        # breakpoint()
        x_train, x_valid, y_train, y_valid = train_test_split(train_data.drop('answerCode',axis=1), train_data['answerCode'], test_size=0.2, random_state=42)
        x_train['answerCode']=y_train
        x_valid['answerCode']=y_valid
        return x_train, test_data, x_valid
    return train_data, test_data


def indexing_data(data):
    userid, itemid = (
        sorted(list(set(data.userID))),
        sorted(list(set(data.assessmentItemID))),
    )
    n_user, n_item = len(userid), len(itemid)

    userid_2_index = {v: i for i, v in enumerate(userid)}
    itemid_2_index = {v: i + n_user for i, v in enumerate(itemid)}
    id_2_index = dict(userid_2_index, **itemid_2_index)

    return id_2_index


def process_data(data, id_2_index, device, isTrain=False):
    edge, label = [], []
    for user, item, acode in zip(data.userID, data.assessmentItemID, data.answerCode):
        uid, iid = id_2_index[user], id_2_index[item]
        edge.append([uid, iid])
        label.append(acode)

    edge = torch.LongTensor(edge).T
    label = torch.LongTensor(label)

    return dict(edge=edge.to(device), label=label.to(device))


def print_data_stat(data, name, logger):
    userid, itemid = list(set(data.userID)), list(set(data.assessmentItemID))
    n_user, n_item = len(userid), len(itemid)

    logger.info(f"{name} Dataset Info")
    logger.info(f" * Num. Users    : {n_user}")
    logger.info(f" * Max. UserID   : {max(userid)}")
    logger.info(f" * Num. Items    : {n_item}")
    logger.info(f" * Num. Records  : {len(data)}")
    
