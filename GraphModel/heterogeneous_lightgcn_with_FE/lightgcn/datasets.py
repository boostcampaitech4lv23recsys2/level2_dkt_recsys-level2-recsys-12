import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import sys
sys.path.append("../../")
from feature_engineering import feature_engineering

from config import CFG

# feature engineering 불러오기
user_features = CFG.user_features

item_features = CFG.item_features

# code/feature_engineering.py
def prepare_dataset(device, basepath, verbose=True, logger=None, isTrain=False):
    data = load_data(basepath)
    user_nunique = data.userID.nunique()
    
    if isTrain:
        train_data, test_data, valid_data, train_user_feature_engineering, train_item_feature_engineering, test_user_feature_engineering, test_item_feature_engineering, valid_user_feature_engineering, valid_item_feature_engineering = separate_data(data, device=device, isTrain=True)
        # user_nunique는 userid와 tag가 겹쳐서 key값이 서로 겹치는 상황을 방지하기 위한 offset 역할을 함
        user_problem_id_2_index, user_test_id_2_index, user_tag_id_2_index = indexing_data(data, user_nunique)
        
        train_data_proc = process_data(train_data, user_problem_id_2_index, user_test_id_2_index, user_tag_id_2_index, user_nunique, device)
        test_data_proc = process_data(test_data, user_problem_id_2_index, user_test_id_2_index, user_tag_id_2_index, user_nunique, device)
        valid_data_proc = process_data(valid_data, user_problem_id_2_index, user_test_id_2_index, user_tag_id_2_index, user_nunique, device)

        return train_data_proc, test_data_proc, valid_data_proc, len(user_problem_id_2_index), len(user_test_id_2_index), len(user_tag_id_2_index), train_user_feature_engineering, train_item_feature_engineering, test_user_feature_engineering, test_item_feature_engineering, valid_user_feature_engineering, valid_item_feature_engineering
    else:
        train_data, test_data, train_user_feature_engineering, train_item_feature_engineering, test_user_feature_engineering, test_item_feature_engineering = separate_data(data, device=device)
        user_problem_id_2_index, user_test_id_2_index, user_tag_id_2_index = indexing_data(data, user_nunique)
        
        train_data_proc = process_data(train_data, user_problem_id_2_index, user_test_id_2_index, user_tag_id_2_index, user_nunique, device)
        test_data_proc = process_data(test_data, user_problem_id_2_index, user_test_id_2_index, user_tag_id_2_index, user_nunique, device)
        
        return train_data_proc, test_data_proc, len(user_problem_id_2_index), len(user_test_id_2_index), len(user_tag_id_2_index), train_user_feature_engineering, train_item_feature_engineering, test_user_feature_engineering, test_item_feature_engineering


def load_data(basepath):
    if CFG.use_custom:
        path1 = os.path.join(basepath, "preprocessed_custom_train_data.csv")
        path2 = os.path.join(basepath, "preprocessed_custom_test_data.csv")
    else:
        path1 = os.path.join(basepath, "preprocessed_train_data.csv")
        path2 = os.path.join(basepath, "preprocessed_test_data.csv")
    data1 = pd.read_csv(path1)
    data2 = pd.read_csv(path2)
    data = pd.concat([data1, data2])
    return data


def separate_data(data, device, isTrain=False, test_size=CFG.train_test_split_rate):
    train_data = data[data.answerCode >= 0]
    test_data = data[data.answerCode < 0]
    
    test_user_feature_engineering = torch.Tensor(test_data[user_features].to_numpy()).float().to(device)
    test_item_feature_engineering = torch.Tensor(test_data[item_features].to_numpy()).float().to(device)
    if isTrain:
        x_train, x_valid, y_train, y_valid = train_test_split(
            train_data.drop("answerCode", axis=1),
            train_data["answerCode"],
            test_size=CFG.train_test_split_rate,
            random_state=42,
        )
        x_train["answerCode"] = y_train
        x_valid["answerCode"] = y_valid
        train_user_feature_engineering = torch.Tensor(x_train[user_features].to_numpy()).float().to(device)
        train_item_feature_engineering = torch.Tensor(x_train[item_features].to_numpy()).float().to(device)
        
        valid_user_feature_engineering = torch.Tensor(x_valid[user_features].to_numpy()).float().to(device)
        valid_item_feature_engineering = torch.Tensor(x_valid[item_features].to_numpy()).float().to(device)
        

        return x_train, test_data, x_valid, train_user_feature_engineering, train_item_feature_engineering, test_user_feature_engineering, test_item_feature_engineering, valid_user_feature_engineering, valid_item_feature_engineering
    else:
        train_user_feature_engineering = torch.Tensor(train_data[user_features].to_numpy()).float().to(device)
        train_item_feature_engineering = torch.Tensor(train_data[item_features].to_numpy()).float().to(device)
        return train_data, test_data, train_user_feature_engineering, train_item_feature_engineering, test_user_feature_engineering, test_item_feature_engineering


def indexing_data(data, user_nunique):
    # int64들을 str으로 바꿔줌  
    userid, problemid, testid, tagid = (
        sorted(list(map(str, set(data.userID)))),
        sorted(list(set(data.assessmentItemID))),
        sorted(list(set(data.testId))),
        sorted(list(map(lambda x: str(x + user_nunique), set(data.KnowledgeTag)))),
    )
    
    n_user, n_problem, n_test, n_tag = len(userid), len(problemid), len(testid), len(tagid)
    userid_2_index = {v: i for i, v in enumerate(userid)}
    
    # user_problem 그래프 구성
    problemid_2_index = {v: i + n_user for i, v in enumerate(problemid)}
    user_problem_id_2_index = dict(userid_2_index, **problemid_2_index)
    
    # user_test 그래프 구성
    testid_2_index = {v: i + n_user for i, v in enumerate(testid)}
    user_test_id_2_index = dict(userid_2_index, **testid_2_index)
    
    # user_tag 그래프 구성
    tagid_2_index = {v: i + n_user for i, v in enumerate(tagid)}
    user_tag_id_2_index = dict(userid_2_index, **tagid_2_index)
    
    u_keys = set([k for k in userid_2_index.keys()])
    p_keys = set([k for k in problemid_2_index.keys()])
    test_keys = set([k for k in testid_2_index.keys()])
    tag_keys = set([k for k in tagid_2_index.keys()])

    return user_problem_id_2_index, user_test_id_2_index, user_tag_id_2_index


def process_data(data, user_problem_id_2_index, user_test_id_2_index, user_tag_id_2_index, user_nunique, device, isTrain=False):
    user_problem_edge, user_test_edge, user_tag_edge, label = [], [], [], []
    
    for user, problem, test, tag, acode in zip(data.userID, data.assessmentItemID, data.testId, data.KnowledgeTag, data.answerCode):
        # int64들을 str으로 바꿔줌
        user_id, problem_id, test_id, tag_id = user_problem_id_2_index[str(user)], user_problem_id_2_index[problem], user_test_id_2_index[test], user_tag_id_2_index[str(tag + user_nunique)]
        
        user_problem_edge.append([user_id, problem_id])
        user_test_edge.append([user_id, test_id])
        user_tag_edge.append([user_id, tag_id])
        
        label.append(acode)

    user_problem_edge = torch.LongTensor(user_problem_edge).T
    user_test_edge = torch.LongTensor(user_test_edge).T
    user_tag_edge = torch.LongTensor(user_tag_edge).T
    label = torch.LongTensor(label)

    dict_data = [dict(edge=user_problem_edge.to(device), label=label.to(device)), dict(edge=user_test_edge.to(device), label=label.to(device)), dict(edge=user_tag_edge.to(device), label=label.to(device))]
    
    return dict_data