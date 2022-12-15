import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split

import sys
sys.path.append("../../")
from feature_engineering import feature_engineering

from config import CFG


# code/feature_engineering.py
def prepare_dataset(device, basepath, verbose=True, logger=None, isTrain=False):
    data = load_data(basepath)
    user_nunique = data.userID.nunique()
    
    if isTrain:
        train_data, test_data, valid_data = separate_data(data, isTrain=True)
        # user_nunique는 userid와 tag가 겹쳐서 key값이 서로 겹치는 상황을 방지하기 위한 offset 역할을 함
        user_problem_id_2_index, user_test_id_2_index, user_tag_id_2_index = indexing_data(data, user_nunique)
        
        train_data_proc = process_data(train_data, user_problem_id_2_index, user_test_id_2_index, user_tag_id_2_index, user_nunique, device)
        test_data_proc = process_data(test_data, user_problem_id_2_index, user_test_id_2_index, user_tag_id_2_index, user_nunique, device)
        valid_data_proc = process_data(valid_data, user_problem_id_2_index, user_test_id_2_index, user_tag_id_2_index, user_nunique, device)

        return train_data_proc, test_data_proc, valid_data_proc, len(user_problem_id_2_index), len(user_test_id_2_index), len(user_tag_id_2_index)
    else:
        train_data, test_data = separate_data(data)
        user_problem_id_2_index, user_test_id_2_index, user_tag_id_2_index = indexing_data(data, user_nunique)
        
        train_data_proc = process_data(train_data, user_problem_id_2_index, user_test_id_2_index, user_tag_id_2_index, user_nunique, device)
        test_data_proc = process_data(test_data, user_problem_id_2_index, user_test_id_2_index, user_tag_id_2_index, user_nunique, device)
        
        return train_data_proc, test_data_proc, len(user_problem_id_2_index), len(user_test_id_2_index), len(user_tag_id_2_index)


def load_data(basepath):
    path1 = os.path.join(basepath, "train_data.csv")
    if CFG.use_custom:
        path2 = os.path.join(basepath, "custom_test_data.csv")
    else:
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
            test_size=0.2,
            random_state=42,
        )
        x_train["answerCode"] = y_train
        x_valid["answerCode"] = y_valid
        return x_train, test_data, x_valid
    return train_data, test_data


def indexing_data(data, user_nunique):
    # int64들을 str으로 바꿔줌  
    userid, problemid, testid, tagid = (
        sorted(list(map(str, set(data.userID)))),
        sorted(list(set(data.assessmentItemID))),
        sorted(list(set(data.testId))),
        # tag랑 userid랑 겹치면 아래에서 dictionary들 합칠 때 중복된 키들 때문에 tag정보가 사라지므로 user_nunique를 offset으로 줌
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