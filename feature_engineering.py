import pandas as pd
import numpy as np

def get_user_answer_rate(df):
    answer_rate = df.groupby('userID').agg({
        "answerCode":"mean"
    })
    answer_rate.columns=["userAnswerRate"]
    new_df = pd.merge(df,answer_rate, how='left', on='userID')
    return new_df


def get_user_solved_len(df):
    answer_rate = df.groupby('userID').agg({
        "answerCode":"count"
    })
    answer_rate.columns=["userSolvedLen"]
    new_df = pd.merge(df,answer_rate, how='left', on='userID')
    return new_df


def get_test_answer_rate(df):
    answer_rate = df.groupby('testId').agg({
        "answerCode":"mean"
    })
    answer_rate.columns=["testAnswerRate"]
    new_df = pd.merge(df,answer_rate, how='left', on='testId')
    return new_df


def get_test_solved_len(df):
    answer_rate = df.groupby('testId').agg({
        "answerCode":"count"
    })
    answer_rate.columns=["testSolvedLen"]
    new_df = pd.merge(df,answer_rate, how='left', on='testId')
    return new_df


def split_time(data):
    new_data = data.copy()
    new_data["temp1"] = new_data["Timestamp"].apply(lambda x:x.split()[0])
    new_data["temp2"] = new_data["Timestamp"].apply(lambda x:x.split()[1])
    
    new_data["year"] = new_data["temp1"].apply(lambda x:x.split("-")[0])
    new_data["month"] = new_data["temp1"].apply(lambda x:x.split("-")[1])
    new_data["day"] = new_data["temp1"].apply(lambda x:x.split("-")[2])
    
    new_data["hour"] = new_data["temp2"].apply(lambda x:x.split(":")[0])
    new_data["minute"] = new_data["temp2"].apply(lambda x:x.split(":")[1])
    new_data["second"] = new_data["temp2"].apply(lambda x:x.split(":")[2])
    
    return new_data.drop(["temp1","temp2"],axis=1)


def get_first3(data):
    data["first3"]=data["assessmentItemID"].apply(lambda x:x[1:4])
    return data


ADD_LIST = [
    get_user_answer_rate,
    get_user_solved_len,
    get_test_answer_rate,
    get_test_solved_len,
    split_time,
    get_first3,
    
]
DROP_LIST = ["assessmentItemID", "testId", "Timestamp"]


def feature_engineering(df):
    for func in ADD_LIST:
        df = func(df)
    return df.drop(DROP_LIST, axis=1)