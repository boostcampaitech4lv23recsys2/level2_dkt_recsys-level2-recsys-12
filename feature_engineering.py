import numpy as np
import pandas as pd

def get_groupby_user_features(df):
    """AnswerRate and solvedCount groupby userID"""
    answer_rate = df.groupby("userID")["answerCode"].agg(["mean", "count"])
    answer_rate.columns = ["userAnswerRate", "userSolvedLen"]
    new_df = pd.merge(df, answer_rate, how="left", on="userID")
    return new_df


def get_groupby_test_features(df):
    """AnswerRate and solvedCount groupby testId"""
    answer_rate = df.groupby("testId")["answerCode"].agg(["mean", "count"])
    answer_rate.columns = ["testAnswerRate", "testSolvedLen"]
    new_df = pd.merge(df, answer_rate, how="left", on="testId")
    return new_df

def get_groupby_item_features(df):
    """AnswerRate and solvedCount groupby assessmentItemID"""
    answer_rate = df.groupby('assessmentItemID')["answerCode"].agg(["mean","count"])
    answer_rate.columns=["itemAnswerRate","itemSolvedLen"]
    new_df = pd.merge(df,answer_rate, how='left', on='assessmentItemID')
    return new_df

def get_groupby_tag_features(df):
    """AnswerRate and solvedCount groupby KnowledgeTag"""
    answer_rate = df.groupby("KnowledgeTag")["answerCode"].agg(["mean", "count"])
    answer_rate.columns = ["tagAnswerRate", "tagSolvedLen"]
    new_df = pd.merge(df, answer_rate, how="left", on="KnowledgeTag")
    return new_df


def split_time(data):
    """Split Timestamp into year, month, day, hour, minute and second"""
    new_data = data.copy()
    new_data["temp1"] = new_data["Timestamp"].apply(lambda x: x.split()[0])
    new_data["temp2"] = new_data["Timestamp"].apply(lambda x: x.split()[1])

    new_data["year"] = new_data["temp1"].apply(lambda x: x.split("-")[0])
    new_data["month"] = new_data["temp1"].apply(lambda x: x.split("-")[1])
    new_data["day"] = new_data["temp1"].apply(lambda x: x.split("-")[2])

    new_data["hour"] = new_data["temp2"].apply(lambda x: x.split(":")[0])
    new_data["minute"] = new_data["temp2"].apply(lambda x: x.split(":")[1])
    new_data["second"] = new_data["temp2"].apply(lambda x: x.split(":")[2])

    return new_data.drop(["temp1", "temp2"], axis=1)


def get_first3(data):
    """Get first3 token from assessmentItemID"""
    data["first3"] = data["assessmentItemID"].apply(lambda x: x[1:4])
    return data


def get_time_concentration(data):
    """
    Get answerRate and concentrationLevel groupby hour.
    over 0.65 -> 2
    0.63~0.65 -> 1
    less 0.63 -> 0
    Count value of user groupby concentrationLevel = 1:2:2
    """
    if "hour" not in data.columns:
        data = split_time(data)
    timeConcentration = data.groupby("hour")["answerCode"].agg(["mean", "count"])
    timeConcentration.columns = ["timeConcentrationRate", "timeConcentrationCount"]
    new_df = pd.merge(data, timeConcentration, how="left", on="hour")
    new_df["timeConcentrationLevel"] = new_df["timeConcentrationRate"].apply(
        lambda x: 2 if x > 0.65 else 0 if x < 0.63 else 1
    )
    return new_df


def get_user_log(data):
    """
    get features about user`s prev solved problems(about user`s history).
    user_correct_answer : Number of correct answers to previously solved questions by the user
    user_total_answer : Number of previous problems solved by the user
    user_acc : Answer rate of previous problems solved by the user
    """
    data["user_correct_answer"] = data.groupby("userID")["answerCode"].transform(
        lambda x: x.cumsum().shift(1)
    )
    data["user_total_answer"] = data.groupby("userID")["answerCode"].cumcount()
    data["user_acc"] = data["user_correct_answer"] / data["user_total_answer"]
    return data


def get_seoson_concentration(data):
    """
    Get features abount month
    monthAnswerRate : Monthly correct answer rate
    monthSolvedCount : Monthly solved count
    """
    if "month" not in data.columns:
        data = split_time(data)
    groupby_month = data.groupby("month")["answerCode"].agg(["mean", "count"])
    groupby_month.columns = ["monthAnswerRate", "monthSolvedCount"]
    new_df = pd.merge(data, groupby_month, how="left", on="month")
    return new_df


ADD_LIST = [
    get_groupby_user_features,
    get_groupby_test_features,
    get_groupby_tag_features,
    get_groupby_item_features,
    split_time,
    get_first3,
    get_time_concentration,
    get_user_log,
    get_seoson_concentration,
]
DROP_LIST = []


def feature_engineering(df):
    """
    Make features in ADD_LIST (not in DROP_LIST)
    """
    for func in ADD_LIST:
        df = func(df)
    return df.drop(DROP_LIST, axis=1)
