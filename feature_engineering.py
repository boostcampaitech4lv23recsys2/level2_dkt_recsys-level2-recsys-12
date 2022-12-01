"""
    feature_engineering 함수를 실행시키면 아래 함수들이 차례대로 실행됩니다. 실행 후 Feature들이 생성된 상태의 DataFrame이 반환됩니다.

    get_groupby_user_features,
    get_groupby_test_features,
    get_groupby_tag_features,
    get_groupby_item_features,
    split_time,
    split_assessmentItemID,
    get_time_concentration,
    get_user_log,
    get_seoson_concentration,
"""
from datetime import datetime

import numpy as np
import pandas as pd


def make_datetime(val):
    a, b = val.split()
    return datetime(*list(map(int, a.split("-"))), *list(map(int, b.split(":"))))


def get_statistic_value(
    df: pd.DataFrame, group: list, target: str = "answerCode"
) -> pd.DataFrame:
    """Get target`s mean, count, var, sum, median groupby group"""
    if type(group) == str:
        group = [group]
    statistics = ["mean", "count", "sum", "var", "median"]
    new_df = df.groupby(group)[target].agg(statistics)
    new_df.columns = ["_".join(group + [target, i]) for i in statistics]
    return pd.merge(left=df, right=new_df, how="left", on=group)


def get_groupby_user_features(df):
    """Get statistic features / user, answerCode"""
    new_df = get_statistic_value(df, "userID", "answerCode")
    return new_df


def get_groupby_test_features(df):
    """Get statistic features / test, answerCode"""
    new_df = get_statistic_value(df, "testId", "answerCode")
    return new_df


def get_groupby_item_features(df):
    """Get statistic features / item, answerCode"""
    new_df = get_statistic_value(df, "assessmentItemID", "answerCode")
    return new_df


def get_groupby_tag_features(df):
    """Get statistic features / tag, answerCode"""
    new_df = get_statistic_value(df, "KnowledgeTag", "answerCode")
    return new_df


def get_groupby_hour_features(df):
    """Get statistic features / hour, answerCode"""
    if "hour" not in df.columns:
        df = split_time(df)
    new_df = get_statistic_value(df, "hour", "answerCode")
    return new_df


def get_groupby_month_features(df):
    """Get statistic features / month, answerCode"""
    if "month" not in df.columns:
        df = split_time(df)
    new_df = get_statistic_value(df, "month", "answerCode")
    return new_df


def get_groupby_dayofweek_features(df):
    """Get statistic features / dayofweek, answerCode"""
    if "dayofweek" not in df.columns:
        df = split_time(df)
    new_df = get_statistic_value(df, "dayofweek", "answerCode")
    return new_df


def get_groupby_user_first3_features(df):
    if "first3" not in df.columns:
        df = split_assessmentItemID(df)
    new_df = get_statistic_value(df, ["userID", "first3"], "answerCode")
    return new_df


def split_time(df):
    """Split Timestamp into year, month, day, hour, minute and second"""
    new_data = df.copy()
    if new_data["Timestamp"].dtype == "object":
        new_data["Timestamp"] = df["Timestamp"].apply(make_datetime)
    new_data["year"] = new_data["Timestamp"].dt.year
    new_data["month"] = new_data["Timestamp"].dt.month
    new_data["day"] = new_data["Timestamp"].dt.day
    new_data["hour"] = new_data["Timestamp"].dt.hour
    new_data["minute"] = new_data["Timestamp"].dt.minute
    new_data["second"] = new_data["Timestamp"].dt.second
    new_data["dayofweek"] = new_data["Timestamp"].dt.dayofweek

    return new_data


def split_assessmentItemID(df):
    """Split assessmentItemID into size=3 tokens"""
    df["first3"] = df["assessmentItemID"].apply(lambda x: x[1:4])
    df["mid3"] = df["assessmentItemID"].apply(lambda x: x[4:7])
    df["last3"] = df["assessmentItemID"].apply(lambda x: x[7:10])
    return df


def get_time_concentration(df):
    """
    Get answerRate and concentrationLevel groupby hour.
    over 0.65 -> 2
    0.63~0.65 -> 1
    less 0.63 -> 0
    Count value of user groupby concentrationLevel = 1:2:2
    """
    new_df = get_groupby_hour_features(df)
    new_df["hour_answerCode_Level"] = new_df["hour_answerCode_mean"].apply(
        lambda x: 2 if x > 0.65 else 0 if x < 0.63 else 1
    )
    return new_df


def get_user_log(df):
    """
    get features about user`s prev solved problems(about user`s history).
    user_correct_answer : Number of correct answers to previously solved questions by the user
    user_total_answer : Number of previous problems solved by the user
    user_acc : Answer rate of previous problems solved by the user
    """
    df["user_correct_answer"] = df.groupby("userID")["answerCode"].transform(
        lambda x: x.cumsum().shift(1)
    )
    df["user_total_answer"] = df.groupby("userID")["answerCode"].cumcount()
    df["user_acc"] = df["user_correct_answer"] / df["user_total_answer"]
    return df


def get_season_concentration(df):
    """
    Get features abount month
    monthAnswerRate : Monthly correct answer rate
    monthSolvedCount : Monthly solved count
    """
    new_df = get_groupby_month_features(df)
    return new_df


def get_elapsed_time(df):
    """Get elapsed time from 'Timestamp'"""
    df["elapsedTime"] = pd.to_timedelta(df["Timestamp"] - df["Timestamp"].shift(1))
    df["elapsedTime"] = df["elapsedTime"].dt.total_seconds()
    minus_idx = df["elapsedTime"] < 0
    df.loc[minus_idx, "elapsedTime"] = np.nan
    out_of_75_idx = df["elapsedTime"] > 136
    df.loc[out_of_75_idx, "elapsedTime"] = np.nan
    nan_idx = df["elapsedTime"].isnull()
    df.loc[nan_idx, "elapsedTime"] = df["elapsedTime"].mean()
    return df


def get_median_time(df):
    """Get median elapsed time of userID, KnowledgeTag, assessmentItemID, testId"""
    if "elapsedTime" not in df.columns:
        df = get_elapsed_time(df)
    agg_df = df.groupby("userID")["elapsedTime"].agg(["median"])
    agg_dict = agg_df.to_dict()
    df["userID_elapsedTime_median"] = df["userID"].map(agg_dict["median"])
    agg_df = df.groupby("KnowledgeTag")["elapsedTime"].agg(["median"])
    agg_dict = agg_df.to_dict()
    df["KnowledgeTag_elapsedTime_median"] = df["KnowledgeTag"].map(agg_dict["median"])
    agg_df = df.groupby("assessmentItemID")["elapsedTime"].agg(["median"])
    agg_dict = agg_df.to_dict()
    df["assessmentItemID_elapsedTime_median"] = df["assessmentItemID"].map(
        agg_dict["median"]
    )
    agg_df = df.groupby("testId")["elapsedTime"].agg(["median"])
    agg_dict = agg_df.to_dict()
    df["testId_elapsedTime_median"] = df["testId"].map(agg_dict["median"])
    return df


def get_median_time_with_answerCode(df):
    """
    Get median elapsed time of userID, KnowledgeTag, assessmentItemID, testId
    with answerCode
    """
    if "elapsedTime" not in df.columns:
        df = get_elapsed_time(df)
    agg_df = df.groupby(["userID", "answerCode"])["elapsedTime"].agg(["median"])
    agg_df.columns = ["userID_answerCode_elapsedTime_median"]
    df = pd.merge(left=df, right=agg_df, how="left", on=["userID", "answerCode"])
    agg_df = df.groupby(["KnowledgeTag", "answerCode"])["elapsedTime"].agg(["median"])
    agg_df.columns = ["KnowledgeTag_answerCode_elapsedTime_median"]
    df = pd.merge(left=df, right=agg_df, how="left", on=["KnowledgeTag", "answerCode"])
    agg_df = df.groupby(["assessmentItemID", "answerCode"])["elapsedTime"].agg(
        ["median"]
    )
    agg_df.columns = ["assessmentItemID_answerCode_elapsedTime_median"]
    df = pd.merge(
        left=df, right=agg_df, how="left", on=["assessmentItemID", "answerCode"]
    )
    agg_df = df.groupby(["testId", "answerCode"])["elapsedTime"].agg(["median"])
    agg_df.columns = ["testId_answerCode_elapsedTime_median"]
    df = pd.merge(left=df, right=agg_df, how="left", on=["testId", "answerCode"])
    return df


ADD_LIST = [
    get_groupby_user_features,
    get_groupby_test_features,
    get_groupby_item_features,
    get_groupby_tag_features,
    get_groupby_dayofweek_features,
    get_groupby_user_first3_features,
    get_user_log,
    split_assessmentItemID,
    split_time,
    get_time_concentration,
    get_season_concentration,
    get_elapsed_time,
    get_median_time,
    get_median_time_with_answerCode,
]


def feature_engineering(df):
    """
    Make features in ADD_LIST
    """
    for func in ADD_LIST:
        df = func(df)
    return df


#####################################################################
########################### SequneceModel ###########################

# ADD FUNCTIONS YOU WANT TO APPLY
SEQ_ADD_LIST = [
    split_assessmentItemID,
]
# ADD COLUMNS YOU WANT TO DROP
SEQ_DROP_LIST = []
# ADD COLUMNS WHOSE TYPE IS CATEGOCY
SEQ_CATE_COLS = [
    "assessmentItemID",
    "testId",
    "KnowledgeTag",
    "first3",
]

# FEATURE ENGINEERING FUNCTION FOR SEQUENCE MODEL
def seq_feature_engineering(df):
    """
    make features in SEQ_ADD_LIST (not in SEQ_DROP_LIST)
    """
    for func in SEQ_ADD_LIST:
        df = func(df)
    return df.drop(SEQ_DROP_LIST, axis=1), SEQ_CATE_COLS


#####################################################################
#####################################################################
