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
import pandas as pd
from datetime import datetime

def make_datetime(val):
    a, b = val.split()
    return datetime(*list(map(int, a.split('-'))), *list(map(int, b.split(':'))))


def get_statistic_value(df, col, target):
    """Get target`s mean, count, var, sum, median groupby col"""
    statistics = ["mean", "count", "sum", "var", "median"]
    statistic_df = df.groupby(col)[target].agg(statistics)
    statistic_df.columns = [col+"_"+target+"_"+i for i in statistics]
    new_df = pd.merge(df, statistic_df,how="left",on=col)
    return new_df


def get_groupby_user_features(df):
    """Get statistic features / user, answerCode"""
    new_df = get_statistic_value(df, "userID","answerCode")
    return new_df


def get_groupby_test_features(df):
    """Get statistic features / test, answerCode"""
    new_df = get_statistic_value(df, "testId","answerCode")
    return new_df


def get_groupby_item_features(df):
    """Get statistic features / item, answerCode"""
    new_df = get_statistic_value(df, "assessmentItemID","answerCode")
    return new_df


def get_groupby_tag_features(df):
    """Get statistic features / tag, answerCode"""
    new_df = get_statistic_value(df, "KnowledgeTag","answerCode")
    return new_df


def get_groupby_hour_features(df):
    """Get statistic features / hour, answerCode"""
    if "hour" not in df.columns:
        df = split_time(df)
    new_df = get_statistic_value(df, "hour","answerCode")
    return new_df


def get_groupby_month_features(df):
    """Get statistic features / month, answerCode"""
    if "month" not in df.columns:
        df = split_time(df)
    new_df = get_statistic_value(df, "month","answerCode")
    return new_df

def get_groupby_dayofweek_features(df):
    """Get statistic features / dayofweek, answerCode"""
    if "datofweek" not in df.columns:
        df = split_time(df)
    new_df = get_statistic_value(df, "dayofweek", "answerCode")
    return new_df

def split_time(df):
    """Split Timestamp into year, month, day, hour, minute and second"""
    new_data = df.copy()
    if new_data["Timestamp"].dtype == "object":
        new_data["Timestamp"] = df["Timestamp"].apply(make_datetime)
    new_data["year"]=new_data["Timestamp"].dt.year
    new_data["month"]=new_data["Timestamp"].dt.month
    new_data["day"]=new_data["Timestamp"].dt.day
    new_data["hour"]=new_data["Timestamp"].dt.hour
    new_data["minute"]=new_data["Timestamp"].dt.minute
    new_data["second"]=new_data["Timestamp"].dt.second
    new_data["dayofweek"]=new_data["Timestamp"].dt.dayofweek

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
    new_df["hour_answerCode_Level"] = new_df["hour_answerCode_mean"].apply(lambda x: 2 if x > 0.65 else 0 if x < 0.63 else 1)
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


def get_seoson_concentration(df):
    """
    Get features abount month
    monthAnswerRate : Monthly correct answer rate
    monthSolvedCount : Monthly solved count
    """
    new_df = get_groupby_month_features(df)
    return new_df


ADD_LIST = [
    get_groupby_user_features,
    get_groupby_test_features,
    get_groupby_item_features,
    get_groupby_tag_features,
    get_groupby_dayofweek_features,
    get_user_log,
    split_assessmentItemID,
    split_time,
    get_time_concentration,
    get_seoson_concentration,
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
