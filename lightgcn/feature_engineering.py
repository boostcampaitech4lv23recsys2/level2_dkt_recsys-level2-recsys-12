import pandas as pd


def get_user_answer_rate(df):
    answer_rate = df.groupby("userID").agg({"answerCode": "mean"})
    answer_rate.columns = ["userAnswerRate"]
    new_df = pd.merge(df, answer_rate, how="left", on="userID")
    return new_df


def get_user_solved_len(df):
    answer_rate = df.groupby("userID").agg({"answerCode": "count"})
    answer_rate.columns = ["userSolvedLen"]
    new_df = pd.merge(df, answer_rate, how="left", on="userID")
    return new_df


def get_test_answer_rate(df):
    answer_rate = df.groupby("testId").agg({"answerCode": "mean"})
    answer_rate.columns = ["testAnswerRate"]
    new_df = pd.merge(df, answer_rate, how="left", on="testId")
    return new_df


def get_test_solved_len(df):
    answer_rate = df.groupby("testId").agg({"answerCode": "count"})
    answer_rate.columns = ["testSolvedLen"]
    new_df = pd.merge(df, answer_rate, how="left", on="testId")
    return new_df


ADD_LIST = [
    get_user_answer_rate,
    get_user_solved_len,
    get_test_answer_rate,
    get_test_solved_len,
]

DROP_LIST = []


def lightgcn_feature_engineering(df):
    for func in ADD_LIST:
        df = func(df)
    return df.drop(DROP_LIST, axis=1)
