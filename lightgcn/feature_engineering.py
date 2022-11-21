import pandas as pd


def get_answer_rate(df):
    # Todo
    answer_rate = df.groupby('userID').agg({
        "answerCode":"mean"
    })
    answer_rate.columns=["answerRate"]
    df = pd.merge(df,answer_rate, how='left', on='userID')
    return df