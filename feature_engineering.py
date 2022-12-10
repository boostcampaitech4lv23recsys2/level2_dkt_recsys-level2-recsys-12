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
from tqdm import tqdm


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
    new_data["year"] = new_data["Timestamp"].dt.year.astype(int)
    new_data["month"] = new_data["Timestamp"].dt.month.astype(int)
    new_data["day"] = new_data["Timestamp"].dt.day.astype(int)
    new_data["hour"] = new_data["Timestamp"].dt.hour.astype(int)
    new_data["minute"] = new_data["Timestamp"].dt.minute.astype(int)
    new_data["second"] = new_data["Timestamp"].dt.second.astype(int)
    new_data["dayofweek"] = new_data["Timestamp"].dt.dayofweek.astype(int)

    return new_data


def split_assessmentItemID(df):
    """Split assessmentItemID into size=3 tokens"""
    df["first3"] = df["assessmentItemID"].apply(lambda x: int(x[1:4]) // 10 - 1)
    df["mid3"] = df["assessmentItemID"].apply(lambda x: int(x[4:7]) - 1)
    df["last3"] = df["assessmentItemID"].apply(lambda x: int(x[7:10]) - 1)
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
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["elapsedTime"] = pd.to_timedelta(df["Timestamp"] - df["Timestamp"].shift(1))
    df["elapsedTime"] = df["elapsedTime"].dt.total_seconds()
    minus_idx = df["elapsedTime"] < 0
    df.loc[minus_idx, "elapsedTime"] = np.nan
    out_of_time_idx = df["elapsedTime"] > 300
    df.loc[out_of_time_idx, "elapsedTime"] = np.nan
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


def get_elo_based_ratings(df):
    left_asymptote = 0
    # 찍을 확률 == 좌측 점근선 -> Riiid는 무조건 0.25보다 모든 것들이 큰데, 우리는 0도 가능함
    """
    Get ELO based rating features.
    assessmentItemID_elo_score: assessmentItemID based ELO rating score
    testId_elo_score: testId based ELO rating score
    KnowledgeTag_elo_score: KnowledgeTag based ELO rating score
    
    assessmentItemID_elo_score predicts better than the rest.
    """
    """
        theta의 정성적 의미:
            학생의 고유 능력(학습 상태라든가)
        세타 업데이트하는 ELO 수식 구현:
            is_good_answer: 
                정답 유무 (0 or 1)
            learning_rate_theta(nb_previous_answers):
                세타에 대한 learning rate 구하기
    """

    def get_new_theta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
        return theta + learning_rate_theta(nb_previous_answers) * (
            is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
        )

    """
        beta의 정성적 의미:
            문항 별 함수의 모수(문항별로 갖고 있는 고유한 특성 혹은 잠재 벡터 난이도라든가)
        베타 업데이트하는 ELO 수식 구현:
            is_good_answer:
                정답 유무 (0 or 1)
            learning_rate_theta(nb_previous_answers):
                베타에 대한 learning rate 구하기
    """

    def get_new_beta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
        return beta - learning_rate_beta(nb_previous_answers) * (
            is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
        )

    """
        theta의 정성적 의미:
            학생의 고유 능력(학습 상태라든가)
        세타에 대한 learning rate 구하기
    """

    def learning_rate_theta(nb_answers):
        return max(0.3 / (1 + 0.01 * nb_answers), 0.04)

    """
        beta의 정성적 의미:
            문항 별 함수의 모수(문항별로 갖고 있는 고유한 특성 혹은 잠재 벡터 난이도라든가)
        베타에 대한 learning rate 구하기
    """

    def learning_rate_beta(nb_answers):
        return 1 / (1 + 0.05 * nb_answers)

    """
        probability_of_good_answer의 정성적 의미:
            문항이 가진 고유 함수임. (찍는것과 난이도 고려하는 함수)
    """

    def probability_of_good_answer(theta, beta, left_asymptote):
        return left_asymptote + (1 - left_asymptote) * sigmoid(theta - beta)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def estimate_parameters(answers_df, granularity_feature_name):
        # 문항 별 함수의 모수(문항별로 갖고 있는 고유한 특성 혹은 잠재 벡터 난이도라든가) 를 추정하는 부분
        item_parameters = {
            granularity_feature_value: {"beta": 0, "nb_answers": 0}
            for granularity_feature_value in np.unique(
                answers_df[granularity_feature_name]
            )
        }
        # 학생의 고유 능력(학습 상태라든가)를 추정하는 부분
        student_parameters = {
            student_id: {"theta": 0, "nb_answers": 0}
            for student_id in np.unique(answers_df.userID)
        }

        print(f"{granularity_feature_name} based Parameter estimation is starting...")

        for student_id, item_id, left_asymptote, answered_correctly in tqdm(
            zip(
                answers_df.userID.values,
                answers_df[granularity_feature_name].values,
                answers_df.left_asymptote.values,
                answers_df.answerCode.values,
            )
        ):
            theta = student_parameters[student_id]["theta"]
            beta = item_parameters[item_id]["beta"]

            item_parameters[item_id]["beta"] = get_new_beta(
                answered_correctly,
                beta,
                left_asymptote,
                theta,
                item_parameters[item_id]["nb_answers"],
            )
            student_parameters[student_id]["theta"] = get_new_theta(
                answered_correctly,
                beta,
                left_asymptote,
                theta,
                student_parameters[student_id]["nb_answers"],
            )

            item_parameters[item_id]["nb_answers"] += 1
            student_parameters[student_id]["nb_answers"] += 1

        return student_parameters, item_parameters

    def get_elo(df, left_asymptote, granularity_feature_name):
        # 찍을 확률 == 좌측 점근선 -> Riiid는 무조건 0.25보다 모든 것들이 큰데, 우리는 0도 가능함
        df["left_asymptote"] = left_asymptote

        # 파라미터 추정해서: 학생의 고유 능력 & 문항 별 함수의 모수 추정
        student_parameters, item_parameters = estimate_parameters(
            df, granularity_feature_name=granularity_feature_name
        )

        pred = [
            probability_of_good_answer(
                student_parameters[student]["theta"],
                item_parameters[item]["beta"],
                left_asymptote,
            )
            for student, item in zip(
                df.userID.values, df[granularity_feature_name].values
            )
        ]

        df[f"{granularity_feature_name}_elo_pred"] = pred
        return df.drop(columns=["left_asymptote"])

    based_features = ["assessmentItemID", "testId", "KnowledgeTag"]
    left_asymptote = left_asymptote

    for feature_name in based_features:
        df = get_elo(df, left_asymptote, feature_name)

    # feature 간 앙상블
    # df["feature_ensemble_elo_pred"] = (1 / len(based_features)) * (df["assessmentItemID_elo_pred"] + df["testId_elo_pred"] + df["KnowledgeTag_elo_pred"])
    df["feature_ensemble_elo_pred"] = (
        0.5 * df["assessmentItemID_elo_pred"]
        + 0.25 * df["testId_elo_pred"]
        + 0.25 * df["KnowledgeTag_elo_pred"]
    )

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
    get_elo_based_ratings,
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
    get_groupby_user_features,
    get_groupby_test_features,
    get_groupby_item_features,
    get_groupby_tag_features,
    get_groupby_dayofweek_features,
    get_groupby_user_first3_features,
    split_assessmentItemID,
    split_time,
    get_time_concentration,
    get_season_concentration,
    get_elapsed_time,
    get_median_time,
    get_median_time_with_answerCode,
    get_elo_based_ratings,
]

# FEATURE ENGINEERING FUNCTION FOR SEQUENCE MODEL
def seq_feature_engineering(df):
    """
    Make features in ADD_LIST
    """
    for func in ADD_LIST:
        df = func(df)
    return df


# FEATURE ENGINEERING FUNCTION FOR LASTQUERY
# ADD FUNCTIONS YOU WANT TO APPLY
LQ_ADD_LIST = [
    get_elapsed_time,
    get_elo_based_ratings,
]
# ADD COLUMNS YOU WANT TO DROP
LQ_DROP_LIST = []


def lq_feature_engineering(df):
    for func in LQ_ADD_LIST:
        df = func(df)
    return df.drop(LQ_DROP_LIST, axis=1)


#####################################################################
#####################################################################
