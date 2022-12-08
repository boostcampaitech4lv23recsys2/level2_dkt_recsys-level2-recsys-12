import pandas as pd


def compare_acc(
    COMPARE_PATH="SequenceModel/output/lastquery_ensemble-aw_hidden_dim256.csv",
):
    threshold = 0.5
    SOTA_PATH = "/opt/ml/input/data/sota.csv"  # sota 파일은 data 폴더 안에 넣어주세요.
    sota = pd.read_csv(SOTA_PATH)
    compare = pd.read_csv(COMPARE_PATH)

    sota["prediction"] = sota["prediction"].apply(lambda x: 1 if x > threshold else 0)
    compare["prediction"] = compare["prediction"].apply(
        lambda x: 1 if x > threshold else 0
    )

    return sota.compare(compare)  # 개수만 확인하고 싶으면 return 값의 len을 출력하면 됩니다.


if __name__ == "__main__":
    compare_df = compare_acc()
    print(f"num_diff : {len(compare_df)}\n\n{compare_df}")
