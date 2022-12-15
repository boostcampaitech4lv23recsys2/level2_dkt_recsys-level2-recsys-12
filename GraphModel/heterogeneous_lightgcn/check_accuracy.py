import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

def check_accuracy(PRED_PATH = "Boosting/output/lgbm_submission.csv"):
    threshold = 0.5

    ANSWER_PATH = "/opt/ml/input/data/custom_answer.csv"

    submission_result = pd.read_csv(PRED_PATH)
    answer = pd.read_csv(ANSWER_PATH)

    y_pred, y = submission_result["prediction"], answer["prediction"]

    return roc_auc_score(y,y_pred), accuracy_score(y,y_pred.apply(lambda x: 1 if x > threshold else 0))

if __name__ == "__main__":
    auc, acc = check_accuracy()
    print(f"auc : {auc} \t\t acc : {acc}")