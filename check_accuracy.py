import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

def get_accuracy(PRED_PATH = "XGB/XGB_grid_kfold_custom_submission_994629.csv"):
    threshold=0.5
    ANSWER_PATH = "../data/custom_answer.csv"

    submission_result = pd.read_csv(PRED_PATH)
    answer = pd.read_csv(ANSWER_PATH)

    y_pred, y = submission_result["prediction"], answer["prediction"]

    return f"accuracy_score: {accuracy_score(y,y_pred.apply(lambda x: 1 if x > threshold else 0))}\nroc  auc_score: {roc_auc_score(y,y_pred)}"

if __name__ == "__main__":
    print(get_accuracy())
