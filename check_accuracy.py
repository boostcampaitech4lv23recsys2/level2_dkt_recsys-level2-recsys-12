import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

threshold=0.5
PRED_PATH = "lightgcn/output/submission.csv"
ANSWER_PATH = "../data/custom_answer.csv"

submission_result = pd.read_csv(PRED_PATH)
answer = pd.read_csv(ANSWER_PATH)

y_pred, y = submission_result["prediction"], answer["prediction"]

print(f"accuracy_score: {accuracy_score(y,y_pred.apply(lambda x: 1 if x > threshold else 0))}")
print(f"roc  auc_score: {roc_auc_score(y,y_pred)}")
