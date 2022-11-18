import pandas as pd
import numpy as np


threshold=0.5


submission_result = pd.read_csv("lightgcn/output/submission.csv")
answer = pd.read_csv("custom_answer.csv")

result_for_accuracy = submission_result['prediction'].apply(lambda x: 0 if x < threshold else 1)


true_values=np.array([answer['prediction'].values])
predictions=np.array([result_for_accuracy.values])


N = true_values.shape[1]
accuracy = (true_values == predictions).sum() / N
TP = ((predictions == 1) & (true_values == 1)).sum()
FP = ((predictions == 1) & (true_values == 0)).sum()
precision = TP / (TP+FP)
print(precision)