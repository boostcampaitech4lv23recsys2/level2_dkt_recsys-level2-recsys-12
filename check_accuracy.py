import pandas as pd
import numpy as np
standard_threshold=0.4
submission_result = pd.read_csv("lightgcn/output/submission.csv")
answer = pd.read_csv("custom_answer.csv")

def get_acc(threshold):
    result_for_accuracy = submission_result['prediction'].apply(lambda x: 0 if x < threshold else 1)
    true_values=np.array([answer['prediction'].values])
    predictions=np.array([result_for_accuracy.values])
    N = true_values.shape[1]
    accuracy = (true_values == predictions).sum() / N
    # print((true_values == predictions).sum(),N)
    # TP = ((predictions == 1) & (true_values == 1)).sum()
    # FP = ((predictions == 1) & (true_values == 0)).sum()
    # precision = TP / (TP+FP)
    return accuracy
auroc = max_val = 0
for i in submission_result['prediction']:
    acc = get_acc(i)
    if auroc < acc:
        auroc = acc
        max_val = i
print(f'auroc : {auroc} in {max_val}')
print(f'acc : {get_acc(standard_threshold)} in {standard_threshold}')