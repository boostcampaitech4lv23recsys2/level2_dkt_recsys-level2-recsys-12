import pandas as pd

df = pd.read_csv('submission.csv', index_col=0)

def processing(n):
    if n < 0.25:
        return 0
    if n > 0.75:
        return 1
    else:
        return n

df['prediction'] = df['prediction'].apply(lambda x:processing(x))

df.to_csv("output.csv")