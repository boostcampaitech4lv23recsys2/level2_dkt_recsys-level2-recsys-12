import pandas as pd
import feature_engineering as fe

train_data = pd.read_csv("../data/train_data.csv")
featured_train_data = fe.feature_engineering(train_data)
featured_train_data = featured_train_data
featured_train_data.to_csv("../data/featured_train_data.csv", index=False)