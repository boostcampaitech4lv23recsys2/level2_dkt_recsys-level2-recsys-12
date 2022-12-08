import pandas as pd

import feature_engineering as fe

train_data = pd.read_csv("../data/train_data.csv")
test_data = pd.read_csv("../data/test_data.csv")
featured_train_data = fe.feature_engineering(train_data)
featured_test_data = fe.feature_engineering(test_data)
featured_train_data.to_csv("../data/featured_train_data.csv", index=False)
featured_test_data.to_csv("../data/featured_test_data.csv", index=False)
