import os

import matplotlib.pyplot as plt
import xgboost
from dataloader import xgboost_preprocessing
from sklearn.datasets import load_boston
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = xgboost_preprocessing()
