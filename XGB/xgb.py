from sklearn.datasets import load_boston
import xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
from dataloader import xgboost_preprocessing

X_train, X_test, y_train, y_test = xgboost_preprocessing()
