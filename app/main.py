import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import os
from sklearn.ensemble import RandomForestClassifier

modelsDir = os.getcwd() + "/models"

train = pd.read_csv(modelsDir + "/sample_submission.csv")
train_label = pd.read_csv(modelsDir + "/train_labels.csv")
test = pd.read_csv(modelsDir + "/test_features.csv")
submission = pd.read_csv(modelsDir + "/sample_submission.csv")

features = ['id', 'acc_x', 'acc_y', 'acc_z', 'gy_x', 'gy_y', 'gy_z']
X_train = train[features].groupby('id').agg(['max', 'min', 'mean'])
X_test = test[features].groupby('id').agg(['max', 'min', 'mean'])

y_train = train_label['label']

model = RandomForestClassifier(n_jobs=-1, random_state=0, min_samples_leaf=30)

y_pred = model.predict_proba(X_test)

submission.iloc[:, 1:] = y_pred

submission.to_csv('baseline_rf.csv', index=False)
