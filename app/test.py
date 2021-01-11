import pandas as pd
import os

modelsDir = os.getcwd() + "/models"

train = pd.read_csv(modelsDir + "/sample_submission.csv")
train_labels = pd.read_csv(modelsDir + "/train_labels.csv")
test = pd.read_csv(modelsDir + "/test_features.csv")
submission = pd.read_csv(modelsDir + "/sample_submission.csv")

print(train_labels['label_desc'].nunique())
