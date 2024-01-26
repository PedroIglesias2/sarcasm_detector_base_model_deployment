from dill import dump, load
from src import train
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
from src.data.make_dataset import main as make_dataset
from src.data.make_dataset import pre_process_corpus
from src.data.make_dataset import read_test_data
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

processed_path = os.getenv("PROCESSED_PATH")
model_path = os.getenv("MODEL_PATH")
model_file_name = model_path + "/LogRegression.pkl"
cv_file_name = model_path + "/cv"
if not os.path.exists(model_file_name):
    train()
with open(model_file_name, "rb") as f:
    lr, cv = load(f)


X_test, y_test = read_test_data(processed_path)

norm_test_texts = pre_process_corpus(X_test['headline'].values)
cv_test_features = cv.transform(norm_test_texts)
# predict on test data
lr_bow_predictions = lr.predict(cv_test_features)
# Test model on test data
print(classification_report(y_test, lr_bow_predictions))
print(pd.DataFrame(confusion_matrix(y_test, lr_bow_predictions)))