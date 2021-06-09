import pandas as pd
from processing.basic_process import basic_process
import csv


def predict_test(clf, encoder, imputer, scaler, pca):
    data = pd.read_csv('train.csv')
    data = basic_process(data)  # only drop feature 19
    stages = [encoder, imputer, scaler, pca]
    for stage in stages:
        data = stage.transform(data)
    return clf.predict(data)


def write_csv(predicted_labels):
    pass








