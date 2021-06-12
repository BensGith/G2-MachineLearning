import pandas as pd
from processing.basic_process import basic_process
import csv


def predict_test(df, clf, encoder, imputer, scaler, pca):
    data = basic_process(df)  # only drop feature 19
    stages = [encoder, imputer, scaler, pca]
    for stage in stages:
        data = stage.transform(data)
    return clf.predict_proba(data)[:, 1]


def write_csv(predicted_probas):
    fieldnames = ['Sample', 'Predicted Probability']
    with open('predictions.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for i, prediction in enumerate(predicted_probas):
            writer.writerow({'Sample': i, 'Predicted Label': prediction})









