import pandas as pd
from processing.basic_process import basic_process
import csv


def predict_test(df, clf, encoder, imputer, scaler, pca):
    data = basic_process(df)  # only drop feature 19
    print()
    stages = [encoder, imputer, scaler, pca]
    for stage in stages:
        data = stage.transform(data)
    return clf.predict(data)


def write_csv(predicted_labels):
    fieldnames = ['Sample', 'Predicted Label']
    with open('predictions.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for i, prediction in enumerate(predicted_labels):
            writer.writerow({'Sample': i, 'Predicted Label': prediction})









