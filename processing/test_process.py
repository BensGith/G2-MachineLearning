import pandas as pd
from processing.basic_process import basic_process
from classifiers.ANN import ann
from sklearn.decomposition import PCA


def predict_test(data, imputer, scaler, pca):
    data = pd.read_csv('train.csv')
    feature_names = list(data.columns.values)  # save column names
    data = basic_process(data)
    data = pd.DataFrame(imputer.transform(data))  # impute data
    data = data.set_axis(feature_names, axis=1, inplace=False)  # rename columns after imputing
    data = pd.DataFrame(pca.transform(data.drop(columns=['label'], axis=1)))




