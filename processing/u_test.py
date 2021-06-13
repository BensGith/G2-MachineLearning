import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from graphs.AUC import AUC
from processing.test_process import write_csv, predict_test
import numpy as np
from imputers.DistributionImputer import DistributionImputer
from outliers.remove_outlier_stddev import remove_outlier_stddev
from graphs.ConfusionMatrix import ConfusionMatrix
from sklearn.metrics import roc_curve
import time
from imputers.ChoiceImputer import ChoiceImputer
import copy
from helper.feature_classfier import classify_features
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from processing.feature_selection import process_feature_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss
from processing.basic_process import basic_process
from classifiers.KNN import knn
from classifiers.ANN import ann
from classifiers.SVM import svm
from classifiers.LogisticRegression import logistic_regression
from processing.OneHotEncoder import OneHotEncoder
from scalers.CustomMinMax import CustomMinMax


start_time = time.time()

df = pd.read_csv('train.csv')
kf = KFold(n_splits=10, random_state=None, shuffle=True)
for train_index, validate_index in kf.split(df):
    train_df = pd.DataFrame.reset_index(df.iloc[train_index], drop=True)
    train_df = basic_process(train_df, train=True)
    train_labels = train_df['label']

    validate_df = pd.DataFrame.reset_index(df.iloc[validate_index].drop(columns=['label'], axis=1), drop=True)
    validate_labels = df.iloc[validate_index]['label']
    encoder = OneHotEncoder()
    #imputer = IterativeImputer(random_state=0)
    imputer = SimpleImputer(strategy='median')
    #imputer = ChoiceImputer()
    #scaler = MinMaxScaler()
    scaler = CustomMinMax()
    pca = PCA(0.95, svd_solver='full')
    train_df.drop(columns=['label'], axis=1, inplace=True)
    train_df = encoder.fit_transform(train_df)
    feature_names = list(train_df.columns.values)  # save column names
    print(all([1 in train_df[col] for col in train_df.columns if col[:-1] =='d']))
    train_df = pd.DataFrame(imputer.fit_transform(train_df))  # impute data
    train_df = train_df.set_axis(feature_names, axis=1, inplace=False)  # rename columns after imputing

    train_df = pd.DataFrame(scaler.fit_transform(train_df), index=train_df.index,
                            columns=train_df.columns)  # scale data
    train_df = pca.fit_transform(train_df)
    train_df = pd.DataFrame(train_df)
    ann.fit(train_df, train_labels)

    # process the validate df

    validate_df = basic_process(validate_df)
    validate_df = encoder.transform(validate_df)
    feature_names = list(validate_df.columns.values)  # save column names
    validate_df = pd.DataFrame(imputer.transform(validate_df))
    validate_df = validate_df.set_axis(feature_names, axis=1, inplace=False)  # rename columns after imputing

    validate_df = pd.DataFrame(scaler.transform(validate_df), index=validate_df.index,
                            columns=validate_df.columns)  # scale data

    validate_df = pca.transform(validate_df)
    validate_df = pd.DataFrame(validate_df)

    # make predictions

    validate_predictions = ann.predict_proba(validate_df)
    train_predictions = ann.predict_proba(train_df)
    predicted_labels = ann.predict(validate_df)
    train_predicted_labels = ann.predict(train_df)

    log_validate = log_loss(validate_labels, validate_predictions)
    log_train = log_loss(train_labels, train_predictions)
    score_validate = accuracy_score(validate_labels, predicted_labels)
    score_train = accuracy_score(train_labels, train_predicted_labels)
    auc_validate = (roc_auc_score(validate_labels, validate_predictions[:, 1]))
    auc_train = (roc_auc_score(train_labels, train_predictions[:, 1]))

    print(f'AUC validate is {auc_validate}')
    print(f'AUC train is {auc_train}')

    print(f'acc_validate is {score_validate}')
    print(f'acc_train is {score_train}')

    print(f'log_validate is {log_validate}')
    print(f'log_train is {log_train}')



