import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import numpy as np

from imputers.DistributionImputer import DistributionImputer
from outliers.remove_outlier_stddev import remove_outlier_stddev
from helper.feature_classfier import classify_features
from sklearn.impute import SimpleImputer
from processing.feature_selection import process_feature_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


def basic_process(data, train=False):
    data.drop(columns=['19'], axis=1, inplace=True)
    feature_classes = classify_features(data)
    numeric_features = feature_classes.get("numerical")
    if train:
        rows = remove_outlier_stddev(data[numeric_features])
        data.drop(index=rows, axis=0, inplace=True)
    return data


def one_hot_encode(data):
    l_col = data['label']
    data = data.drop(columns=['label'], axis=1)
    data = pd.get_dummies(data)
    return pd.concat([data, l_col], axis=1)


def main():
    # decide what's the best way to process the data
    df = pd.read_csv('train.csv')
    df = basic_process(df, train=True)
    scalers = []  # holds the list of scalers
    data_sets = [one_hot_encode(df), one_hot_encode(process_feature_selection(df))]
    imputers = [DistributionImputer(), SimpleImputer(strategy='median') ]
    imputed_sets = []
    pca_data = []
    explained_var = [0.9, 0.95, 0.98]
    for data_set in data_sets:
        scaler = MinMaxScaler()
        data_set = pd.DataFrame(scaler.fit_transform(data_set), index=df.index, columns=data_set.columns) # scale data
        scalers.append(scaler)  # add scaler to scalers list
        for imputer in imputers:
            data_set = pd.DataFrame(imputer.fit_transform(data_set))
            print(any(data_set.isna().any()))  # verify impute filled all NaNs
            imputed_sets.append(imputer)
            for var in explained_var:
                pca = PCA(var, svd_solver='full')
                data = pca.fit_transform(data_set)
                pca_data.append(pca)





main()
df = pd.read_csv('train.csv')

#df = process(df)
#
# label = df['label'].tolist()
# scaler = MinMaxScaler()
# scaler.fit(df)
# normal_data = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
# # fill NaN with median
# df = normal_data.apply(lambda x: x.fillna(x.median()), axis=1)
#
# pca = PCA(0.95, svd_solver='full')
# df = pca.fit_transform(df.loc[:, df.columns != 'label'])  # pca object holds the fit
# df = pd.DataFrame(df)
# df['label'] = label
# # split to train, validation
#
# kf = KFold(n_splits=5, random_state=None, shuffle=True)
#
# for train_index, validate_index in kf.split(df):
#     train = df.iloc[train_index].loc[:, df.columns != 'label']  # get rows by index list, drop label column
#     validate = df.iloc[validate_index].loc[:, df.columns != 'label']
#     train_label = np.array(df['label'].iloc[train_index])
#     validate_label = np.array(df['label'].iloc[validate_index])
#     knn = KNeighborsClassifier(3)
#     knn.fit(train, train_label)
#     predictions = knn.predict(validate)
#     print(roc_auc_score(validate_label, predictions))
# # validate_pca = pca.transform(validation.loc[:,train.columns!='label'])
#
# # knn = KNeighborsClassifier(3)
# # knn.fit(train, train_label)
# # predictions = knn.predict(validation)

print("x")
