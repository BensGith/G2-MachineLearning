import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import numpy as np

from imputers.DistributionImputer import DistributionImputer
from outliers.remove_outlier_stddev import remove_outlier_stddev
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


def process(data, process_func, train=False):
    data.drop(columns=['19'], axis=1, inplace=True)
    if train:
        num_cols = list(map(str.strip, data._get_numeric_data().columns))[:-1]
        rows = remove_outlier_stddev(data[num_cols])
        data.drop(index=rows, axis=0, inplace=True)
    data = process_func(data, train)
    l_col = data['label']
    data.drop(columns=['label'], axis=1, inplace=True)
    data = pd.get_dummies(data)
    return pd.concat([data, l_col], axis=1)


def process_pca(data, train=False):
    pass
    return data


def process_feature_selection(data, train=False):
    # train will be imputed differently than test
    data['2'] = data['2'].apply(lambda x: int(x[:-1]) if type(x) == str else x)  # remove  "d" suffix cast as int
    data['12'] = data['12'].replace(['y', 'n'], [1, 0])  # change to binary
    data['18'] = data['18'].apply(
        lambda x: int(x[1:]) if type(x) == str else x)  # drop the leading "a" in column, cast as int
    data['5'] = data['5'].apply(lambda x: float(x) if type(x) == str else x)
    data['8'] = data['8'].apply(lambda x: int(x) if type(x) == str else x)
    data['14'] = data['14'].fillna(-1)  # fill with -1 where NaN, drop samples
    # df.drop(columns=['0', '2', '15', '17', '19', 'label'], axis=1, inplace=True)
    data.dropna(inplace=True)
    return data



def main():
    # decide what's the best way to process the data
    df = pd.read_csv('train.csv')
    # remove outliers
    scalers = [MinMaxScaler(), StandardScaler()]
    dimension_reduction = []
    imputers = [DistributionImputer(), SimpleImputer(strategy='median')]
    for scale in scalers:
        pass


df = pd.read_csv('train.csv')
df = process_pca(df)
label = df['label'].tolist()
scaler = MinMaxScaler()
scaler.fit(df)
normal_data = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
# fill NaN with median
df = normal_data.apply(lambda x: x.fillna(x.median()), axis=1)

pca = PCA(0.95, svd_solver='full')
df = pca.fit_transform(df.loc[:, df.columns != 'label'])  # pca object holds the fit
df = pd.DataFrame(df)
df['label'] = label
# split to train, validation

kf = KFold(n_splits=5, random_state=None, shuffle=True)

for train_index, validate_index in kf.split(df):
    train = df.iloc[train_index].loc[:, df.columns != 'label']  # get rows by index list, drop label column
    validate = df.iloc[validate_index].loc[:, df.columns != 'label']
    train_label = np.array(df['label'].iloc[train_index])
    validate_label = np.array(df['label'].iloc[validate_index])
    knn = KNeighborsClassifier(3)
    knn.fit(train, train_label)
    predictions = knn.predict(validate)
    print(roc_auc_score(validate_label, predictions))
# validate_pca = pca.transform(validation.loc[:,train.columns!='label'])

# knn = KNeighborsClassifier(3)
# knn.fit(train, train_label)
# predictions = knn.predict(validation)

print("x")
