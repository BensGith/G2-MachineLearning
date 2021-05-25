import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.model_selection import KFold
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('train.csv')


def process_data_1(data):
    data.drop(columns=['19'], axis=1, inplace=True)
    num_cols = list(map(str.strip,df._get_numeric_data().columns))[:-1]
    rows = remove_outlier_stddev(data[num_cols])
    data.drop(index=rows, axis=0, inplace=True)
    l_col = data['label']
    data.drop(columns=['label'], axis=1, inplace=True)
    data = pd.get_dummies(data)
    return pd.concat([data, l_col], axis=1)


def process_data_2(data):
    data['2'] = data['2'].apply(lambda x: int(x[:-1]) if type(x) == str else x)  # remove  "d" suffix cast as int
    data['12'] = data['12'].replace(['y', 'n'], [1, 0])  # change to binary
    data['18'] = data['18'].apply(lambda x: int(x[1:]) if type(x) == str else x)  # drop the leading "a" in column, cast as int
    data['5'] = data['5'].apply(lambda x: float(x) if type(x) == str else x)
    data['8'] = data['8'].apply(lambda x: int(x) if type(x) == str else x)
    data['14'] = data['14'].fillna(-1)  # fill with -1 where NaN, drop samples
    data.drop(columns=['19'], axis=1, inplace=True)
    #df.drop(columns=['0', '2', '15', '17', '19', 'label'], axis=1, inplace=True)
    data.dropna(inplace=True)
    l_col = data['label']
    data.drop(columns=['label'], axis=1, inplace=True)
    data = pd.get_dummies(data)
    # remove outliers
    return pd.concat([data, l_col], axis=1)




def remove_outlier_stddev(data):
    """
    find row index's containing outliers
    :param data: df
    :return: list of indexes
    """
    drop_rows = set()
    for col in data.columns:
        col_mean = data[col].mean()
        col_std = data[col].std()
        for i, x in enumerate(data[col]):
            if abs((x - col_mean) / col_std) > 3:
                drop_rows.add(i)
    return drop_rows


    # return data[data.apply(lambda x: np.abs((x - x.mean()) / x.std()) < 3).all(axis=1)]


# def remove_outlier_iqr(data):
#     Q3 = np.quantile(data[col], 0.75)
#     Q1 = np.quantile(data[col], 0.25)
#     IQR = Q3 - Q1
#
#     lower_range = Q1 - 1.5 * IQR
#     upper_range = Q3 + 1.5 * IQR
#     outlier_free_list = [x for x in data[col] if (
#             (x > lower_range) & (x < upper_range))]
#     filtered_data = data.loc[data[col].isin(outlier_free_list)]
#
#
# for i in data.columns:
#     removeOutliers(data, i)

df = process_data_1(df)
label = df['label'].tolist()
scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
scaler.fit(df)
normal_data = pd.DataFrame(scaler.transform(df), index=df.index, columns=df.columns)
df = normal_data.apply(lambda x: x.fillna(x.median()), axis=1)

pca = PCA(0.95,svd_solver='full')
pca.fit(df.loc[:,df.columns!='label'])
df = pca.transform(df.loc[:,df.columns!='label'])
df=pd.DataFrame(df)
df['label']=label
# split to train, validation

kf = KFold(n_splits=5, random_state=None, shuffle=True)

for train_index, validate_index in kf.split(df):
    train = df.iloc[train_index].loc[:,df.columns!='label'] # get rows by index list, drop label column
    validate = df.iloc[validate_index].loc[:,df.columns!='label']
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
#


print("x")

