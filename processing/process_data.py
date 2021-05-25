
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import preprocessing

#import datawig
from sklearn.model_selection import train_test_split
data = pd.read_csv("train.csv")

df_train, df_test = train_test_split(data, 0.2, 0.8, shuffle=True)




def create_pca(x, n_components, y):
    """
    create prinicipal component analysis with transformed data
    :param x: normalized transformed data set
    :param n_components: precentage we want to gurantee variance explained
    :return:
    """
    pca = PCA(n_components)
    pca.fit(x, y)
    return pca


df = pd.read_csv('train.csv')
label_col = df['label']
df['12'] = df['12'].replace(['y', 'n'], [1, 0])
df['18'] = df['18'].apply(lambda x: int(x[1:]) if type(x) == str else -1)  # drop the leading "a" in column, cast as int
df['5'] = df['5'].apply(lambda x: float(x) if type(x) == str else x)
df['8'] = df['8'].apply(lambda x: int(x) if type(x) == str else x)
df['14'] = df['14'].fillna(-1)  # fill with -1 where NaN, drop samples

# df.drop(columns=['0', '2', '4', '9', '10', '15', '17', '19', 'label'], axis=1, inplace=True)
df.drop(columns=['0', '2', '15', '17', '19', 'label'], axis=1, inplace=True)



# dropping decisions
# drop 17 because of strong correlation to 5
# drop col 9 because of strong correlation to 7


# pre - process data - drop columns 0, 2, 4, 10, 15, 19
# col 1 - categorize with OH - default NaN
# col 3 - default to mean (normally distributed)
# col 5 - default to mean (normally distributed)
# col 6 - categorize with OH - default NaN
# col 7 - default to mean
# col 8 - categorize with OH
# col 11 - value in range - check distribution
# col 12 - categorize with OH - add NaN to category
# col 13 - categorize with OH - add NaN to category
# col 14 - value in range - check distribution
# col 16 - categorize with OH - add NaN to category
# col 18 - drop/categorize with OH
# col 20 - value in range - check distribution

# PCA


# Feature selection


# drop col 14
# create new feature columns with get dummies (One Hot)

a = pd.get_dummies(df['1'], prefix="1")
b = pd.get_dummies(df['6'], prefix="6")
c = pd.get_dummies(df['16'], prefix="16")
df.drop(columns=['1', '6', '16'], axis=1, inplace=True)  # drop old feature cols
df = pd.concat([df, a, b, c], axis=1)  # concat to one df

df = (df-df.min())/(df.max()-df.min())  # normalize with MinMax

# create with DF with one hot
#



print(df.isna().any())
pca = create_pca(df, 0.95, label_col)
knn = KNeighborsClassifier(3)
knn.fit(pca.transform(df), label_col)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

