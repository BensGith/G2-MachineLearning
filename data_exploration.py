from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# from sklearn.preprocessing import Imputer , Normalizer , scale
# from sklearn.cross_validation import train_test_split , StratifiedKFold

from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns


#data exploration

# col 0 - int range (1 jumps - consistent range), min 1, max 51 and empty cells, avg 2.7
# col 1 - a, b, unknown empty cells
# col 2 - range of <number>d
# col 3 - range of floats - min -11.1, max 17.64, avg 4.204
# col 4 - range of ints - min -8019, max 102,127, avg 1383.257
# col 5 - 3.7 min, 10.3 max 6.58 avg
# col 6 - A-K and empty cells
# col 7 - ints with 10 jumps (10-310)
# col 8 - 0s and 1s (no empty cells)
# col 9 - -1.892 min, 347.19 max, 256.61 avg
# col 10 - 0 min, 41 max, 0.608 avg
# col 11 - 0.003 min, 108 max, 268.8 avg
# col 12 - y, n, empty cells
# col 13 - 0, 1, 2, 3 (avg 2.06)
# col 14 - 0 min, 4918 max, 276.022 -with empty cells
# col 15 - 0s and 1s
# col 16 - M,D,S
# col 17 - min 1.93, max 7.14, avg 3.64
# col 18 - a<number>
# col 19 - A, B, C, empty cells
# col 20 - min -54.54, max 102.59, avg 26.27


# histograms - 0,1, 2?, 6, 7, 10, 13, 16, 18, 19
# gaussian - 3,4,5, 9, 11, 17, 20


col_data = []
train_set = pd.read_csv('train.csv')
# uncomment for col2, col13, col15, col17 (nan being dropped automatically, None string creates error)
#train_set = train_set .replace(np.nan, 'None', regex=True)
for col in range(train_set.shape[1] - 1):  # iterate over columns but label col
    col_data.append(train_set[str(col)].tolist())

# #### col 0 ########
# plt.hist(col_data[0])
# plt.xticks([i for i in range(1,52, 5)])
# plt.title("Col0 Histogram")
# plt.show()

# #### col 1 ########
# plt.hist(col_data[1])
# plt.title("Col1 Histogram")
# plt.show()

# #### col 2 ########
# col2_dict = {}
# for value in col_data[2]:
#     if value in col2_dict:
#         col2_dict[value] += 1
#     else:
#         col2_dict[value] = 1
# histogram = sorted([(key, value) for key, value in col2_dict.items() ], key=lambda z: z[1], reverse=True)
# x, y = zip(*histogram[:10])
# plt.bar(x, y)
# # plt.hist(col_data[2], bins=10)
# plt.title("Col2 Histogram")
# plt.show()


# #### col 3 ########
# plt.hist(col_data[3], bins=50)
# plt.title("Col3 Histogram")
# plt.show()

# #### col 4 ########
# plt.hist(col_data[4], bins=30)
# plt.title("Col4 Histogram")
# plt.show()


# #### col 5 ########
# plt.hist(col_data[5], bins=50)
# plt.title("Col5 Histogram")
# plt.show()


# # #### col 6 ########
# plt.hist(col_data[6])
# plt.title("Col6 Histogram")
# plt.show()


# # #### col 7 ########
# plt.hist(col_data[7], bins=100)
# plt.title("Col7 Histogram")
# plt.show()


# # #### col 8 ########

# col8_dict = {}
# for value in col_data[8]:
#     if value in col8_dict:
#         col8_dict[value] += 1
#     else:
#         col8_dict[value] = 1
#
# plt.bar(range(len(col8_dict)), list(col8_dict.values()), align='center')
# plt.xticks(range(len(col8_dict)), list(col8_dict.keys()))
#
# plt.title("Col8 Histogram")
# plt.show()

# # #### col 9 ########
# plt.hist(col_data[9], bins=100)
# plt.title("Col9 Histogram")
# plt.show()


# # #### col 10 ########
# plt.hist(col_data[10],bins=50)
# plt.title("Col10 Histogram")
# plt.show()


# # #### col 11 ########
# plt.hist(col_data[11], bins=300)
# plt.title("Col11 Histogram")
# plt.show()
#

# # #### col 12 ########

# col12_dict = {}
# for value in col_data[12]:
#     if value in col12_dict:
#         col12_dict[value] += 1
#     else:
#         col12_dict[value] = 1
#
# plt.bar(range(len(col12_dict)), list(col12_dict.values()), align='center')
# plt.xticks(range(len(col12_dict)), list(col12_dict.keys()))
# plt.title("Col12 Histogram")
# plt.show()

# # #### col 13 ########

# labels, counts = np.unique(col_data[13], return_counts=True)
# plt.bar(labels, counts, align='center')
# plt.gca().set_xticks(labels)
# plt.title("Col13 Histogram")
# plt.show()

# # #### col 14 ########
# plt.hist(col_data[14], bins=300)
# plt.title("Col14 Histogram")
# plt.show()

# # #### col 15 ########
# labels, counts = np.unique(col_data[15], return_counts=True)
# plt.bar(labels, counts, align='center')
# plt.gca().set_xticks(labels)
# plt.title("Col15 Histogram")
# plt.show()

# # #### col 16 ########
# labels, counts = np.unique(col_data[16], return_counts=True)
# plt.bar(labels, counts, align='center')
# plt.gca().set_xticks(labels)
# plt.title("Col16 Histogram")
# plt.show()


# # #### col 17 ########
# plt.hist(col_data[17], bins=100)
# plt.title("Col17 Histogram")
# plt.show()


# # #### col 18 ########
# labels, counts = np.unique(col_data[18], return_counts=True)
# plt.bar(labels, counts, align='center')
# plt.gca().set_xticks(labels)
# plt.title("Col18 Histogram")
# plt.show()

# # #### col 19 ########
# labels, counts = np.unique(col_data[19], return_counts=True)
# plt.bar(labels, counts, align='center')
# plt.gca().set_xticks(labels)
# plt.title("Col19 Histogram")
# plt.show()


# #### col 20 ########
# plt.hist(col_data[20], bins=100)
# plt.title("Col20 Histogram")
# plt.show()
#


x = [i for i in range(1, train_set.shape[1])]

print(train_set.describe())
correlation_matrix = train_set.corr()
cov_matrix = train_set.cov()
y = np.array([1, 1, 2, 2])
pred = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
metrics.auc(fpr, tpr)


