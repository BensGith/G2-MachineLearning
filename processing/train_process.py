import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from graphs.AUC import AUC
from sklearn.metrics import accuracy_score
#from graphs.AUC import plot_roc_curve

import matplotlib.patches as mpatches
import numpy as np
from imputers.DistributionImputer import DistributionImputer
from imputers.choice import ChoiceImputer
from outliers.remove_outlier_stddev import remove_outlier_stddev
from graphs.ConfusionMatrix import ConfusionMatrix
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.impute import KNNImputer
from helper.feature_classfier import classify_features
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from processing.feature_selection import process_feature_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from processing.basic_process import basic_process
from classifiers.ANN import ann
from classifiers.SVM import svm


def one_hot_encode(data):
    l_col = data['label']
    data = data.drop(columns=['label'], axis=1)
    data = pd.get_dummies(data)
    return pd.concat([data, l_col], axis=1)


def main():
    # decide what's the best way to process the data
    df = pd.read_csv('train.csv')
    df = basic_process(df, train=True)
    final_data = []
    scalers = []  # holds the list of scalers
    data_sets = [one_hot_encode(df.copy()), one_hot_encode(process_feature_selection(df.copy()))]
    labels = data_sets[0]['label']
    imputers = [IterativeImputer()]
    # imputers = [IterativeImputer()]
    #imputers = [DistributionImputer(), SimpleImputer(strategy='median'),ChoiceImputer()]
    imputed_sets = []
    pca_data = []
    explained_var = [0.9,0.95, 0.98]
    for data_set in data_sets:
        for imputer in imputers:
            feature_names = list(data_set.columns.values)  # save column names
            data_set = pd.DataFrame(imputer.fit_transform(data_set))  # impute data
            data_set = data_set.set_axis(feature_names, axis=1, inplace=False)  # rename columns after imputing
            scaler = MinMaxScaler()
            data_set = pd.DataFrame(scaler.fit_transform(data_set), index=df.index,
                                    columns=data_set.columns)  # scale data
            scalers.append(scaler)  # add scaler to scalers list
            imputed_sets.append(imputer)
            for var in explained_var:
                pca = PCA(var, svd_solver='full')
                data = pca.fit_transform(data_set.drop(columns=['label'], axis=1))
                pca_data.append(pca)
                final_data.append(pd.DataFrame(data))

    return final_data, labels


ANN_parametersOptions = {'activation': ["relu", "logistic"],  #
                         'hidden_layer_sizes': [(100,)],  # multiple small sized layers
                         'batch_size': [40,35],
                         'learning_rate_init': [0.0015,0.002],
                         'beta_1':[0.87,0.88],
                         'epsilon':[1e-8,1e-10],
                         'solver':['adam']}


LR_parametersOptions = {'penalty': ['l2'],
                        'C': [0.001, 0.01, 0.1, 0.5, 1., 10., 100.],  # differnt regoltions valus
                        'max_iter': [150, 350, 1500]}  # max itertion for the algoritem

SVM_parameters = {'C': [0.001, 0.01, 0.1, 0.5, 1., 10., 100.],
                  'max_iter': [150, 350, 1500],
                  'degree':[1, 3, 5, 7],
                  'gamma': [1, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001],

                  'kernel': ['poly', 'rbf', 'sigmoid'],
                  'probability': [True]}

logistic_regression = LogisticRegression(penalty='l2',
                        C=0.1,
                        max_iter=150)

knn = KNeighborsClassifier(3)

# svm = SVC(C=0.01, kernel='rbf', probability=True)

data_sets, labels = main()
print([d.isnull().values.any() for d in data_sets])

kf = KFold(n_splits=5, random_state=None, shuffle=True)
pp_option = []
train_pp_option = []

for i, df in enumerate(data_sets):
    scores = []
    train_scores = []
    # the X axis (average fpr)
    avg_fpr_test = np.linspace(0, 1, 100)
    # the Y axis (average tpr)
    avg_tpr_test = np.linspace(0, 1, 100)

    avg_fpr_train = np.linspace(0, 1, 100)
    avg_tpr_train = np.linspace(0, 1, 100)

    tprs_test = np.linspace(0, 0, 100)
    tprs_train = np.linspace(0, 0, 100)
    # Initialize the mean TPR
    mean_tpr = 0.0
    # Creates an array of 100 numbers between 0 and 1 in equal jumps
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    base_fpr = np.linspace(0, 1, 101)

    for train_index, validate_index in kf.split(df):
        train = df.iloc[train_index]  # get rows by index list, drop label column
        validate = df.iloc[validate_index]
        train_label = np.array(labels.iloc[train_index])
        validate_label = np.array(labels.iloc[validate_index])

        # knn.fit(train, train_label)
        # predictions = knn.predict_proba(validate)
        # train_predictions = knn.predict_proba(train)
        # predicted_labels = knn.predict(validate)

        # svm.fit(train, train_label)
        # predictions = svm.predict_proba(validate)
        # train_predictions = svm.predict_proba(train)
        # predicted_labels = svm.predict(validate)

        # logistic_regression.fit(train, train_label)
        # predictions = logistic_regression.predict_proba(validate)
        # train_predictions = logistic_regression.predict_proba(train)
        # predicted_labels = logistic_regression.predict(validate)

        ann.fit(train, train_label)
        predictions = ann.predict_proba(validate)
        predicted_labels = ann.predict(validate)
        train_predictions = ann.predict_proba(train)

        fpr, tpr, threshold = roc_curve(validate_label, predictions[:, 1])
        fpr_train, tpr_train, thresholds_train = roc_curve(train_label, train_predictions[:,1])


        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

        # confusion_mat = ConfusionMatrix("Logistic", validate_label, predicted_labels)

        # confusion_mat = ConfusionMatrix("KNN", validate_label, predicted_labels)
        # confusion_mat.plot()

        scores.append(roc_auc_score(validate_label, predictions[:, 1]))
        train_scores.append(roc_auc_score(train_label, train_predictions[:, 1]))
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)

    avg_auc_train = auc(base_fpr, mean_tprs)

    pp_option.append((i, sum(scores) / len(scores)))

    train_pp_option.append((i, sum(train_scores) / len(train_scores)))
print(pp_option)
print(train_pp_option)

max_pp_index = sorted(pp_option, key=lambda x: x[1], reverse=True)[0][0]

classifiers = [ann]
# classifiers = [ann,knn, logistic_regression, svm]
df = data_sets[1]  # chosen best set - no feature selection with OH, Distribution imputer, 0.98 PCA
auc_plot = AUC()
for i, clf in enumerate(classifiers):
    clf_name = clf.__class__.__name__
    kf = KFold(n_splits=5, random_state=None, shuffle=True)
    tprs_test = np.linspace(0, 0, 100)
    tprs_train = np.linspace(0, 0, 100)
    # Initialize the mean TPR
    mean_tpr = 0.0
    # Creates an array of 100 numbers between 0 and 1 in equal jumps
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    fpr_tprs = []
    base_fpr = np.linspace(0, 1, 101)
    for train_index, validate_index in kf.split(df):
        train = df.iloc[train_index]  # get rows by index list, drop label column
        validate = df.iloc[validate_index]
        train_label = np.array(labels.iloc[train_index])
        validate_label = np.array(labels.iloc[validate_index])

        clf.fit(train, train_label)
        predictions = clf.predict_proba(validate)
        train_predictions = clf.predict_proba(train)
        predicted_labels = clf.predict(validate)
        train_predicted_labels = clf.predict(train)

        score_validate = accuracy_score(validate_label, predicted_labels)
        score_train = accuracy_score(train_label, train_predicted_labels)
        print(f'score_validate is {score_validate}')
        print(f'score_train is {score_train}')

        fpr, tpr, threshold = roc_curve(validate_label, predictions[:, 1])
        fpr_train, tpr_train, thresholds_train = roc_curve(train_label, train_predictions[:, 1])

        fpr_tprs.append((fpr, tpr))
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    avg_auc_train = auc(base_fpr, mean_tprs)
    auc_plot.plot_auc(clf_name, fpr_tprs, base_fpr, mean_tprs,  avg_auc_train, i)
plt.show()

# gs = GridSearchCV(ann, ANN_parametersOptions, cv=3, scoring='roc_auc')
# gs.fit(data_sets[1].copy(), labels.copy())
# print(gs.best_params_)
# lr_gs = GridSearchCV(LogisticRegression(), LR_parametersOptions, cv=3, scoring='roc_auc')
# lr_gs.fit(data_sets[max_pp_index].copy(),labels.copy())
# print(lr_gs.best_params_)
#
# svm_gs = GridSearchCV(SVC(), SVM_parameters, cv=3, scoring='roc_auc')
# svm_gs.fit(data_sets[2].copy(), labels.copy())
# print(svm_gs.best_params_)
print("x")
