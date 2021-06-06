import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
import numpy as np
from sklearn.neural_network import MLPClassifier
from imputers.DistributionImputer import DistributionImputer
from outliers.remove_outlier_stddev import remove_outlier_stddev
from graphs.ConfusionMatrix import ConfusionMatrix
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression
from helper.feature_classfier import classify_features
from sklearn.impute import SimpleImputer
from processing.feature_selection import process_feature_selection
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC


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
    final_data = []
    scalers = []  # holds the list of scalers
    data_sets = [one_hot_encode(df), one_hot_encode(process_feature_selection(df))]
    labels = data_sets[0]['label']
    imputers = [DistributionImputer(), SimpleImputer(strategy='median')]
    # imputers = [SimpleImputer(strategy='median')]
    imputed_sets = []
    pca_data = []
    explained_var = [0.9, 0.95, 0.98]
    for data_set in data_sets:
        scaler = MinMaxScaler()
        data_set = pd.DataFrame(scaler.fit_transform(data_set.copy()), index=df.index,
                                columns=data_set.columns)  # scale data
        scalers.append(scaler)  # add scaler to scalers list
        for imputer in imputers:
            feature_names = list(data_set.columns.values)  # save column names
            data_set = pd.DataFrame(imputer.fit_transform(data_set))  # impute data
            data_set = data_set.set_axis(feature_names, axis=1, inplace=False)  # rename columns after imputing
            imputed_sets.append(imputer)
            for var in explained_var:
                pca = PCA(var, svd_solver='full')
                data = pca.fit_transform(data_set.drop(columns=['label'], axis=1))
                pca_data.append(pca)
                final_data.append(pd.DataFrame(data))

    return final_data, labels


ANN_parametersOptions = {'activation': ["relu", "logistic"],  #
                         'hidden_layer_sizes': [(100,),  # 1 large hidden layer
                                                (50, 50),  # 2 medium size layers
                                                (20, 20, 10, 10, 10)],  # multiple small sized layers
                         'batch_size': [10, 50],
                         'learning_rate_init': [0.01, 0.001],
                         'alpha':[0.01,0.1,1],
                         # In some of the runs we saw that the network got stuck on a local min, for this reason we enlearge the defualt momentum
                         'max_iter': [200, 300, 350]}

ann_best = MLPClassifier(  # -----The architecture:------#
    activation="relu",  # What is the activation function between neurons {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}?
    hidden_layer_sizes=(50, 50),  # What is the architecture? what happens if we add more layers?
    alpha=0.01,  # The regularization: loss + alpha*W^2, you know it as lambda.
    # -----The optimizer:------#
    solver="sgd",  # Stochastic Gradient Descent, other optimizers are out of the scope of the course.
    learning_rate_init=0.01,  # What is the initial learning rate? in some optimizers the learning rate changes.

    learning_rate="invscaling",  # How does the learning rate update itself? {‘constant’, ‘invscaling’, ‘adaptive’}
    power_t=0.5,  # When we choose learning rate to be invscaling, it means that we multiply this number each epoch.

    early_stopping=False,
    # If True, then we set an internal validation data and stop training when there is no imporovement.
    tol=1e-4,  # A broad concept of converges, when we can say the algorithm converged?

    batch_size=10,  # The number of samples each batch.
    max_iter=200,  # The total number of epochs.
    warm_start=False,  # if we fit at the second time, do we start from the last fit?

    random_state=42  # seed
)

LR_parametersOptions = {'penalty': ['l2'],
                        'C': [0.001, 0.01, 0.1, 0.5, 1., 10., 100.],  # differnt regoltions valus
                        'max_iter': [150, 350, 1500]}  # max itertion for the algoritem

SVM_parameters = {'C': [0.001, 0.01, 0.1, 0.5, 1., 10., 100.],
                  'max_iter': [150, 350, 1500],
                  'degree':[1, 3, 5, 7],
                  'gamma': [1, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001],

                  'kernel': ['poly', 'rbf', 'sigmoid'],
                  'probability': [True]}

lr = LogisticRegression(penalty='l2',
                        C=0.1,
                        max_iter=150)

knn = KNeighborsClassifier(3)

svm = SVC(C=0.01, kernel='rbf', probability=True)

data_sets, labels = main()
#print(data_sets[2].isnull().values.any())
kf = KFold(n_splits=5, random_state=None, shuffle=True)
pp_option = []
train_pp_option = []

for i, df in enumerate([data_sets[2]]):
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

        # svm.fit(train,train_label)
        # predictions = svm.predict_proba(validate)
        # train_predictions = svm.predict_proba(train)
        # predicted_labels = svm.predict(validate)

        # lr.fit(train, train_label)
        # predictions = lr.predict_proba(validate)
        # train_predictions = lr.predict_proba(train)
        # predicted_labels = lr.predict(validate)

        ann_best.fit(train, train_label)
        predictions = ann_best.predict_proba(validate)
        predicted_labels = ann_best.predict(validate)
        train_predictions = ann_best.predict_proba(train)

        fpr, tpr, threshold = roc_curve(validate_label, predictions[:, 1])
        fpr_train, tpr_train, thresholds_train = roc_curve(train_label, train_predictions[:,1])

        plt.plot(fpr, tpr, 'b', color="gray")
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

    avg_tpr_test[:-1] = tprs_test[:-1] / 5
    avg_auc_test = auc(avg_fpr_test, avg_tpr_test)
    avg_tpr_train[:-1] = tprs_train[:-1] / 5
    avg_auc_train = auc(base_fpr, mean_tprs)
    plt.title('Receiver Operating Characteristic')
    blue_patch = mpatches.Patch(color='blue', label='Mean AUC test = %0.2f' % avg_auc_test)
    gray_patch = mpatches.Patch(color='gray', label='K-folds')
    red_patch = mpatches.Patch(color='red', label='Random Classifier', ls='--')
    plt.legend(handles=[blue_patch, gray_patch, red_patch], loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.plot(base_fpr, mean_tprs, 'b', label='Mean (AUC train = %0.2f, AUC validation = %0.2f)' % (avg_auc_train, 0))
    plt.show()
    pp_option.append((i, sum(scores) / len(scores)))
    train_pp_option.append((i, sum(train_scores) / len(train_scores)))
print(pp_option)
print(train_pp_option)

max_pp_index = sorted(pp_option, key=lambda x: x[1], reverse=True)[0][0]

# gs = GridSearchCV(ann_best, ANN_parametersOptions, cv=3, scoring='roc_auc')
# gs.fit(data_sets[max_pp_index].copy(), labels.copy())
# print(gs.best_params_)
# lr_gs = GridSearchCV(LogisticRegression(), LR_parametersOptions, cv=3, scoring='roc_auc')
# lr_gs.fit(data_sets[max_pp_index].copy(),labels.copy())
# print(lr_gs.best_params_)

# svm_gs = GridSearchCV(SVC(), SVM_parameters, cv=3, scoring='roc_auc')
# svm_gs.fit(data_sets[2].copy(), labels.copy())
# print(svm_gs.best_params_)
print("x")
