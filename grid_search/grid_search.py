from sklearn.model_selection import GridSearchCV
from classifiers.ANN import ann
from classifiers.SVM import svm
from classifiers.LogisticRegression import logistic_regression
from classifiers.KNN import knn


class Grid:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.best_params = {}

    def find_best_params(self):

        ann_params = {
            'activation': ["relu", "logistic"],
            "solver": ["sgd"],
            'hidden_layer_sizes': [(100,), (50, 50), (20, 20, 10, 10, 10)],
            'batch_size': [10, 50, 100, 200],
            'learning_rate_init': [0.01, 0.001],
            'alpha': [0.01, 0.1],
            'max_iter': [1500]}

        lr_params = {
            'penalty': ['l2'],
            'C': [0.001, 0.01, 0.1, 0.5, 1., 10., 100.],
            'max_iter': [150, 350, 1500]}

        svm_parameters = {
            'C': [0.001, 0.01, 0.1, 0.5, 1., 10., 100.],
            'degree': [1, 3, 5, 7],
            'kernel': ['poly', 'rbf', 'sigmoid'],
            'probability': [True]}

        knn_params = {"n_neighbors": [x for x in range(31, 202, 10)]}
        # classifiers = [ann, svm, logistic_regression, knn]
        # params = [ann_params, svm_parameters, lr_params, knn_params]
        classifiers = [ann]
        params = [ann_params]
        for i, clf in enumerate(classifiers):
            gs = GridSearchCV(clf, params[i], cv=3, scoring='roc_auc')
            gs.fit(self.data, self.labels)
            self.best_params[clf.__class__.__name__] = gs.best_params_
        return self.best_params
