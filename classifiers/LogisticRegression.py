from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression(penalty='l2',
                        C=15.0,
                        solver='sag',
                        max_iter=150)
