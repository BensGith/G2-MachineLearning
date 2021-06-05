from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2',
                        C=0.1,
                        max_iter=150)
