from sklearn.neighbors import KNeighborsClassifier


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

knn = KNeighborsClassifier(3)
knn.fit(train, train_label)
predictions = knn.predict(validation)
validate_pca = pca.transform(validation.loc[:,train.columns!='label'])

