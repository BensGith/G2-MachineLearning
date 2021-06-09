import pandas as pd


class OneHotEncoder:
    def __init__(self):
        self.columns = set()

    def __repr__(self):
        return str(self.columns)

    def fit(self, data):
        data = data.drop(columns=['label'], axis=1)
        data = pd.get_dummies(data)
        for col in data.columns:
            self.columns.add(col)

    def transform(self, data):
        l_col = None
        if 'label' in data:
            l_col = data['label']
            data = data.drop(columns=['label'], axis=1)
        data = pd.get_dummies(data)
        cols = set([col for col in data.columns])   # find current columns
        missing_cols_test = self.columns - cols  # find columns that are fit has and missing
        extra_cols_test = cols - self.columns  # find columns that fit doesn't have
        for col in missing_cols_test:
            data[col] = 0  # if the column is missing, add it with 0s
        for col in extra_cols_test:
            data.drop(col, axis=1, inplace=True)  # if a col doesn't exist in self.columns, drop it
        if 'label' in data:
            return pd.concat([data, l_col], axis=1)
        else:
            return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
