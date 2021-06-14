import pandas as pd


class LabelEncoder:
    def __init__(self):
        self.columns = set()

    def __repr__(self):
        return str(self.columns)

    def fit(self, data):
        data = pd.get_dummies(data)  # transform categorical features into binary ones
        for col in data.columns:  # save columns
            self.columns.add(col)

    def transform(self, data):
        data['2'] = data['2'].apply(lambda x: int(x[:-1]) if type(x) == str else x)  # remove  "d" suffix cast as int
        data['12'] = data['12'].replace(['y', 'n'], [1, 0])  # change to binary
        data['18'] = data['18'].apply(lambda x: int(x[1:]) if type(x) == str else x)  # drop the leading "a" in column, cast as int
        data['1'] = data['1'].replace(['a', 'b','unknown'],[0,1,2])
        data['6'] = data['6'].replace(['A', 'B', 'C','D','E','F','G','H','I','J','K'], [0,1,2,3,4,5,6,7,8,9,10])
        data['16'] = data['16'].replace(['D', 'M','S'], [0,1, 2])

        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
