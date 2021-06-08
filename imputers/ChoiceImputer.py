import pandas as pd
import numpy as np


class ChoiceImputer:
    def __init__(self):
        self.feature_values = []

    def fit(self, data):

        for feature in data[:-1].columns:
            self.feature_values.append(list(filter(lambda z: not pd.isnull(z), data[feature])))

    def transform(self, data):
        """
        imputes data based on random choice from feature
        :return:
        """
        for i, feature in enumerate(data.columns):
            if feature in set(([str(i) for i in range(21)])):  # work on the original columns
                for j, cell in enumerate(data[feature]):
                    if pd.isnull(cell):
                        # call random function from dictionary, pass parameters from fit
                        random_impute = np.random.choice(self.feature_values[i])
                        data.iat[j, data.columns.get_loc(feature)] = random_impute

        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
