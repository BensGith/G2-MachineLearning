import pandas as pd
import numpy as np


class ChoiceImputer:
    def __init__(self):
        self.max_values = []
        self.min_values = []

    def fit(self, data):

        for feature in data[:-1].columns:
            self.max_values.append(max(data[feature]))
            self.min_values.append(min(data[feature]))

    def transform(self, data):
        """
        imputes data based on feature distribution
        :return:
        """
        for i, feature in enumerate(data.columns):
            if feature in set(([str(i) for i in range(21)])):  # work on the original columns
                data_no_nan = pd.Series(list(filter(lambda z: not pd.isnull(z), data[feature])))
                for j, cell in enumerate(data[feature]):
                    if pd.isnull(cell):
                        # call random function from dictionary, pass parameters from fit
                        random_impute=np.random.choice(data_no_nan)
                        data.iat[j, data.columns.get_loc(feature)] = random_impute

        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)