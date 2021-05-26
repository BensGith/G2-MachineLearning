import scipy.stats as st
import numpy as np
import pandas as pd


class DistributionImputer:
    def __init__(self):
        self.distributions = []  # holds a tuple of distribution name with most likely fitted parameters
        self.distribution_names = [st.chi2, st.f, st.norm, st.expon]
        self.numeric_cols = []
        self.function_map = {"chi2": np.random.chisquare,
                             "norm": np.random.normal,
                             "expon": np.random.exponential,
                             "f": np.random.f}

    def fit(self, data):
        """
        finds most likely distribution and parameters
        :param data:
        :return:
        """

        for feature in data[:-1].columns:
            best_distribution = st.norm
            best_params = (0.0, 1.0)
            best_sse = np.inf
            if feature in set(([str(i) for i in range(21)])):
                data_no_nan = pd.Series(list(filter(lambda z: not pd.isnull(z), data[feature])))
                y, x = np.histogram(data_no_nan, bins=200, density=True)
                x = (x + np.roll(x, -1))[:-1] / 2.0
                for distribution in self.distribution_names:
                    params = distribution.fit(data_no_nan)
                    arg = params[:-2]
                    loc = params[-2]
                    scale = params[-1]
                    pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)  # probability density function
                    sse = np.sum(np.power(y - pdf, 2.0))
                    if best_sse > sse > 0:  # if the sse we found is smaller than previous best one, set it as best
                        best_distribution = distribution
                        if best_distribution.name =="norm":
                            best_params = params
                        elif best_distribution.name =="expon":
                            best_params = [params[1]]
                        else:
                            best_params = arg
                        best_sse = sse
                print(
                    f'Best fitted distribution for {feature} is {best_distribution.name}. Fitted params {best_params}')
                self.distributions.append((best_distribution.name, best_params))

    def transform(self, data):
        """
        imputes data based on feature distribution
        :return:
        """
        for i, feature in enumerate(data.columns):
            if feature in set(([str(i) for i in range(21)])):
                for j, cell in enumerate(data[feature]):
                    if pd.isnull(cell):
                        # call random function from dictionary, pass parameters from fit
                        data[feature].iloc[j] = self.function_map[self.distributions[i][0]](*self.distributions[i][1])

        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)
