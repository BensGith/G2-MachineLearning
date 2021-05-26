import scipy.stats as st
import numpy as np


class DistributionImputer:
    def __init__(self):
        self.distributions = []  # holds a tuple of distribution name with most likely fitted parameters
        self.distribution_names = [st.chi, st.chi2, st.f, st.norm, st.pareto, st.erlang, st.expon]

    def fit(self, data):
        """
        finds most likely distribution and parameters
        :param data:
        :return:
        """
        best_distribution = st.norm
        best_params = (0.0, 1.0)
        best_sse = np.inf
        num_cols = list(map(str.strip, data._get_numeric_data().columns))[:-1]  # impute only numeric columns
        for feature in data[num_cols].columns:
            y, x = np.histogram(data[feature], bins=200, density=True)
            x = (x + np.roll(x, -1))[:-1] / 2.0
            for distribution in self.distribution_names:
                params = distribution.fit(data)
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)  # probability density function
                sse = np.sum(np.power(y - pdf, 2.0))
                if best_sse > sse > 0:  # if the sse we found is smaller than previous best one, set it as best
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse
            print(f'Best fitted distribution for {feature} is {best_distribution.name}. Fitted params {best_params}')
            self.distributions.append((best_distribution.name, best_params))

    def impute(self, data):
        """
        imputes data based on feature distribution
        :return:
        """