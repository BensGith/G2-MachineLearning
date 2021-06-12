
def normalize(x):
    if x > 1:
        return 1
    elif x < 0:
        return 0
    else:
        return x


class CustomMinMax:
    def __init__(self):
        self.features = {}

    def fit(self, data):
        for col in data.columns:
            self.features[col] = [min(data[col]), max(data[col])]

    def transform(self, data):
        for i, col in enumerate(data.columns):
            data[col] = (data[col] - self.features[col][0])/(self.features[col][1] - self.features[col][0])
            data[col] = data[col].apply(lambda x:  normalize(x))
        return data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

