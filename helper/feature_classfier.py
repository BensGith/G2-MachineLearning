import pandas as pd


def get_non_binary_cols(data):
    """
    get the indexes of the non-binary, numeric columns
    we will use this function to scale on these columns
    :return: list of indexes
    """
    num_cols = list(map(str.strip, data._get_numeric_data().columns))[:-1]
    indexes = []

    for feature in data[num_cols]:
        values = [x != 1 and x != 0 and not pd.isnull(x) for x in data[feature]]
        if any(values):
            indexes.append(feature)
    return indexes


def classify_features(data):
    """
    get classifications for our data set
    :return: dictionary of class: [list of features that belong to this class]
    """
    classes = {"categorical": [],
               "binary": [],
               "numerical": []
               }
    features = set(data.columns)
    num_cols = list(map(str.strip, data._get_numeric_data().columns))[:-1]
    classes["categorical"] = sorted(features - set(num_cols))
    indexes = []
    for feature in data[num_cols]:
        values = [x != 1 and x != 0 and not pd.isnull(x) for x in data[feature]]
        if any(values):
            indexes.append(feature)
    classes["numerical"] = indexes
    classes["binary"] = sorted(set(num_cols) - set(indexes))
    return classes
