from helper.feature_classfier import classify_features
from outliers.remove_outlier_stddev import remove_outlier_stddev


def basic_process(data, train=False):
    data.drop(columns=['0', '19', '15'], axis=1, inplace=True)  # dropping features 0, 15 and 19
    feature_classes = classify_features(data)  # create a dictionary with feature classifications
    numeric_features = feature_classes.get("numerical")
    if train:  # remove outliers only for training
        rows = remove_outlier_stddev(data[numeric_features])
        data.drop(index=rows, axis=0, inplace=True)
    return data
