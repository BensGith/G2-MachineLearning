from helper.feature_classfier import classify_features
from outliers.remove_outlier_stddev import remove_outlier_stddev


def basic_process(data, train=False):
    data.drop(columns=['19','15','9'], axis=1, inplace=True)
    feature_classes = classify_features(data)
    numeric_features = feature_classes.get("numerical")
    if train:
        rows = remove_outlier_stddev(data[numeric_features])
        data.drop(index=rows, axis=0, inplace=True)
    return data


