def process_feature_selection(data):
    # train will be imputed differently than test
    data['2'] = data['2'].apply(lambda x: int(x[:-1]) if type(x) == str else x)  # remove  "d" suffix cast as int
    data['12'] = data['12'].replace(['y', 'n'], [1, 0])  # change to binary
    data['18'] = data['18'].apply(
        lambda x: int(x[1:]) if type(x) == str else x)  # drop the leading "a" in column, cast as int

    data = data.drop(columns=['0', '15', '17'], axis=1)
    return data
