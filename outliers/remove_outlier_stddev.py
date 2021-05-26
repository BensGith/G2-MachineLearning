
def remove_outlier_stddev(data):
    """
    find row index's containing outliers
    :param data: df
    :return: list of indexes
    """
    drop_rows = set()
    for col in data.columns:
        col_mean = data[col].mean()
        col_std = data[col].std()
        for i, x in enumerate(data[col]):
            if abs((x - col_mean) / col_std) > 3:
                drop_rows.add(i)
    return drop_rows
