from metrics import base_metrics
from sklearn.preprocessing import OneHotEncoder
from scipy.io.arff import loadarff
import pandas as pd
import numpy as np

def get_valid_similarity_measures(X, data_type="numeric"):
    l = []
    for metric in base_metrics.get_available_metrics(data_type=data_type):
        m = base_metrics.get_metric(metric)
        if m.is_valid_data(X):
            ok = True
            if metric=="mahalanobis":
                try:
                    m.fit(X)
                except:
                    ok = False
            if ok:
                l.append(metric)
    return l

def get_valid_similarity_pairs(Xnum, Xcat):
    enc = OneHotEncoder(handle_unknown='ignore')
    Xdummy = enc.fit_transform(Xcat).toarray()
    l = []
    for num_metric in get_valid_similarity_measures(Xnum, data_type="numeric"):
        for cat_metric in get_valid_similarity_measures(Xcat, data_type="categorical"):
            l.append(f"{num_metric}_{cat_metric}")
        for bin_metric in get_valid_similarity_measures(Xdummy, data_type="binary"):
            l.append(f"{num_metric}_{bin_metric}")
    return l

def load_arff_file(f):
    data, _ = loadarff(f)
    return pd.DataFrame(data)

def load_csv_file(f):
    return pd.DataFrame(data)

def load_data_file(f, file_extension):
    if file_extension==".arff":
        return load_arff_file(f)
    if file_extension==".csv":
        return load_csv_file(f)
    else:
        raise(Exception(f"Not known file extension {file_extension}"))

def handle_na(frame, drop_first=True, min_samples_after_drop=0.8, max_nan_per_attributes=0.5):
    df = frame.copy()
    df = df.replace(to_replace=[b'?', b'unknown',
                    '?', 'unknown'], value=np.nan)
    for column in df.columns:
        nan_portion = df[column].isna().sum()/df.shape[0]
        if nan_portion > max_nan_per_attributes:
            df = df.drop(columns=column)
            print(
                f"column {column} dropped from data set because it contains more than \
{int(max_nan_per_attributes*100)}% of nan values ({int(nan_portion*100)}%)")
    droped_df = df.dropna()
    if drop_first and droped_df.shape[0] > min_samples_after_drop*df.shape[0]:
        df = droped_df
    else:
        nums = df.select_dtypes(include=["number"]).columns
        filling_dict = {}
        for column in df.columns:
            if column in nums:
                filling_dict[column] = df[column].median(skipna=True)
            else:
                unique, count = np.unique(
                    df[column][df[column].notna()], return_counts=True)
                most_represented = unique[np.argmax(count)]
                filling_dict[column] = most_represented
        df = df.fillna(filling_dict)
    assert (df.isna().sum().sum() == 0)
    return df