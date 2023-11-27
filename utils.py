import numpy as np
import os
import pandas as pd
from scipy.io import arff
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import confusion_matrix, adjusted_rand_score as ari, silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import minmax_scale
import openml


def contain_na(frame):
    return frame.replace(to_replace=[b'?', b'unknown', '?', 'unknown'], value=np.nan).isna().any(axis=None)


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


def load_openml_data(data_id, data_type="mixed"):
    try:
        dataset = openml.datasets.get_dataset(
            data_id, download_data=True, download_qualities=False, download_features_meta_data=False)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute
        )
    except:
        print(f"Not able to load dataset with id {data_id}")
        return None

    X = handle_na(X)
    X = X.loc[:, (X != X.iloc[0]).any()]
    y = y[X.index]
    y = pd.Categorical(
        y).codes if y.dtype.name == 'category' else y.to_numpy()
    if data_type == "mixed":
        num_columns = X.select_dtypes(include=["number"]).columns.values
        cat_columns = [col for col in X.columns if col not in num_columns]
        if len(num_columns) == 0 or len(cat_columns) == 0:
            print(f"dataset with id {data_id} is suposed to be mixed but got \
{len(num_columns)} numeric attributes and {len(cat_columns)} \
categorical attributes after handling nan values"
                  )
            return None
        Xnum = X.loc[:, num_columns]
        Xnum = Xnum.to_numpy()
        Xcat = X.loc[:, cat_columns]
        for col in Xcat.columns:
            Xcat.loc[:, col] = pd.Categorical(Xcat.loc[:, col]).codes
        Xcat = Xcat.to_numpy(dtype=int)

        return {
            "id": data_id,
            "data_type": data_type,
            "numeric_attributes": num_columns,
            "categorical_attributes": cat_columns,
            "samples": X.index.values,
            "Xnum": Xnum,
            "Xcat": Xcat,
            "y": y,
        }
    elif data_type == "numeric":
        return {
            "id": data_id,
            "data_type": data_type,
            "attributes_names": X.columns.values,
            "samples": X.index.values,
            "X": X.to_numpy(),
            "y": y,
        }

    elif data_type == "categorical":
        for col in X.columns:
            X.loc[:, col] = pd.Categorical(X.loc[:, col]).codes
        X = X.to_numpy(dtype=int)
        return {
            "id": data_id,
            "data_type": data_type,
            "attributes_names": X.columns.values,
            "samples": X.index.values,
            "X": X.to_numpy(),
            "y": y,
        }
    else:
        raise Exception(f"Not handled data type ({data_type})")


def accuracy(labels, predicted_labels):
    cm = confusion_matrix(labels, predicted_labels)
    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)
    rows, cols = linear_assignment(_make_cost_m(cm))
    indexes = zip(rows, cols)
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    return np.trace(cm2) / np.sum(cm2)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    cm = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)


def get_score(labels, predicted_labels, eval_metric="acc"):
    if eval_metric == "acc":
        return accuracy(labels, predicted_labels)
    elif eval_metric == "ari":
        return ari(labels, predicted_labels)
    elif eval_metric == "purity":
        return purity_score(labels, predicted_labels)


def get_unsupervised_score(X, predicted_labels, eval_metric="sil", **kwargs):
    if eval_metric == "sil":
        return silhouette_score(X, predicted_labels, **kwargs)


EVAL_METRICS = ["acc", "ari", "purity"]

if __name__ == "__main__":
    ID = 29
    data = load_openml_data(ID)
    print(data["numeric_attributes"])
    print(data["categorical_attributes"])
    print(len(data["samples"]))
