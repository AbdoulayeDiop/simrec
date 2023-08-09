import numpy as np
import os
import pandas as pd
from scipy.io import arff
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import confusion_matrix, adjusted_rand_score as ari, silhouette_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import minmax_scale
from sklearn.datasets import fetch_openml

def contain_na(frame):
    return frame.replace(to_replace=[b'?', b'unknown', '?', 'unknown'], value=np.nan).isna().any(axis=None)

def handle_na(frame, drop_first=True):
    df = frame.copy()
    df = df.replace(to_replace=[b'?', b'unknown', '?', 'unknown'], value=np.nan)
    for column in df.columns:
        nan_portion = df[column].isna().sum()/df.shape[0]
        if nan_portion >= 0.5:
            df = df.drop(columns=column)
            print(f"column {column} dropped from data set because it contains more than {int(nan_portion*100)} % of nan values")
    droped_df = df.dropna()
    if droped_df.shape[0] > 0.8*df.shape[0] and drop_first:
        df = droped_df
    else:
        nums = df.select_dtypes(include=["number"]).columns
        filling_dict = {}
        for column in df.columns:
            if column in nums:
                filling_dict[column] = df[column].median(skipna=True)
            else:
                unique, count = np.unique(df[column][df[column].notna()], return_counts=True)
                most_represented = unique[np.argmax(count)]
                filling_dict[column] = most_represented
        df = df.fillna(filling_dict)
    assert(df.isna().sum().sum()==0)
    return df

def load_data_from_file(dataset_file, exclude_columns=None):
    # TODO : Handle non arff files. Example : csv files
    print(f"Reading file: {dataset_file}")
    data = arff.loadarff(dataset_file)
    df = pd.DataFrame(data[0])
    if exclude_columns is not None:
        df = df.drop(columns=exclude_columns)
    df = df.replace(to_replace=[b'?', b'unknown', '?', 'unknown'], value=np.nan)
    for column in df.columns:
        nan_portion = df[column].isna().sum()/df.shape[0]
        if nan_portion >= 0.5:
            df = df.drop(columns=column)
            print(f"column {column} dropped from data set {dataset_file.split('/')[-1]} because it contains {int(nan_portion*100)} % of nan values")
    droped_df = df.dropna()
    if droped_df.shape[0] > 0.8*df.shape[0]:
        df = droped_df
    else:
        filling_dict = {}
        for column in df.columns:
            if df.dtypes[column]==object:
                unique, count = np.unique(df[column][df[column].notna()], return_counts=True)
                most_represented = unique[np.argmax(count)]
                filling_dict[column] = most_represented
            else:
                filling_dict[column] = df[column].median(skipna=True)
        df = df.fillna(filling_dict)
    assert(df.isna().sum().sum()==0)
    new_df = df.drop(columns=["class"])
    categorical = [i for i, column in enumerate(new_df.columns) if new_df.dtypes[column]==object]
    numeric = [i for i, column in enumerate(new_df.columns) if new_df.dtypes[column]==float]
    types = {"numeric": numeric, "categorical": categorical}
    n_cat = []
    freq = []
    N = 0
    print(categorical)
    
    for i in categorical:
        name = new_df.columns[i]
        n_cat.append(len(new_df[name].unique())) 
        new_df[name] = pd.Categorical(new_df[name])
        new_df[name] = new_df[name].cat.codes
        freq.append(dict(zip(*np.unique(new_df[name], return_counts=True))))

    X = new_df.to_numpy()
    print(f"instances: {X.shape[0]}, dimension: {X.shape[1]}")
    Xnum = X[:,numeric]
    Xnum = minmax_scale(Xnum)
    X[:,numeric] = Xnum
    
    df["class"] = pd.Categorical(df["class"])
    print(dict(zip(df["class"].to_list(), df["class"].cat.codes)))

    df["class"] = df["class"].cat.codes
    y =  df["class"].tolist()
    N = X.shape[0]
    exclude_num = []
    if len(numeric) < 2:
        exclude_num.append("mahalanobis")
    if len(numeric) < 5:
        exclude_num.append("pearson")
        
    return {
        "X": X,
        "y": y,
        "types": types,
        "params": {
            "numeric": {
                "VI" : np.linalg.inv(np.cov(Xnum, rowvar=False)) if Xnum.shape[1] > 1 else 1/np.cov(Xnum, rowvar=False)
            },
            "categorical": {
                "N": N,
                "n_cat": n_cat,
                "f": freq,
                "U": X[:,categorical]
            }
        },
        "exclude":{
            "numeric": exclude_num,
            "categorical": ["co-oc"] if len(categorical) < 2 else []
        }
    }

def load_data(dataset_folder):
    datasets = {}
    for dataset_file in os.listdir(dataset_folder):
        name = dataset_file.split(".")[0].split("/")[-1]
        if name == "zoo":
            datasets[name] = load_data_from_file(os.path.join(dataset_folder, dataset_file), exclude_columns=["animal"])
        else:
            datasets[name] = load_data_from_file(os.path.join(dataset_folder, dataset_file))
    return datasets

def load_openml_data(data_id):
    try:
        data = fetch_openml(data_id=data_id, as_frame=True)
    except:
        raise(Exception(f"Can not fetch dataset with id {data_id} from open_ml"))
    df = data["frame"]
    df = handle_na(df)
    for column in df.columns:
        if len(df[column].unique())<2:
            df = df.drop(columns=column)
            print(f"column {column} is constant and has been dropped from data set {data_id}")
    categorical = list(df.select_dtypes(include=["category"]).columns)
    numeric = list(df.select_dtypes(include=["number"]).columns)
    Xcat_df = df.select_dtypes(include=["category"]).copy()
    for column in Xcat_df.columns:
        Xcat_df[column] = pd.Categorical(Xcat_df[column]).codes
    Xcat = Xcat_df.to_numpy()
    Xnum = df.select_dtypes(include=["number"]).to_numpy()
    Xnum = minmax_scale(Xnum)
    X = np.c_[Xnum, Xcat]
    y =  pd.Categorical(df[data["target_names"][0]]).codes
    types = {
        "numeric": np.arange(len(numeric)),
        "categorical": np.arange(len(numeric), len(numeric)+len(categorical))
    }
    return {
        "id": data_id,
        "name": data_id,
        "X": X,
        "y": y,
        "types": types
    }

def show_result(result, eval_metric="acc"):
    dict_ = {}
    for metric1 in result:
        dict_[metric1] = {}
        for metric2 in result[metric1]:
            if len(result[metric1][metric2]) > 0:
                dict_[metric1][metric2] = max([obj["scores"][eval_metric] for obj in result[metric1][metric2]])
    df = pd.DataFrame.from_dict(dict_, orient="index")
    print(df)

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
    if eval_metric=="acc":
        return accuracy(labels, predicted_labels)
    elif eval_metric=="ari":
        return ari(labels, predicted_labels)
    elif eval_metric=="purity":
        return purity_score(labels, predicted_labels)

def get_unsupervised_score(X, predicted_labels, eval_metric="sil", **kwargs):
    if eval_metric=="sil":
        return silhouette_score(X, predicted_labels, **kwargs)

EVAL_METRICS = ["acc", "ari", "purity"]