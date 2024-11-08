import numpy as np
import os
import pandas as pd
import pickle
# import openml
from sklearn.metrics import make_scorer
from sklearn.ensemble import IsolationForest

def _ndcg(y1, y2, p=None):
    y1 = np.array(y1)
    y2 = np.array(y2)
    sorted_ind1 = np.argsort(-y1)
    sorted_ind2 = np.argsort(-y2)
    new_p = sum(y1 > 0)
    if p is not None:
        new_p = min(p, new_p)
    idcg = np.sum((y1[sorted_ind1[:new_p]])/np.log2(np.arange(2, new_p+2)))
    ind = [k for k in sorted_ind2 if y1[k] > 0]
    dcg = np.sum(y1[ind[:new_p]]/np.log2(np.arange(2, new_p+2)))
    return dcg/idcg

def ndcg(y_true, y_pred, p=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(y_true.shape) <= 1:
        return _ndcg(y_true, y_pred, p)
    return np.array([_ndcg(y, y_pred[i], p) for i, y in enumerate(y_true)])

def ndcg_sim(y1, y2, p=None):
    return _ndcg(y1, y2, p) * _ndcg(y2, y1, p) #pylint: disable=arguments-out-of-order

def custom_sim(y1, y2, threshold=0.95):
    set1 = set([j for j, yj in enumerate(y1) if yj/max(y1) > threshold])
    set2 = set([j for j, yj in enumerate(y2) if yj/max(y2) > threshold])
    return len(set1.intersection(set2)) / len(set1.union(set2)) #pylint: disable=arguments-out-of-order

def load_meta_dataset(meta_features_file, scores_dir, algorithm, eval_metric):
    np.random.seed(0)
    mixed_meta_df = pd.read_csv(meta_features_file, index_col="id").drop_duplicates()
    # openml_df = openml.datasets.list_datasets(output_format="dataframe")
    # mixed_meta_df = mixed_meta_df.loc[[ind for ind in mixed_meta_df.index if openml_df.loc[ind, "version"]==1]]
    # indices = IsolationForest(random_state=0).fit_predict(mixed_meta_df.to_numpy())>0
    # mixed_meta_df = mixed_meta_df.iloc[indices]
    mixed_meta_df.index = mixed_meta_df.index.astype(str)
    # print("Number of meta features:", mixed_meta_df.shape[1])
    # print("Number of instances:", mixed_meta_df.shape[0])

    benchmark_results = {}
    if eval_metric in ["ari", "purity"]:
        benchmark_results_acc = {}
    for filename in os.listdir(scores_dir):
        data_id = filename.split('.')[0]
        benchmark_results[data_id] = {}
        if eval_metric in ["ari", "purity"]:
            benchmark_results_acc[data_id] = {}
        with open(os.path.join(scores_dir, filename), "rb") as f:
            result = pickle.load(f)
        for sim_pair in result:
            if eval_metric in result[sim_pair]:
                benchmark_results[data_id][sim_pair] = \
                    max([v["score"] for v in result[sim_pair][eval_metric] \
                        if eval_metric != "sil" or 0.05 <= v["params"]["gamma" if algorithm=="kprototypes" else "alpha"] <= (20 if algorithm=="kprototypes" else 0.95)])
            if eval_metric in ["ari", "purity"] and "acc" in result[sim_pair]:
                benchmark_results_acc[data_id][sim_pair] =max([v["score"] for v in result[sim_pair]["acc"]])
        
    benchmark_results = pd.DataFrame.from_dict(benchmark_results, orient='index')
    benchmark_results = benchmark_results.fillna(-1)
    indices = np.random.permutation(benchmark_results.shape[0])
    benchmark_results = benchmark_results.iloc[indices]
    # if eval_metric in ["ari", "purity"]:
    #     benchmark_results_acc = pd.DataFrame.from_dict(benchmark_results_acc, orient='index')
    #     benchmark_results_acc = benchmark_results_acc.fillna(-1)
    #     benchmark_results_acc = benchmark_results_acc.iloc[indices]
    #     max_ = benchmark_results_acc.max(axis=1)
    #     benchmark_results = benchmark_results[max_ >= 0.7]
    # if eval_metric == "acc":
    #     max_ = benchmark_results.max(axis=1)
    #     benchmark_results = benchmark_results[max_ >= 0.7]

    index = benchmark_results.index
    mixed_meta_df = mixed_meta_df.loc[[i for i in index if i in mixed_meta_df.index]]
    benchmark_results = benchmark_results[index.isin(mixed_meta_df.index)]
    return mixed_meta_df, benchmark_results

top1_scorer = make_scorer(lambda yt, yp: np.mean([y[y>-1][np.argmax(yp[i][y>-1])]/max(y) for i, y in enumerate(yt)]))
top1_func = lambda yt, yp: np.mean([y[y>-1][np.argmax(yp[i][y>-1])]/max(y) for i, y in enumerate(yt)])

if __name__=="__main__":
    BENCHMARK_RESULTS_DIR = "../meta_dataset_creation/data/benchmark_results/"
    META_FEATURES_FILE = "../meta_dataset_creation/data/meta_features/meta_features.csv"
    algorithm = "kprototypes"
    scores_dir = os.path.join(BENCHMARK_RESULTS_DIR, algorithm, "scores")
    mixed_meta_df, benchmark_results = load_meta_dataset(META_FEATURES_FILE, scores_dir, "acc")
    print(mixed_meta_df.head())
    print(benchmark_results.head())