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

def load_meta_dataset(meta_features_file, benchmark_results_dir, verbose=1):
    np.random.seed(0)
    if verbose > 0:
        print("Loading the meta-dataset")
        print("1. Loading the meta-features...", end="")
    meta_features_df = pd.read_csv(meta_features_file, index_col="id").drop_duplicates()
    # openml_df = openml.datasets.list_datasets(output_format="dataframe")
    # mixed_meta_df = mixed_meta_df.loc[[ind for ind in mixed_meta_df.index if openml_df.loc[ind, "version"]==1]]
    # indices = IsolationForest(random_state=0).fit_predict(mixed_meta_df.to_numpy())>0
    # mixed_meta_df = mixed_meta_df.iloc[indices]
    meta_features_df.index = meta_features_df.index.astype(str)
    # print("Number of meta features:", mixed_meta_df.shape[1])
    # print("Number of instances:", mixed_meta_df.shape[0])
    if verbose > 0: print("DONE")

    if verbose > 0: print("2. Loading the clustering benchmark results...", end="")
    benchmark_results = {}
    common_dids = None
    clustering_algorithms = os.listdir(benchmark_results_dir)
    for algorithm in clustering_algorithms:
        benchmark_results[algorithm] = {}
        folder_path = os.path.join(benchmark_results_dir, algorithm)
        scores_dir = os.path.join(folder_path, "scores/")
        for filename in os.listdir(scores_dir):
            data_id = filename.split('.')[0]
            with open(os.path.join(scores_dir, filename), "rb") as f:
                result = pickle.load(f)
            for sim_pair in result:
                for cvi in result[sim_pair]:
                    if cvi not in benchmark_results[algorithm]:
                        benchmark_results[algorithm][cvi] = {}
                    if data_id not in benchmark_results[algorithm][cvi]:
                        benchmark_results[algorithm][cvi][data_id] = {}
                    benchmark_results[algorithm][cvi][data_id][sim_pair] = max([v["score"] for v in result[sim_pair][cvi] \
                            if cvi != "sil" or 0.05 <= v["params"]["gamma" if algorithm=="kprototypes" else ("alpha" if "alpha" in v["params"] else "w")] <= (20 if algorithm=="kprototypes" else 0.95)])
                    # if cvi == "sil":
                    #     print(sim_pair, sorted(result[sim_pair][cvi], reverse=True, key= lambda v: v["score"])[0])
            
        for cvi in benchmark_results[algorithm]:
            benchmark_results[algorithm][cvi] = \
                pd.DataFrame.from_dict(benchmark_results[algorithm][cvi], orient='index')
            benchmark_results[algorithm][cvi] = \
                benchmark_results[algorithm][cvi].fillna(-1)
            
        if len(benchmark_results[algorithm]) > 0:
            if common_dids is None:
                common_dids = set(benchmark_results[algorithm]["acc"].index.to_list())
            else:
                common_dids = common_dids.intersection(set(benchmark_results[algorithm]["acc"].index.to_list()))

    common_dids = np.random.permutation(list(common_dids))
    common_dids = np.array([did for did in common_dids if did in meta_features_df.index])
    meta_features_df = meta_features_df.loc[common_dids]
    for algorithm in benchmark_results:
        for cvi in benchmark_results[algorithm]:
            benchmark_results[algorithm][cvi] = benchmark_results[algorithm][cvi].loc[common_dids]
    if verbose > 0: print("DONE")

    if verbose > 0: print("3. Filtering for external CVIs...", end="")
    df = pd.concat(
        [benchmark_results["kprototypes"]["acc"].max(axis=1) >= 0.75, 
        benchmark_results["fasterpam"]["acc"].max(axis=1) >= 0.75,
        benchmark_results["haverage"]["acc"].max(axis=1) >= 0.75],
        axis=1
    )
    filtered = df.any(axis=1)
    for algorithm in benchmark_results:
        for cvi in ["acc", "ari", "purity"]:
            benchmark_results[algorithm][cvi] = benchmark_results[algorithm][cvi][filtered]
    if verbose > 0: 
        print("DONE")
        print("# of datasets:", meta_features_df.shape[0])
        print("# of datasets for external CVIs:", benchmark_results["kprototypes"]["acc"].shape[0])
        print("# of meta-features:", meta_features_df.shape[1])
        print("# of pairs of similarity measures:", benchmark_results["kprototypes"]["acc"].shape[1])

    return meta_features_df, benchmark_results

def lower_bound(cvi):
    if cvi.lower() == "sil": return -1
    elif cvi.lower() == "ari": return -0.5
    elif cvi.lower() == "acc": return 0
    else:
        raise(Exception(f"Not nown CVI {cvi}, avilable CVIs are 'sil', 'ari', and 'acc'"))

def top_r(yt, yp, cvi, r=1):
    lower_bound_cvi = lower_bound(cvi)
    def _top_r(yt_i, yp_i):
        top_r_similarity_pairs = np.argsort(-yp_i[yt_i > -1])[:r]
        score = max(yt_i[yt_i > -1][top_r_similarity_pairs])
        return (score - lower_bound_cvi) / (max(yt_i) - lower_bound_cvi)
    return np.mean([_top_r(yt_i, yp[i]) for i, yt_i in enumerate(yt)])

if __name__=="__main__":
    BENCHMARK_RESULTS_DIR = "../meta_dataset_creation/data/benchmark_results/"
    META_FEATURES_FILE = "../meta_dataset_creation/data/meta_features/meta_features.csv"
    meta_features_df, benchmark_results = load_meta_dataset(META_FEATURES_FILE, BENCHMARK_RESULTS_DIR)
    # print(meta_features_df.head())
    # print(benchmark_results)
