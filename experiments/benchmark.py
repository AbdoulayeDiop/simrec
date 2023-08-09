import gc
import json
import os
import pickle
import sys
import tempfile
import time
import traceback

import numpy as np
import pandas as pd
from filelock import FileLock
from joblib import Parallel, delayed
from kmedoids import alternating, fasterpam
from kmodes.kprototypes import KPrototypes as KP
from sklearn.cluster import AgglomerativeClustering, spectral_clustering
from sklearn.preprocessing import OneHotEncoder, minmax_scale
from utils import get_score, get_unsupervised_score

sys.path.append("..")
import base_metrics

gamma_values = np.concatenate((np.linspace(0, 1, 6), np.arange(2, 10)))
alpha_values = np.linspace(0, 1, 101)

np.random.seed(0)

def run_ac(D, n_clusters, linkage):
    ac = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        affinity="precomputed"
    )
    clusters = None
    try:
        clusters = ac.fit_predict(D)
    except:
        print(
            f"Error : h{linkage}"
        )
        print(
            f"Traceback : {traceback.format_exc()}"
        )
    return clusters

def optimize_ac(Ds, y, n_clusters, eval_metric, linkage):
    Dnum, Dcat = Ds
    def black_box_function(alpha):
        D = (1 - alpha)*Dnum + alpha*Dcat
        D[np.abs(D)<1e-9] = 0
        clusters = run_ac(D, n_clusters, linkage)
        if clusters is None : return -1
        if eval_metric=="sil":
            try:
                return get_unsupervised_score(D, clusters, eval_metric=eval_metric, metric="precomputed")
            except:
                print(D)
                return -1
        return get_score(y, clusters, eval_metric=eval_metric)

    # optimizer = BayesianOptimization(
    #     f=black_box_function,
    #     pbounds={"alpha" : (0, 1)},
    #     random_state=1,
    #     verbose=0
    # )
    # optimizer.maximize(init_points=5, n_iter=50)
    # res = optimizer.max

    scores = [black_box_function(alpha) for alpha in alpha_values]
    pos_best = np.argmax(scores)
    res = {
        "params": {
            "alpha": alpha_values[pos_best]
        },
        "target": scores[pos_best]
    }
    return res["params"], res["target"]

def run_kmedoids(D, n_clusters, method, n_init=10):
    final_clusters = None
    best_score = np.inf
    for random_state in range(n_init):
        try:
            res = fasterpam(D, n_clusters, random_state=random_state, n_cpu=1) if method=="fasterpam" \
                else alternating(D, n_clusters, random_state=random_state)
            clusters = res.labels
            score = res.loss
            if score < best_score:
                final_clusters = [val for val in clusters]
        except:
            print(f"Error : k-medoids -> method={method} ; random_state={random_state}")
            print(
                f"Traceback : {traceback.format_exc()}"
            )
    return final_clusters

def optimize_kmedoids(Ds, y, n_clusters, eval_metric, method):
    Dnum, Dcat = Ds
    def black_box_function(alpha):
        D = (1 - alpha)*Dnum + alpha*Dcat
        clusters = run_kmedoids(D, n_clusters, method)
        if clusters is None : return -1
        if eval_metric=="sil":
            try:
                return get_unsupervised_score(D, clusters, eval_metric=eval_metric, metric="precomputed")
            except:
                print(D)
                return -1
        return get_score(y, clusters, eval_metric=eval_metric)

    # optimizer = BayesianOptimization(
    #     f=black_box_function,
    #     pbounds={"alpha" : (0, 1)},
    #     random_state=1,
    #     verbose=0,
    # )
    # optimizer.maximize(init_points=10, n_iter=100)
    # res = optimizer.max
    # return res["params"], res["target"]

    scores = [black_box_function(alpha) for alpha in alpha_values]
    pos_best = np.argmax(scores)
    res = {
        "params": {
            "alpha": alpha_values[pos_best]
        },
        "target": scores[pos_best]
    }
    return res["params"], res["target"]

def run_sc(D, n_clusters, sigma):
    clusters = None
    try:
        S = np.exp(-D**2/(2*sigma**2))
        S = (S + S.T)/2
        clusters = spectral_clustering(S, n_clusters=n_clusters, random_state=0)
    except:
        print(f"Error : spectral -> sigma={sigma}")
        # print("Unexpected error:", sys.exc_info()[0])
    return clusters


def optimize_sc(Ds, y, n_clusters, eval_metric):
    Dnum, Dcat = Ds
    def black_box_function(alpha, sigma):
        D = (1 - alpha)*Dnum + alpha*Dcat
        # D[np.abs(D)<1e-9] = 0
        clusters = run_sc(D, n_clusters, sigma)
        if clusters is None : return -1
        if eval_metric=="sil":
            try:
                return get_unsupervised_score(D, clusters, eval_metric=eval_metric, metric="precomputed")
            except:
                print(D)
                return -1
        return get_score(y, clusters, eval_metric=eval_metric)

    # optimizer = BayesianOptimization(
    #     f=black_box_function,
    #     pbounds={"alpha" : (0, 1), "sigma" : (1e-3, 10)},
    #     random_state=1,
    #     verbose=0
    # )
    # optimizer.maximize(init_points=50)
    # res = optimizer.max

    alpha_values = np.linspace(0, 1, 21)
    sigma_values = np.linspace(0.01, 0.15, 15)
    scores = [black_box_function(alpha, sigma) for alpha, sigma in zip(alpha_values, sigma_values)]
    pos_best = np.argmax(scores)
    res = {
        "params": {
            "alpha": alpha_values[pos_best],
            "sigma": sigma_values[pos_best]
        },
        "target": scores[pos_best]
    }
    return res["params"], res["target"]

def run_kprototypes(X, types, n_clusters, num_metric, cat_metric, gamma):
    # kp = KPrototypes(
    #     n_clusters=n_clusters,
    #     gamma=gamma, 
    #     num_metric=num_metric,
    #     cat_metric=cat_metric,
    #     random_state=0
    # )
    kp = KP(
        n_clusters=n_clusters,
        gamma=gamma, 
        num_dissim=num_metric.flex,
        cat_dissim=cat_metric.flex,
        random_state=0,
        n_init=5,
        init='random',
    )
    clusters = None
    try:
        kp = kp.fit(X, categorical=list(types["categorical"]))
        clusters = kp.labels_
    except:
        print()
        print(
            f"Error : KPrototypes {num_metric}, {cat_metric}"
        )
        print(
            f"Traceback : {traceback.format_exc()}"
        )
    return clusters

def optimize_kprototypes(X, y, types, n_clusters, num_metric, cat_metric, eval_metric):
    def black_box_function(gamma):
        clusters = run_kprototypes(X, types, n_clusters, num_metric, cat_metric, gamma)
        if clusters is None : return -1
        return get_score(y, clusters, eval_metric=eval_metric)

    # optimizer = BayesianOptimization(
    #     f=black_box_function,
    #     pbounds={"gamma" : (0, 20)},
    #     random_state=1,
    #     verbose=0
    # )
    # optimizer.maximize(init_points=10)
    # res = optimizer.max
    # return res["params"], res["target"]
    scores = [black_box_function(gamma) for gamma in gamma_values]
    pos_best = np.argmax(scores)
    res = {
        "params": {
            "gamma": gamma_values[pos_best]
        },
        "target": scores[pos_best]
    }
    return res["params"], res["target"]

def optimize_with_pairwise_dist(algorithm, Ds, y, n_clusters, eval_metric):
    if algorithm == "haverage":
        return optimize_ac(Ds, y, n_clusters, eval_metric, linkage="average")
    if algorithm == "fasterpam":
        return optimize_kmedoids(Ds, y, n_clusters, eval_metric, method="fasterpam")
    if algorithm == "sfkm":
        return optimize_kmedoids(Ds, y, n_clusters, eval_metric, method="alternate")
    if algorithm == "spectral":
        return optimize_sc(Ds, y, n_clusters, eval_metric)

def optimize_with_data(algorithm, X, y, types, n_clusters, num_metric, cat_metric, eval_metric):
    if algorithm == "kprototypes":
        return optimize_kprototypes(X, y, types, n_clusters, num_metric, cat_metric, eval_metric)

def benchmark(id_, data, results_file, algorithm="ac", eval_metric = "acc"):
    start = time.time()
    Xnum, Xcat, y = data
    n_clusters = len(set(y))
    enc = OneHotEncoder(handle_unknown='ignore')
    Xdummy = enc.fit_transform(Xcat).toarray()

    result = {}

    def compute_dissim(X, metric):
        m = base_metrics.get_metric(metric)
        D = None
        if m.is_valid_data(X):
            try:
                D = m.fit(X).pairwise(X, n_jobs=-1)
                if np.isnan(D).any():
                    print("Warning: Distance matrix contain Nan values for metric:", metric)
                    return None
                if np.isinf(D).any():
                    print("Warning: Distance matrix contain infinite values for metric:", metric)
                    return None
                np.fill_diagonal(D, 0)
            except:
                print(
                    f"Error: While computing pairwise distance | dataset: {id_}, metric: {metric}"
                )
                print(
                    f"Traceback: {traceback.format_exc()}"
                )
        return D

    def run_with_pairwise_dist(Dnum, Dcat):
        res = {}
        weights, score = optimize_with_pairwise_dist(algorithm, [Dnum, Dcat], y, n_clusters, eval_metric)
        res["params"] = weights
        res["params"]["n_clusters"] = n_clusters
        res["score"] = score
        return res

    def run_with_data(X, types, num_metric, cat_metric):
        res = {}
        weights, score = optimize_with_data(algorithm, X, y, types, n_clusters, num_metric, cat_metric, eval_metric)
        res["params"] = weights
        res["params"]["n_clusters"] = n_clusters
        res["score"] = score
        return res

    if algorithm in ["kprototypes"]:
        X = np.c_[Xnum, Xcat]
        types = {
            "numeric": np.arange(Xnum.shape[1]),
            "categorical": np.arange(Xcat.shape[1]) + Xnum.shape[1]
        }
        all_fitted_metric = {}
        for num_metric in base_metrics.get_available_metrics(data_type="numeric"):
            if base_metrics.get_metric(num_metric).is_valid_data(Xnum):
                if num_metric not in all_fitted_metric:
                    fitted_num_metric = base_metrics.get_metric(num_metric).fit(Xnum)
                    all_fitted_metric[num_metric] = fitted_num_metric
                else:
                    fitted_num_metric = all_fitted_metric[num_metric]
                for cat_metric in base_metrics.get_available_metrics(data_type="categorical"):
                    if base_metrics.get_metric(cat_metric).is_valid_data(Xcat):
                        if cat_metric not in all_fitted_metric:
                            fitted_cat_metric = base_metrics.get_metric(cat_metric).fit(Xcat)
                            all_fitted_metric[cat_metric] = fitted_cat_metric
                        else:
                            fitted_cat_metric = all_fitted_metric[cat_metric]
                        result[f"{num_metric}_{cat_metric}"] = run_with_data(X, types, fitted_num_metric, fitted_cat_metric)
        X = np.c_[Xnum, Xdummy]
        types = {
            "numeric": np.arange(Xnum.shape[1]),
            "categorical": np.arange(Xdummy.shape[1]) + Xnum.shape[1]
        }
        for num_metric in base_metrics.get_available_metrics(data_type="numeric"):
            if base_metrics.get_metric(num_metric).is_valid_data(Xnum):
                fitted_num_metric = all_fitted_metric[num_metric]
                for bin_metric in base_metrics.get_available_metrics(data_type="binary"):
                    if base_metrics.get_metric(bin_metric).is_valid_data(Xdummy):
                        if bin_metric not in all_fitted_metric:
                            fitted_bin_metric = base_metrics.get_metric(cat_metric).fit(Xdummy)
                            all_fitted_metric[bin_metric] = fitted_bin_metric
                        else:
                            fitted_bin_metric = all_fitted_metric[bin_metric]
                        result[f"{num_metric}_{bin_metric}"] = run_with_data(X, types, fitted_num_metric, fitted_bin_metric)
    else:
        tempdir = tempfile.TemporaryDirectory(dir="/homedir/adiop/tmp/")
        for num_metric in base_metrics.get_available_metrics(data_type="numeric"):
            # print(f"{id_}, {num_metric}", flush=True)
            Dnum = compute_dissim(Xnum, num_metric)
            if Dnum is not None:
                computed = os.listdir(tempdir.name)
                for cat_metric in base_metrics.get_available_metrics(data_type="categorical"):
                    filename = os.path.join(tempdir.name, cat_metric)
                    if cat_metric in computed:
                        with open(filename, "rb") as f:
                            Dcat = pickle.load(f)
                    else:
                        Dcat = compute_dissim(Xcat, cat_metric)
                        with open(filename, "wb") as f:
                            pickle.dump(Dcat, f)
                    if Dcat is not None:
                        result[f"{num_metric}_{cat_metric}"] = run_with_pairwise_dist(Dnum, Dcat)
                    # Dcat = compute_dissim(Xcat, cat_metric)
                    # if Dcat is not None:
                    #     result[f"{num_metric}_{cat_metric}"] = run_with_pairwise_dist(Dnum, Dcat)
                for bin_metric in base_metrics.get_available_metrics(data_type="binary"):
                    filename = os.path.join(tempdir.name, bin_metric)
                    if bin_metric in computed:
                        with open(filename, "rb") as f:
                            Dcat = pickle.load(f)
                    else:
                        Dcat = compute_dissim(Xdummy, bin_metric)
                        with open(filename, "wb") as f:
                            pickle.dump(Dcat, f)
                            f.flush()
                    if Dcat is not None:
                        result[f"{num_metric}_{bin_metric}"] = run_with_pairwise_dist(Dnum, Dcat)
                    # Dcat = compute_dissim(Xdummy, bin_metric)
                    # if Dcat is not None:
                    #     result[f"{num_metric}_{bin_metric}"] = run_with_pairwise_dist(Dnum, Dcat)

    end = time.time()
    duration = None
    gc.collect()
    # print(result, flush=True)
    if len(result) > 0:
        duration = end - start
        time_file = os.path.join(os.path.dirname(results_file), f"benchmark_duration_{args.algorithm}_{args.evalmetric}_augmented_grid.json")
        with FileLock(results_file + ".lock"):
            results = {}
            if os.path.isfile(results_file):
                with open(results_file, "r", encoding="utf-8") as f:
                    results = json.load(f)
            results[id_] = result
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
                f.flush()
            del results

            times = {}
            if os.path.isfile(time_file):
                with open(time_file, "r", encoding="utf-8") as f:
                    times = json.load(f)
            times[id_] = duration
            with open(time_file, "w", encoding="utf-8") as f:
                json.dump(times, f, indent=4, ensure_ascii=False)
                f.flush()
            del times
        # with open(os.path.join(result_dir, id_), "w", encoding="utf-8") as f:
        #     json.dump(result, f, indent=4, ensure_ascii=False)
        #     f.flush()
    return duration


def check_constant_columns(X):
    l = []
    for j in range(X.shape[1]):
        if len(np.unique(X[:,j])) > 1:
            l.append(j)
    return l if len(l)>0 else None

def isconstant(y):
    return len(np.unique(y)) == 1

def preprocess(Xnum, Xcat, y):
    # remove constant columns
    lnum, lcat, yisconstant = check_constant_columns(Xnum), check_constant_columns(Xcat), isconstant(y)
    if lnum is None or lcat is None or yisconstant : return None
    Xnum, Xcat, y = Xnum[:, lnum], Xcat[:, lcat], y

    # remove outliers
    to_keep = set(range(Xnum.shape[0]))
    for j in range(Xnum.shape[1]):
        q1, q3 = np.quantile(Xnum[:,j], 0.25), np.quantile(Xnum[:,j], 0.75)
        iqr = q3 - q1
        cutoff = 3*iqr
        for i in range(Xnum.shape[0]):
            if i in to_keep:
                if Xnum[i, j] < q1 - cutoff or Xnum[i, j] > q3 + cutoff:
                    to_keep.remove(i)
    if len(to_keep) == 0: return None
    samples = list(to_keep)
    Xnum, Xcat, y = Xnum[samples], Xcat[samples], y[samples]
    Xnum = minmax_scale(Xnum)

    # Check for constant columns again
    lnum2, lcat2, yisconstant = check_constant_columns(Xnum), check_constant_columns(Xcat), isconstant(y)
    if lnum2 is None or lcat2 is None or yisconstant : return None

    # store information about keeped elements
    keeped_elements = {
        "samples": samples,
        "numfeats": np.array(lnum, dtype=int)[lnum2],
        "catfeats": np.array(lcat, dtype=int)[lcat2]
    }
    return Xnum[:, lnum2], Xcat[:, lcat2], y, keeped_elements

if __name__=="__main__":
    import argparse
    import os

    import openml
    parser = argparse.ArgumentParser(description='Benchmark data set data sets')
    parser.add_argument("-i", "--infosfile", help="Path to the file containing identification of data sets to benchmarch")
    parser.add_argument("-s", "--selected", help="Path to the file containing ids of selected data sets", default=None)
    parser.add_argument("-o", "--outdir", help="The directory where results will be saved")
    parser.add_argument("-a", "--algorithm", help="The clustering algorithm that will be used")
    parser.add_argument("-e", "--evalmetric", help="The clustering evalution metric")
    parser.add_argument("-j", "--jobs", help="The number of concurent workers", default=-1)
    parser.add_argument("-r", "--replace", help="Update or replace result if exist", action="store_true")
    args = parser.parse_args()

    print(f"outdir: {args.outdir} - algorithm: {args.algorithm} - eval metric: {args.evalmetric}")

    results = {}
    results_file = os.path.join(args.outdir, f"benchmark_results_{args.algorithm}_{args.evalmetric}_augmented_grid.json")
    # tmp_results_dir = tempfile.TemporaryDirectory(dir="/homedir/adiop/tmp")
    if os.path.isfile(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)

    checked_data = None
    if args.selected is not None and os.path.isfile(args.selected):
        checked_data = pd.read_csv(args.selected, sep=" ", index_col="id")
    datasets_info = {}
    infos_file = args.infosfile
    if os.path.isfile(infos_file):
        with open(infos_file, "rb") as f:
            datasets_info = pickle.load(f)
            if checked_data is not None:
                for id_ in list(datasets_info.keys()):
                    if int(id_) not in checked_data.index:
                        datasets_info.pop(id_)

    print(len(results))
    print(len(datasets_info))
    list_ids = []
    i = 0
    for id_ in datasets_info:
        for k in datasets_info[id_]:
            if k <= 10 and f"{id_}_{k}" not in results:
                list_ids.append(id_)
                break

    # print(set([k.split("_")[0] for k in results]))
    n_datasets = len(list_ids)
    print(f"{n_datasets}/{len(datasets_info)} data sets to load ({len(datasets_info) - n_datasets} already handled)")

    print()
    print("LOADING...")
    datasets = {}
    n_loaded = 0
    for i, id_ in enumerate(list_ids):
        print(f"id:{id_}, {i+1}/{len(list_ids)}")
        try:
            dataset = openml.datasets.get_dataset(id_)
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=dataset.default_target_attribute
            )
            for k in datasets_info[id_]:
                if f"{id_}_{k}" not in results:
                    Xnum = X.loc[datasets_info[id_][k]["samples"], datasets_info[id_][k]["num_columns"]]
                    Xcat = X.loc[datasets_info[id_][k]["samples"], datasets_info[id_][k]["cat_columns"]]
                    for col in Xcat.columns:
                        Xcat.loc[:, col] = pd.Categorical(Xcat.loc[:, col]).codes
                    Xnum = Xnum.loc[:, (Xnum != Xnum.iloc[0]).any()]
                    Xnum = Xnum.to_numpy()
                    # Xnum = minmax_scale(Xnum)
                    Xcat = Xcat.to_numpy(dtype=int)
                    
                    new_y = y.loc[datasets_info[id_][k]["samples"]]
                    new_y = pd.Categorical(new_y).codes if new_y.dtype.name == 'category' else new_y.to_numpy()
                    # print(f"samples({new_X.shape[0]}), num({Xnum.shape[1]}), cat({Xcat.shape[1]})...")
                    # ret = preprocess(Xnum, Xcat, new_y)
                    # if ret is not None:
                    # Xnum, Xcat, new_y, _ = ret
                    datasets[f"{id_}_{k}"] = [Xnum, Xcat, new_y]
            n_loaded += 1
        except:
            print(f"Not able to load data set with id {id_}")

    print("END LOADING!")

    print()
    n_augmented = sum([len(datasets_info[id_]) for id_ in set([k.split("_")[0] for k in datasets])])
    print(f"{n_loaded}/{n_datasets} data sets loaded (-> {n_augmented} after augmentation)")
    print(f"{n_augmented - len(datasets)} already handled -> {len(datasets)} remaining")
    
    print()
    print("BENCHMARKING...")
    list_ret = Parallel(n_jobs=args.jobs, verbose=60)(
        delayed(benchmark)(augmented_id, data, results_file, algorithm=args.algorithm, eval_metric=args.evalmetric) \
        for augmented_id, data in datasets.items()
    )
    print(f"END BENCHMARKING!")