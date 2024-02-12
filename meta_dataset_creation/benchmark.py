"""
Clustering Benchmark
"""

# pylint: disable=bare-except

import gc
import json
import os
import pickle
import sys
import tempfile
import time
import traceback

import numpy as np
from filelock import FileLock
from joblib import Parallel, delayed
from sklearn.preprocessing import OneHotEncoder, minmax_scale

sys.path.append("..")
from meta_dataset_creation.utils import (EXTERNAL_EVAL_METRICS, INTERNAL_EVAL_METRICS, get_score,
                   get_unsupervised_score)
from metrics import base_metrics
from clustering_algorithms import haverage, kmedoids, kprototypes

# from hedclus.hedclus.cluster import KPrototypes
# from kmodes.kprototypes import KPrototypes as KP

gamma_values = np.concatenate((np.linspace(0, 0.1, 10), np.linspace(
    0.1, 1, 10), np.arange(2, 10), np.arange(10, 101, 10)))
w_values = np.linspace(0, 1, 51)

np.random.seed(0)

def eval_haverage(Dnum, Dcat, y, n_clusters):
    clusters = [haverage(Dnum, Dcat, w, n_clusters) for w in w_values]
    result = {}
    for eval_metric in EXTERNAL_EVAL_METRICS:
        result[eval_metric] = []
        for w, yp in zip(w_values, clusters):
            score = - \
                1 if yp is None else get_score(y, yp, eval_metric=eval_metric)
            result[eval_metric].append({
                "params": {
                    "alpha": w,
                    "n_clusters": n_clusters
                },
                "score": score
            })

    for eval_metric in INTERNAL_EVAL_METRICS:
        result[eval_metric] = []
        for w, yp in zip(w_values, clusters):
            score = -1 if yp is None else \
                get_unsupervised_score((1-w)*Dnum + w*Dcat, yp,
                                       eval_metric=eval_metric, metric="precomputed")
            result[eval_metric].append({
                "params": {
                    "alpha": w,
                    "n_clusters": n_clusters
                },
                "score": score
            })
    return result
    # mem = (Dnum.itemsize*Dnum.size + Dcat.itemsize*Dcat.size)*2
    # n_jobs = min(64, int(40e9/mem))
    # clusters = Parallel(n_jobs=n_jobs)(
    #     delayed(haverage)(
    #         Dnum, Dcat, w, n_clusters
    #     ) for w in w_values
    # )

    # result = {}
    # for eval_metric in EXTERNAL_EVAL_METRICS:
    #     result[eval_metric] = []
    #     scores = Parallel(n_jobs=51)(
    #         delayed(get_score)(y, yp, eval_metric=eval_metric)
    #         for yp in clusters if yp is not None
    #     )
    #     i = 0
    #     for w, yp in zip(w_values, clusters):
    #         if yp is None:
    #             score = -1
    #         else:
    #             score = scores[i]
    #             i += 1
    #         result[eval_metric].append({
    #             "params": {
    #                 "w": w,
    #                 "n_clusters": n_clusters
    #             },
    #             "score": score
    #         })

    # for eval_metric in INTERNAL_EVAL_METRICS:
    #     result[eval_metric] = []
    #     scores = Parallel(n_jobs=51)(
    #         delayed(get_unsupervised_score)(
    #             (1-w)*Dnum + w*Dcat, yp,
    #             eval_metric=eval_metric, metric="precomputed"
    #         ) for w, yp in zip(w_values, clusters) if yp is not None
    #     )
    #     i = 0
    #     for w, yp in zip(w_values, clusters):
    #         if yp is None:
    #             score = -1
    #         else:
    #             score = scores[i]
    #             i += 1
    #         result[eval_metric].append({
    #             "params": {
    #                 "w": w,
    #                 "n_clusters": n_clusters
    #             },
    #             "score": score
    #         })
    # return result


def eval_kmedoids(Dnum, Dcat, y, n_clusters, method):
    clusters = [kmedoids(Dnum, Dcat, w, n_clusters, method=method)
                for w in w_values]
    result = {}
    for eval_metric in EXTERNAL_EVAL_METRICS:
        result[eval_metric] = []
        for w, yp in zip(w_values, clusters):
            score = - \
                1 if yp is None else get_score(y, yp, eval_metric=eval_metric)
            result[eval_metric].append({
                "params": {
                    "alpha": w,
                    "n_clusters": n_clusters
                },
                "score": score
            })

    for eval_metric in INTERNAL_EVAL_METRICS:
        result[eval_metric] = []
        for w, yp in zip(w_values, clusters):
            score = -1 if yp is None else \
                get_unsupervised_score((1-w)*Dnum + w*Dcat, yp,
                                       eval_metric=eval_metric, metric="precomputed")
            result[eval_metric].append({
                "params": {
                    "alpha": w,
                    "n_clusters": n_clusters
                },
                "score": score
            })
    return result


def eval_kprototypes(Xnum, Xcat, y, n_clusters, num_metric, cat_metric):
    clusters = [kprototypes(Xnum, Xcat, n_clusters, num_metric,
                            cat_metric, gamma) for gamma in gamma_values]
    result = {}
    for eval_metric in EXTERNAL_EVAL_METRICS:
        result[eval_metric] = []
        for gamma, yp in zip(gamma_values, clusters):
            score = -1 if yp is None else \
                get_score(y, yp, eval_metric=eval_metric)
            result[eval_metric].append({
                "params": {
                    "gamma": gamma,
                    "n_clusters": n_clusters
                },
                "score": score
            })

    Dnum = num_metric.pairwise(Xnum)
    Dcat = None
    if np.isnan(Dnum).any():
        print("Warning: Distance matrix contain Nan values for metric:", num_metric)
        Dnum = None
    elif np.isinf(Dnum).any():
        print("Warning: Distance matrix contain infinite values for metric:", num_metric)
        Dnum = None
    if Dnum is not None:
        Dcat = cat_metric.pairwise(Xcat)
        if np.isnan(Dcat).any():
            print("Warning: Distance matrix contain Nan values for metric:", cat_metric)
            Dcat = None
        elif np.isinf(Dcat).any():
            print("Warning: Distance matrix contain infinite values for metric:", cat_metric)
            Dcat = None

    if Dcat is not None:
        for eval_metric in INTERNAL_EVAL_METRICS:
            result[eval_metric] = []
            for gamma, yp in zip(gamma_values, clusters):
                D = Dnum + gamma*Dcat
                np.fill_diagonal(D, 0)
                score = -1 if yp is None else \
                    get_unsupervised_score(D, yp, eval_metric=eval_metric, metric="precomputed")
                result[eval_metric].append({
                    "params": {
                        "gamma": gamma,
                        "n_clusters": n_clusters
                    },
                    "score": score
                })
    return result

    # mem = (Xnum.itemsize*Xnum.size + Xcat.itemsize*Xcat.size)*2
    # n_jobs = min(128, int(16e9/mem))
    # clusters = Parallel(n_jobs=n_jobs)(
    #     delayed(kprototypes)(
    #         Xnum, Xcat, n_clusters,
    #         num_metric, cat_metric, gamma
    #     ) for gamma in gamma_values
    # )

    # result = {}
    # for eval_metric in EXTERNAL_EVAL_METRICS:
    #     result[eval_metric] = []
    #     scores = Parallel(n_jobs=-1)(
    #         delayed(get_score)(y, yp, eval_metric=eval_metric)
    #         for yp in clusters if yp is not None
    #     )
    #     i = 0
    #     for gamma, yp in zip(gamma_values, clusters):
    #         if yp is None:
    #             score = -1
    #         else:
    #             score = scores[i]
    #             i += 1
    #         result[eval_metric].append({
    #             "params": {
    #                 "gamma": gamma,
    #                 "n_clusters": n_clusters
    #             },
    #             "score": score
    #         })

    # Dnum = num_metric.pairwise(Xnum)
    # Dcat = None
    # if np.isnan(Dnum).any():
    #     print("Warning: Distance matrix contain Nan values for metric:", num_metric)
    #     Dnum = None
    # elif np.isinf(Dnum).any():
    #     print("Warning: Distance matrix contain infinite values for metric:", num_metric)
    #     Dnum = None
    # if Dnum is not None:
    #     Dcat = cat_metric.pairwise(Xcat)
    #     if np.isnan(Dcat).any():
    #         print("Warning: Distance matrix contain Nan values for metric:", cat_metric)
    #         Dcat = None
    #     elif np.isinf(Dcat).any():
    #         print(
    #             "Warning: Distance matrix contain infinite values for metric:", cat_metric)
    #         Dcat = None

    # if Dcat is not None:
    #     for eval_metric in INTERNAL_EVAL_METRICS:
    #         result[eval_metric] = []
    #         scores = Parallel(n_jobs=-1)(
    #             delayed(get_unsupervised_score)(
    #                 Dnum + gamma*Dcat, yp,
    #                 eval_metric=eval_metric, metric="precomputed"
    #             ) for gamma, yp in zip(gamma_values, clusters) if yp is not None
    #         )
    #         i = 0
    #         for gamma, yp in zip(gamma_values, clusters):
    #             if yp is None:
    #                 score = -1
    #             else:
    #                 score = scores[i]
    #                 i += 1
    #             result[eval_metric].append({
    #                 "params": {
    #                     "gamma": gamma,
    #                     "n_clusters": n_clusters
    #                 },
    #                 "score": score
    #             })
    # return result


def eval_with_pairwise_dist(algorithm, Dnum, Dcat, y, n_clusters):
    if algorithm == "haverage":
        return eval_haverage(Dnum, Dcat, y, n_clusters)
    if algorithm == "fasterpam":
        return eval_kmedoids(Dnum, Dcat, y, n_clusters, "fasterpam")
    if algorithm == "sfkm":
        return eval_kmedoids(Dnum, Dcat, y, n_clusters, "alternate")


def eval_with_data(algorithm, Xnum, Xcat, y, n_clusters, num_metric, cat_metric):
    if algorithm == "kprototypes":
        return eval_kprototypes(Xnum, Xcat, y, n_clusters, num_metric, cat_metric)


def benchmark(data, outputdir, algorithm="haverage"):
    Xnum, Xcat, y = minmax_scale(data["Xnum"]), data["Xcat"], data["y"]
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
                    print(
                        "Warning: Distance matrix contain Nan values for metric:", metric)
                    return None
                if np.isinf(D).any():
                    print(
                        "Warning: Distance matrix contain infinite values for metric:", metric)
                    return None
                np.fill_diagonal(D, 0)
            except:
                print(
                    f"Error: While computing pairwise distance | dataset: {data['id']}, metric: {metric}"
                )
                print(
                    f"Traceback: {traceback.format_exc()}"
                )
        return D

    start = time.time()
    if algorithm in ["kprototypes"]:
        all_fitted_num_metric = {}
        all_fitted_cat_metric = {}
        all_fitted_bin_metric = {}
        for num_metric in base_metrics.get_available_metrics(data_type="numeric"):
            if base_metrics.get_metric(num_metric).is_valid_data(Xnum):
                try:
                    fitted_num_metric = base_metrics.get_metric(
                        num_metric).fit(Xnum)
                except:
                    fitted_num_metric = None
                    print(
                        f"Error: While fitting metric: {num_metric} | dataset: {data['id']}")
                    print(f"Traceback: {traceback.format_exc()}")
                all_fitted_num_metric[num_metric] = fitted_num_metric

        for cat_metric in base_metrics.get_available_metrics(data_type="categorical"):
            if base_metrics.get_metric(cat_metric).is_valid_data(Xcat):
                try:
                    fitted_cat_metric = base_metrics.get_metric(
                        cat_metric).fit(Xcat)
                except:
                    fitted_cat_metric = None
                    print(
                        f"Error: While fitting metric: {cat_metric} | dataset: {data['id']}")
                    print(f"Traceback: {traceback.format_exc()}")
                all_fitted_cat_metric[cat_metric] = fitted_cat_metric

        for bin_metric in base_metrics.get_available_metrics(data_type="binary"):
            if base_metrics.get_metric(bin_metric).is_valid_data(Xdummy):
                try:
                    fitted_bin_metric = base_metrics.get_metric(
                        bin_metric).fit(Xdummy)
                except:
                    fitted_bin_metric = None
                    print(
                        f"Error: While fitting metric: {bin_metric} | dataset: {data['id']}")
                    print(f"Traceback: {traceback.format_exc()}")
                all_fitted_bin_metric[bin_metric] = fitted_bin_metric

        for num_metric, fitted_num_metric in all_fitted_num_metric.items():
            if fitted_num_metric is not None:
                for cat_metric, fitted_cat_metric in all_fitted_cat_metric.items():
                    if fitted_cat_metric is not None and f"{num_metric}_{cat_metric}" not in result:
                        # print(num_metric, cat_metric, end=", ")
                        result[f"{num_metric}_{cat_metric}"] = eval_with_data(
                            algorithm, Xnum, Xcat, y, n_clusters, fitted_num_metric, fitted_cat_metric)
                for bin_metric, fitted_bin_metric in all_fitted_bin_metric.items():
                    if fitted_bin_metric is not None and f"{num_metric}_{bin_metric}" not in result:
                        # print(num_metric, bin_metric, end=", ")
                        result[f"{num_metric}_{bin_metric}"] = eval_with_data(
                            algorithm, Xnum, Xdummy, y, n_clusters, fitted_num_metric, fitted_bin_metric)
    else:
        tempdir = tempfile.TemporaryDirectory(dir="/homedir/adiop/tmp/")
        for num_metric in base_metrics.get_available_metrics(data_type="numeric"):
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
                        # print(num_metric, cat_metric, end=", ")
                        result[f"{num_metric}_{cat_metric}"] = eval_with_pairwise_dist(
                            algorithm, Dnum, Dcat, y, n_clusters
                        )
                for bin_metric in base_metrics.get_available_metrics(data_type="binary"):
                    filename = os.path.join(tempdir.name, bin_metric)
                    if bin_metric in computed:
                        with open(filename, "rb") as f:
                            Dcat = pickle.load(f)
                    else:
                        Dcat = compute_dissim(Xdummy, bin_metric)
                        with open(filename, "wb") as f:
                            pickle.dump(Dcat, f)
                    if Dcat is not None:
                        # print(num_metric, bin_metric, end=", ")
                        result[f"{num_metric}_{bin_metric}"] = eval_with_pairwise_dist(
                            algorithm, Dnum, Dcat, y, n_clusters
                        )
    end = time.time()
    # print(result, flush=True)
    if len(result) > 0:
        duration = end - start
        result_file = os.path.join(outputdir, f"scores/{data['id']}.pickle")
        with open(result_file, "wb") as f:
            pickle.dump(result, f)

        time_file = os.path.join(outputdir, "times.json")
        with FileLock(time_file + ".lock"):
            times = {}
            if os.path.isfile(time_file):
                with open(time_file, "r", encoding="utf-8") as f:
                    times = json.load(f)
            times[data["id"]] = duration
            with open(time_file, "w", encoding="utf-8") as f:
                json.dump(times, f, indent=4, ensure_ascii=False)
    gc.collect()
    return duration


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Benchmark data set data sets')
    parser.add_argument("-d", "--datasetsdir", default=None)
    parser.add_argument(
        "-o", "--outputdir", help="The directory where results will be saved")
    parser.add_argument("-a", "--algorithm",
                        help="The clustering algorithm that will be used")
    parser.add_argument(
        "-j", "--jobs", help="The number of concurent workers", default=-1)
    parser.add_argument(
        "-r", "--replace", help="Update or replace result if exist", action="store_true")
    args = parser.parse_args()

    print(f"outputdir: {args.outputdir} - algorithm: {args.algorithm}")

    already_handled_datasets = [filename.split('.')[0]
        for filename in os.listdir(os.path.join(args.outputdir, "scores/"))]

    filenames = []
    if args.datasetsdir is not None:
        filenames += [os.path.join(args.datasetsdir, filename)
                      for filename in os.listdir(args.datasetsdir)
                      if filename.split('.')[0] not in already_handled_datasets]
    
    print(f"{len(filenames)}/{len(os.listdir(args.datasetsdir))} \
datasets to benchmark, {len(already_handled_datasets)} already handled")

    datasets = []
    for filename in filenames:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        data['id'] = filename.split('/')[-1].split(".")[0]
        _, data['y'] = np.unique(data['y'], return_inverse=True)
        datasets.append(data)

    print()
    print("BENCHMARKING...")
    start = time.time()
    list_ret = Parallel(n_jobs=int(args.jobs), verbose=60)(
        delayed(benchmark)(
            data, args.outputdir,
            algorithm=args.algorithm
        ) for data in datasets
    )
    # for i, data in enumerate(sorted(datasets, key=lambda d: len(d["samples"]))):
    #     print()
    #     print()
    #     print("On dataset",
    #           data["id"], f"({i+1}/{len(datasets)}) ------------------------------------")
    #     benchmark(
    #         data, args.outputdir,
    #         algorithm=args.algorithm
    #     )
    end = time.time()
    print(f"END. total time = {end - start}")
