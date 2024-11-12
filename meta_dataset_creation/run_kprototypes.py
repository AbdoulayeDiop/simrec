"""
Run K-Prototypes
"""

# pylint: disable=bare-except

import os
import pickle
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import OneHotEncoder, minmax_scale

sys.path.append(".")
from utils import get_valid_similarity_pairs
from metrics import base_metrics
from clustering_algorithms import kprototypes
from sklearn.preprocessing import OrdinalEncoder

gamma_values = np.concatenate((np.linspace(0, 0.1, 10), np.linspace(
    0.1, 1, 10), np.arange(2, 10), np.arange(10, 101, 10)))

np.random.seed(0)

def compute_clusters(Xnum, Xcat, y, num_metric, cat_metric, dataset_name=None, n_jobs=1):
    def run(n_clusters, gamma):
        start = time.time()
        clusters = kprototypes(Xnum, Xcat, n_clusters, num_metric, cat_metric, gamma=gamma, dataset_name=dataset_name)
        end = time.time()
        return {
            "params": {
                "gamma": gamma,
                "n_clusters": n_clusters
            },
            "clusters": clusters,
            "time": end - start
        }
    n_clusters_ground_truth = len(set(y))
    # clustering_results = Parallel(n_jobs=n_jobs)(delayed(run)(n_clusters_ground_truth, gamma) for gamma in gamma_values)
    clustering_results = []
    for gamma in gamma_values:
        clustering_results.append(run(n_clusters_ground_truth, gamma))
    return clustering_results

def compute_clusters_for_all_similarity_measures(data, results_dir, cache_dir="", n_jobs=1):
    Xnum = minmax_scale(data["Xnum"])
    Xcat = OrdinalEncoder().fit_transform(data["Xcat"]).astype(int)
    Xdummy = OneHotEncoder(handle_unknown='ignore').fit_transform(Xcat).toarray()
    y = data["y"]
    dataset_name = data["id"]

    result = {}
    result_file = os.path.join(results_dir, f"{data['id']}.pickle")
    if os.path.isfile(result_file):
        with open(result_file, "rb") as fp:
            result = pickle.load(fp)
            
    for similarity_pair in get_valid_similarity_pairs(Xnum, Xcat):
        if similarity_pair not in result:
            num_metric, cat_metric = similarity_pair.split('_')
            newXcat = Xcat if cat_metric in base_metrics.get_available_metrics(data_type="categorical") else Xdummy
            num_metric = base_metrics.get_metric(num_metric, caching=True, cache_dir=cache_dir).fit(Xnum, dataset_name=dataset_name)
            cat_metric = base_metrics.get_metric(cat_metric, caching=True, cache_dir=cache_dir).fit(newXcat, dataset_name=dataset_name)
            result[similarity_pair] = compute_clusters(Xnum, newXcat, y, num_metric, cat_metric, dataset_name=dataset_name, n_jobs=n_jobs)

    if len(result) > 0:
        with open(result_file, "wb") as fp:
            pickle.dump(result, fp)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Run K-Prototypes algorithm on all datasets')
    parser.add_argument("-d", "--datasetsdir", default=None)
    parser.add_argument(
        "-o", "--outputdir", help="The directory where results will be saved")
    parser.add_argument(
        "-c", "--cachedir", help="The directory where distance data wil be cached")
    parser.add_argument(
        "-j", "--jobs", help="The number of concurent workers", default=1)
    args = parser.parse_args()

    print("Running K-Prototypes algorithm...")
    print("dataset directory:", args.datasetsdir)
    print("# of datasets:", len(os.listdir(args.datasetsdir)))

    results_dir = os.path.join(args.outputdir, "kprototypes/")
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    results_dir = os.path.join(results_dir, "clustering_results")
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    filenames = []
    if args.datasetsdir is not None:
        filenames += [os.path.join(args.datasetsdir, filename)
                      for filename in os.listdir(args.datasetsdir)]

    print()
    print("Loading datasets...", end="")
    datasets = []
    for filename in filenames:
        with open(filename, "rb") as f:
            data = pickle.load(f)
        data['id'] = filename.split('/')[-1].split(".")[0]
        _, data['y'] = np.unique(data['y'], return_inverse=True)
        datasets.append(data)
    print("DONE")

    print()
    print("BENCHMARKING...")
    start = time.time()
    list_ret = Parallel(n_jobs=int(args.jobs), verbose=60)(
        delayed(compute_clusters_for_all_similarity_measures)(
            data, results_dir,
            cache_dir=args.cachedir,
            n_jobs=1
        ) for data in sorted(datasets, key=lambda d: len(d["samples"]) * \
        (len(d["numeric_attributes"])+len(d["categorical_attributes"])))
    )
    # for i, data in enumerate(sorted(datasets, key=lambda d: len(d["samples"]) * \
    #     (len(d["numeric_attributes"])+len(d["categorical_attributes"])))):
    #     print()
    #     print(f"Curent dataset: {data['id']} ({i+1}/{len(datasets)}), time: {time.time() - start}")
    #     compute_clusters_for_all_similarity_measures(
    #         data, results_dir,
    #         cache_dir=args.cachedir,
    #         n_jobs=int(args.jobs)
    #     )
    end = time.time()
    print()
    print(f"END. total time: {end - start}")
