"""
Compute distance matrices
"""

# pylint: disable=bare-except

import json
import os
import pickle
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.preprocessing import OneHotEncoder, minmax_scale

sys.path.append(".")
from utils import get_valid_similarity_measures
from metrics import base_metrics
from clustering_algorithms import kprototypes
from sklearn.preprocessing import OrdinalEncoder

def compute_distance_matrices(data, output_dir, cache_dir="", n_jobs=1):
    Xnum = minmax_scale(data["Xnum"])
    Xcat = OrdinalEncoder().fit_transform(data["Xcat"]).astype(int)
    Xdummy = OneHotEncoder(handle_unknown='ignore').fit_transform(Xcat).toarray()
    y = data["y"]
    dataset_name = data["id"]

    times = {}
    time_file = os.path.join(output_dir, f"{dataset_name}.json")
    if os.path.isfile(time_file):
        with open(time_file, "r", encoding="utf-8") as fp:
            times = json.load(fp)
    
    for metric in get_valid_similarity_measures(Xnum, data_type="numeric"):
        # print("metric:", metric, end="...")
        t0 = time.time()
        m = base_metrics.get_metric(metric, caching=True, cache_dir=cache_dir).fit(Xnum, dataset_name=dataset_name)
        t1 = time.time()
        m.pairwise(Xnum, dataset_name=dataset_name, n_jobs=n_jobs)
        t2 = time.time()
        # print("DONE")
        # if metric not in times:
        times[metric] = {}
        times[metric]["fit"] = t1 - t0
        times[metric]["pairwise"] = t2 - t1
        times[metric]["total_time"] = t2 - t0

    for metric in get_valid_similarity_measures(Xcat, data_type="categorical"):
        # print("metric:", metric, end="...")
        t0 = time.time()
        m = base_metrics.get_metric(metric, caching=True, cache_dir=cache_dir).fit(Xcat, dataset_name=dataset_name)
        t1 = time.time()
        m.pairwise(Xcat, dataset_name=dataset_name, n_jobs=n_jobs)
        t2 = time.time()
        # print("DONE")
        # if metric not in times:
        times[metric] = {}
        times[metric]["fit"] = t1 - t0
        times[metric]["pairwise"] = t2 - t1
        times[metric]["total_time"] = t2 - t0

    for metric in get_valid_similarity_measures(Xdummy, data_type="binary"):
        # print("metric:", metric, end="...")
        t0 = time.time()
        m = base_metrics.get_metric(metric, caching=True, cache_dir=cache_dir).fit(Xdummy, dataset_name=dataset_name)
        t1 = time.time()
        m.pairwise(Xdummy, dataset_name=dataset_name, n_jobs=n_jobs)
        t2 = time.time()
        # print("DONE")
        # if metric not in times:
        times[metric] = {}
        times[metric]["fit"] = t1 - t0
        times[metric]["pairwise"] = t2 - t1
        times[metric]["total_time"] = t2 - t0

    with open(time_file, "w", encoding="utf-8") as f:
        json.dump(times, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    import argparse
    from datetime import datetime
    parser = argparse.ArgumentParser(
        description='Compute distance matrices')
    parser.add_argument("-d", "--datasetsdir", default=None)
    parser.add_argument(
        "-o", "--outputdir", help="The directory where results will be saved")
    parser.add_argument(
        "-c", "--cachedir", help="The directory where distance data wil be cached")
    parser.add_argument(
        "-j", "--jobs", help="The number of concurent workers", default=1)
    args = parser.parse_args()

    print(datetime.now())
    print("Computing distance matrices...")
    print("dataset directory:", args.datasetsdir)
    print("# of datasets:", len(os.listdir(args.datasetsdir)))
    
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
        delayed(compute_distance_matrices)(
            data, args.outputdir,
            cache_dir=args.cachedir,
            n_jobs=1
        ) for data in sorted(datasets, key=lambda d: len(d["samples"]) * \
        (len(d["numeric_attributes"])+len(d["categorical_attributes"])))
    )
    # for i, data in enumerate(sorted(datasets, key=lambda d: len(d["samples"]) * \
    #     (len(d["numeric_attributes"])+len(d["categorical_attributes"])))):
    #     print()
    #     print(f"Curent dataset: {data['id']} ({i+1}/{len(datasets)}), time: {time.time() - start}")
    #     compute_distance_matrices(
    #         data, time_file,
    #         cache_dir=args.cachedir,
    #         n_jobs=int(args.jobs)
    #     )
    end = time.time()
    print()
    print(f"END. total time: {end - start}")
