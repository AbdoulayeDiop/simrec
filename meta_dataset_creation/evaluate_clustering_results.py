"""
Evaluate clustering results
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
from sklearn.preprocessing import OrdinalEncoder
from meta_dataset_creation import cvi

def evaluate_clustering_results(data, clustering_results_dir, scores_dir, cache_dir="", n_jobs=1):
    Xnum = minmax_scale(data["Xnum"])
    Xcat = OrdinalEncoder().fit_transform(data["Xcat"]).astype(int)
    Xdummy = OneHotEncoder(handle_unknown='ignore').fit_transform(Xcat).toarray()
    y = data["y"]
    dataset_name = data["id"]

    clustering_results_file = os.path.join(clustering_results_dir, f"{dataset_name}.pickle")
    with open(clustering_results_file, "rb") as fp:
        clustering_results = pickle.load(fp)

    scores = {}
    scores_file = os.path.join(scores_dir, f"{data['id']}.pickle")
    if os.path.isfile(scores_file):
        with open(scores_file, "rb") as fp:
            scores = pickle.load(fp)
    
    for similarity_pair in clustering_results:# [sp for sp in clustering_results if sp[-5:]=="dilca"] The condition is TEMPORARY and should be deleted
        # print(similarity_pair)
        if similarity_pair not in scores:
            scores[similarity_pair] = {}
            
        for cvi_name in cvi.EXTERNAL_EVAL_METRICS:
            if cvi_name not in scores[similarity_pair] and isinstance(clustering_results[similarity_pair], list):
                scores[similarity_pair][cvi_name] = []
                for result in clustering_results[similarity_pair]:
                    obj = {}
                    obj["params"] = result["params"]
                    if result["clusters"] is not None:
                        start = time.time()
                        obj["score"] = cvi.get_score(y, result["clusters"], eval_metric=cvi_name)
                        obj["time"] = time.time() - start
                    else:
                        obj["score"] = -1
                    scores[similarity_pair][cvi_name].append(obj)

        go = False
        for cvi_name in cvi.INTERNAL_EVAL_METRICS:
            if cvi_name not in scores[similarity_pair]:
                go = True
                break
        
        if go:
            num_metric, cat_metric = similarity_pair.split('_')
            newXcat = Xcat if cat_metric in base_metrics.get_available_metrics(data_type="categorical") else Xdummy
            num_metric = base_metrics.get_metric(num_metric, caching=True, cache_dir=cache_dir).fit(Xnum, dataset_name=dataset_name)
            cat_metric = base_metrics.get_metric(cat_metric, caching=True, cache_dir=cache_dir).fit(newXcat, dataset_name=dataset_name)
            Dnum = num_metric.pairwise(Xnum, dataset_name=dataset_name)
            Dcat = cat_metric.pairwise(newXcat, dataset_name=dataset_name)

            for cvi_name in cvi.INTERNAL_EVAL_METRICS:
                if cvi_name not in scores[similarity_pair] and isinstance(clustering_results[similarity_pair], list):
                    scores[similarity_pair][cvi_name] = []
                    for result in clustering_results[similarity_pair]:
                        obj = {}
                        obj["params"] = result["params"]
                        if result["clusters"] is not None:
                            D = None
                            if "gamma" in obj["params"]:
                                D = Dnum + obj["params"]["gamma"] * Dcat
                            if "w" in obj["params"]:
                                D = (1 - obj["params"]["w"]) * Dnum + obj["params"]["w"] * Dcat
                            start = time.time()
                            obj["score"] = cvi.get_unsupervised_score(
                                D, result["clusters"], eval_metric=cvi_name, metric="precomputed")
                            obj["time"] = time.time() - start
                        else:
                            obj["score"] = -1
                        scores[similarity_pair][cvi_name].append(obj)

    if len(scores) > 0:
        # print(scores)
        with open(scores_file, "wb") as fp:
            pickle.dump(scores, fp)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Compute cluster validity indices')
    parser.add_argument(
        "-d", "--datasetsdir", help="The dataset directory")
    parser.add_argument(
        "-r", "--resultsdir", help="The directory where clustering results have been saved")
    parser.add_argument(
        "-c", "--cachedir", help="The directory where distance data wil be cached")
    parser.add_argument(
        "-j", "--jobs", help="The number of concurent workers", default=1)
    args = parser.parse_args()

    print("Evaluating clustering results...")
    print("Results directory:", args.resultsdir)


    clustering_results_dir = os.path.join(args.resultsdir, "clustering_results/")
    scores_dir = os.path.join(args.resultsdir, "scores/")
    if not os.path.isdir(scores_dir):
        os.makedirs(scores_dir)

    print("# of datasets:", len(os.listdir(clustering_results_dir)))

    print()
    print("Loading datasets...", end="")
    datasets = []
    for filename in os.listdir(clustering_results_dir):
        filepath = os.path.join(args.datasetsdir, filename)
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        data['id'] = filename.split('/')[-1].split(".")[0]
        _, data['y'] = np.unique(data['y'], return_inverse=True)
        datasets.append(data)
    print("DONE")

    print()
    print("BENCHMARKING...")
    start = time.time()
    list_ret = Parallel(n_jobs=int(args.jobs), verbose=60)(
        delayed(evaluate_clustering_results)(
            data, clustering_results_dir,
            scores_dir, cache_dir=args.cachedir,
            n_jobs=int(args.jobs)
        ) for data in sorted(datasets, key=lambda d: len(d["samples"]) * \
        (len(d["numeric_attributes"])+len(d["categorical_attributes"])))
    )
    # for i, data in enumerate(sorted(datasets, key=lambda d: len(d["samples"]) * \
    #     (len(d["numeric_attributes"])+len(d["categorical_attributes"])))):
    #     print()
    #     print(f"Curent dataset: {data['id']} ({i+1}/{len(datasets)}), time: {time.time() - start}")
    #     evaluate_clustering_results(
    #         data, clustering_results_dir,
    #         scores_dir, cache_dir=args.cachedir,
    #         n_jobs=int(args.jobs)
    #     )
    end = time.time()
    print()
    print(f"END. total time: {end - start}")
