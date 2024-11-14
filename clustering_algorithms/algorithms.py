"""
Clustering algorithms
"""
# pylint: disable=bare-except

import traceback

import numpy as np
from kmedoids import alternating, fasterpam
from sklearn.cluster import AgglomerativeClustering, spectral_clustering as sc
import os
import sys

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(FILE_DIR)
sys.path.append(PARENT_DIR)

from metrics import base_metrics
from clustering_algorithms.kprototypes import KPrototypes as KP # pylint: disable=import-error
from clustering_algorithms.LSHkRepresentatives.LSHkPrototypes import LSHkPrototypes


def haverage(Dnum, Dcat, w, n_clusters):
    D = (1-w)*Dnum + w*Dcat
    ac = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="average",
        affinity="precomputed"
    )
    clusters = None
    try:
        clusters = ac.fit_predict(D)
    except:
        print("Error : haverage")
        print(f"Traceback : {traceback.format_exc()}")
    return clusters


def kmedoids(Dnum, Dcat, w, n_clusters, method="fasterpam", n_init=10):
    D = (1-w)*Dnum + w*Dcat
    final_clusters = None
    best_score = np.inf
    for random_state in range(n_init):
        try:
            res = fasterpam(D, n_clusters, random_state=random_state, n_cpu=1) if method == "fasterpam" \
                else alternating(D, n_clusters, random_state=random_state)
            clusters = res.labels
            score = res.loss
            if score < best_score:
                final_clusters = [val for val in clusters]
        except:
            print(
                f"Error : k-medoids -> method={method} ; random_state={random_state}")
            print(
                f"Traceback : {traceback.format_exc()}"
            )
    return final_clusters


def spectral_clustering(Dnum, Dcat, w, n_clusters, sigma):
    D = (1-w)*Dnum + w*Dcat
    clusters = None
    try:
        S = np.exp(-D**2/(2*sigma**2))
        S = (S + S.T)/2
        clusters = sc(S, n_clusters=n_clusters, random_state=0)
    except:
        print(f"Error : spectral -> sigma={sigma}")
    return clusters

def kprototypes(Xnum, Xcat, n_clusters, num_metric, cat_metric, gamma=1, n_init=5, dataset_name=None):
    final_clusters = None
    best_loss = np.inf
    num_metric = check_metric(num_metric).fit(Xnum, dataset_name=dataset_name)
    cat_metric = check_metric(cat_metric).fit(Xcat, dataset_name=dataset_name)
    for random_state in range(n_init):
        try:
            kp = KP(
                n_clusters=n_clusters,
                gamma=gamma,
                numerical_distance=num_metric.pairwise,
                categorical_distance=cat_metric.pairwise,
                random_state=random_state,
            )

            clusters = kp.fit_predict(Xnum, Xcat)
            loss = kp.cost
            if loss < best_loss:
                final_clusters = [val for val in clusters]
                best_loss = loss
        except:
            print(f"Error : KPrototypes, num_metric={num_metric}, cat_metric={cat_metric}, gamma={gamma}")
            print(f"Traceback : {traceback.format_exc()}")

    return final_clusters

def lshkprototypes(Xnum, Xcat, n_clusters, num_metric, cat_metric, w=0.5, n_init=5, dataset_name=None):
    clusters = None
    X = np.c_[Xnum, Xcat]
    attributeMasks = [1 for _ in range(Xnum.shape[1])] + [0 for _ in range(Xcat.shape[1])]
    try:
        lsh = LSHkPrototypes(
            n_clusters,
            num_metric=num_metric,
            cat_metric=cat_metric,
            numerical_weight=1-w,
            categorical_weight=w,
            n_init=n_init
        )
        clusters = lsh.fit(X, attributeMasks=attributeMasks, dataset_name=dataset_name)
    except:
        print(f"Error : lshkprototypes, num_metric={num_metric}, cat_metric={cat_metric}, w={w}")
        print(f"Traceback : {traceback.format_exc()}")
    return clusters

def check_metric(metric):
    if isinstance(metric, base_metrics.Metric):
        return metric
    if isinstance(metric, str):
        return base_metrics.get_metric(metric)
    raise(Exception("metric should be of type str or Metric"))

if __name__ == "__main__":
    import pickle
    import timeit
    with open(os.path.join(PARENT_DIR, "example_datasets/10.pickle"), "rb") as fp: 
        data = pickle.load(fp)
    Xnum = data["Xnum"]
    Xcat = data["Xcat"]
    y = data["y"]
    n_clusters = len(set(y))
    num_metric="divergence"
    cat_metric="co-oc"
    num_metric = base_metrics.get_metric(num_metric, caching=True)
    cat_metric = base_metrics.get_metric(cat_metric, caching=True)
    start = timeit.default_timer()
    # clusters = lshkprototypes(Xnum, Xcat, n_clusters, num_metric, cat_metric, w=0.5, n_init=5, dataset_name=data["id"])
    clusters = kprototypes(Xnum, Xcat, n_clusters, num_metric, cat_metric, dataset_name=data["id"])
    end = timeit.default_timer()
    print(end - start)