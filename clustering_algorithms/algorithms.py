"""
Clustering algorithms
"""
# pylint: disable=bare-except

import traceback

import numpy as np
from kmedoids import alternating, fasterpam
from sklearn.cluster import AgglomerativeClustering, spectral_clustering as sc
import sys
from clustering_algorithms.kprototypes import KPrototypes as KP # pylint: disable=import-error


def haverage(Dnum, Dcat, w, n_clusters):
    D = (1-w)*Dnum + w*Dcat
    ac = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="average",
        metric="precomputed"
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


def kprototypes(Xnum, Xcat, n_clusters, num_metric, cat_metric, gamma, n_init=5):
    final_clusters = None
    best_loss = np.inf
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
            print(f"Error : KPrototypes {num_metric}, {cat_metric}")
            print(f"Traceback : {traceback.format_exc()}")

    return final_clusters
