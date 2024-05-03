import time
from collections.abc import Iterable

import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score, pairwise_distances
from joblib import Parallel, delayed
from itertools import product


def n_instances(X):
    return X.shape[0]


def n_features(X):
    return X.shape[1]


def dim(X):
    return X.shape[1] / X.shape[0]


def num_on_cat(Xnum=None, Xcat=None):
    return Xnum.shape[1] / Xcat.shape[1]

# statistics on attributes #######################################


def means_num_att(X):
    l = [np.mean(X[:, j]) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]


def std_num_att(X):
    l = [np.std(X[:, j]) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]

# statistics on squared attributes ###############################


def means_squared_num_att(X):
    l = [np.mean(X[:, j]**2) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]


def std_squared_num_att(X):
    l = [np.std(X[:, j]**2) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]

# statistics on products of attributes values pairs ##############


def means_internal_product_num_att(X, n_jobs=-1):
    # l = [np.mean([x * y for x, y in product((X[:, j]), X[:, j])]) for j in range(X.shape[1])]
    indices = np.triu_indices(X.shape[0])
    if X.shape[0] < 1000:
        l = [np.mean(X[indices[0], j] * X[indices[1], j]) for j in range(X.shape[1])]
    else:
        l = Parallel(n_jobs=n_jobs)(delayed(lambda x: np.mean(X[indices[0], x] * X[indices[1], x]))(j) for j in range(X.shape[1]))
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]


def std_internal_product_num_att(X, n_jobs=-1):
    # l = [np.std([x * y for x, y in product((X[:, j]), X[:, j])]) for j in range(X.shape[1])]
    indices = np.triu_indices(X.shape[0], 1)
    if X.shape[0] < 1000:
        l = [np.std(X[indices[0], j] * X[indices[1], j]) for j in range(X.shape[1])]
    else:
        l = Parallel(n_jobs=n_jobs)(delayed(lambda x: np.std(X[indices[0], x] * X[indices[1], x]))(j) for j in range(X.shape[1]))
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]

# covariance #####################################################


def covariance(X):
    l = np.cov(X, rowvar=False)[np.triu_indices(
        X.shape[1])] if X.shape[1] > 1 else [np.var(X)]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]


def card_cat_att(X):
    l = [len(set(X[:, j])) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]


def myentropy(labels, base=None):
    _, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)


def entropy_cat_att(X):
    l = [myentropy(X[:, j]) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]


def mutual_info_cat_att(X, n_jobs=16):
    indices = np.triu_indices(X.shape[1])
    if X.shape[1] < 50:
        l = [mutual_info_score(X[:, j1], X[:, j2]) for j1, j2 in zip(*indices)]
    else:
        l = Parallel(n_jobs=n_jobs)(delayed(lambda j1, j2: mutual_info_score(X[:, j1], X[:, j2]))(j1, j2) for j1, j2 in zip(*indices))
    # l = Parallel(n_jobs=n_jobs)(delayed(lambda j1, j2: mutual_info_score(X[:, j1], X[:, j2]))(j1, j2) for j1, j2 in zip(*indices))
    # l = [mutual_info_score(X[:, j1], X[:, j2]) for j1, j2 in zip(*indices)]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]

# std of the frequency of categories of a given attributes #######


def std_freq_cat_att(X):
    l = [np.std(np.unique(X[:, j], return_counts=True)[1]/X.shape[0])
         for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]


GENERAL_META_FEATURES_DICT = {
    "n_instances": n_instances,
    "n_features": n_features,
    "dim": dim,
    "num_on_cat": num_on_cat,
}

NUM_META_FEATURES_DICT = {
    "n_num_att": n_features,

    "means_num_att": means_num_att,
    "std_num_att": std_num_att,

    "means_squared_num_att": means_squared_num_att,
    "std_squared_num_att": std_squared_num_att,

    "means_internal_product_num_att": means_internal_product_num_att,
    "std_internal_product_num_att": std_internal_product_num_att,

    "covariance": covariance,
}

CAT_META_FEATURES_DICT = {
    "n_cat_att": n_features,

    "card_cat_att": card_cat_att,
    "entropy_cat_att": entropy_cat_att,

    "mutual_info_cat_att": mutual_info_cat_att,

    "std_freq_cat_att": std_freq_cat_att
}

def get_general_meta_feature(name):
    return GENERAL_META_FEATURES_DICT[name]


def get_num_meta_feature(name):
    return NUM_META_FEATURES_DICT[name]


def get_cat_meta_feature(name):
    return CAT_META_FEATURES_DICT[name]


def get_general_meta_features():
    return GENERAL_META_FEATURES_DICT


def get_num_meta_features():
    return NUM_META_FEATURES_DICT


def get_cat_meta_features():
    return CAT_META_FEATURES_DICT

def get_attribute_names(name):
    if name in ["n_instances", "n_features", "dim", "num_on_cat", "n_num_att", "n_cat_att"]:
        return [name]
    elif name in [
        "means_num_att", "std_num_att",
        "means_squared_num_att", "std_squared_num_att",
        "means_internal_product_num_att", "std_internal_product_num_att",
        "covariance",
        "card_cat_att", "entropy_cat_att", "mutual_info_cat_att", "std_freq_cat_att"
    ]:
        return [f"{p}_{name}" for p in ["min", "q1", "mean", "q3", "max"]]
    return

ALL_ATTRIBUTE_NAMES = []
for name in get_general_meta_features().keys():
    ALL_ATTRIBUTE_NAMES += get_attribute_names(name)
for name in get_num_meta_features().keys():
    ALL_ATTRIBUTE_NAMES += get_attribute_names(name)
for name in get_cat_meta_features().keys():
    ALL_ATTRIBUTE_NAMES += get_attribute_names(name)

def compute_meta_features(Xnum, Xcat, return_time=False, selected=None):
    X = np.c_[Xnum, Xcat]
    meta_x = []
    if return_time:
        start = time.time()
    for name, meta_feature in get_general_meta_features().items():
        # t1 = time.time()
        if name in ["num_on_cat"]:
            v = meta_feature(Xnum=Xnum, Xcat=Xcat)
        else:
            v = meta_feature(X)
        # t2 = time.time()
        # print(name, t2 - t1)
        if isinstance(v, Iterable):
            meta_x += list(v)
        else:
            meta_x.append(v)

    for name, meta_feature in get_num_meta_features().items():
        # t1 = time.time()
        v = meta_feature(Xnum)
        # t2 = time.time()
        # print(name, t2 - t1)
        if isinstance(v, Iterable):
            meta_x += list(v)
        else:
            meta_x.append(v)

    for name, meta_feature in get_cat_meta_features().items():
        # t1 = time.time()
        v = meta_feature(Xcat)
        # t2 = time.time()
        # print(name, t2 - t1)
        if isinstance(v, Iterable):
            meta_x += list(v)
        else:
            meta_x.append(v)
    meta_x = np.array(meta_x)
    if selected is not None:
        meta_x = meta_x[selected]
    if return_time:
        end = time.time()
        return meta_x, end-start
    return meta_x


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import minmax_scale
    n_samples = 3000
    Xnum, _ = make_blobs(n_samples=n_samples, centers=5, n_features=200)
    Xcat = np.random.randint(2, size=(n_samples, 1000))
    Xnum = minmax_scale(Xnum)
    x, t = compute_meta_features(Xnum=Xnum, Xcat=Xcat, return_time=True)
    print("time (s):", t)
    print(x)
