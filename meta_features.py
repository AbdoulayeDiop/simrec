import time
from collections.abc import Iterable

import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score, pairwise_distances


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


def means_internal_product_num_att(X):
    l = [np.mean([X[i1, j]*X[i2, j] for i1 in range(X.shape[0]-1)
                 for i2 in range(i1+1, X.shape[0])]) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]


def std_internal_product_num_att(X):
    l = [np.std([X[i1, j]*X[i2, j] for i1 in range(X.shape[0]-1)
                for i2 in range(i1+1, X.shape[0])]) for j in range(X.shape[1])]
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


def mutual_info_cat_att(X):
    mi = pairwise_distances(X.T, metric=mutual_info_score)
    l = mi[np.triu_indices(mi.shape[0])]
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

def compute_meta_features(Xnum, Xcat, return_time=False):
    X = np.c_[Xnum, Xcat]
    meta_x = []
    if return_time:
        start = time.time()
    for name, meta_feature in get_general_meta_features().items():
        if name in ["num_on_cat"]:
            v = meta_feature(Xnum=Xnum, Xcat=Xcat)
        else:
            v = meta_feature(X)
        if isinstance(v, Iterable):
            meta_x += list(v)
        else:
            meta_x.append(v)

    for name, meta_feature in get_num_meta_features().items():
        v = meta_feature(Xnum)
        if isinstance(v, Iterable):
            meta_x += list(v)
        else:
            meta_x.append(v)

    for name, meta_feature in get_cat_meta_features().items():
        v = meta_feature(Xcat)
        if isinstance(v, Iterable):
            meta_x += list(v)
        else:
            meta_x.append(v)
    if return_time:
        end = time.time()
        return meta_x, end-start
    return meta_x


if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import minmax_scale
    X, y = make_blobs(n_samples=500, centers=5, n_features=8)
    X = minmax_scale(X)
    x = compute_meta_features(Xnum=X, Xcat=y.reshape(-1, 1))
    print(len(x))
    print(x)
