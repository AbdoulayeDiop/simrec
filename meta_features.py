import numpy as np
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD

import base_metrics
from utils import get_score

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
    l = [np.mean(X[:,j]) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]
    
def std_num_att(X):
    l = [np.std(X[:,j]) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]

def kurtosis_num_att(X):
    l = [kurtosis(X[:,j]) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]

def skewness_num_att(X):
    l = [skew(X[:,j]) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]

# statistics on squared attributes ###############################
def means_squared_num_att(X):
    l = [np.mean(X[:,j]**2) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]
    
def std_squared_num_att(X):
    l = [np.std(X[:,j]**2) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]

def kurtosis_squared_num_att(X):
    l = [kurtosis(X[:,j]**2) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]

def skewness_squared_num_att(X):
    l = [skew(X[:,j]**2) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]

# statistics on products of attributes values pairs ##############
def means_internal_product_num_att(X):
    l = [np.mean([X[i1,j]*X[i2,j] for i1 in range(X.shape[0]-1) for i2 in range(i1+1, X.shape[0])]) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]
    
def std_internal_product_num_att(X):
    l = [np.std([X[i1,j]*X[i2,j] for i1 in range(X.shape[0]-1) for i2 in range(i1+1, X.shape[0])]) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]

def kurtosis_internal_product_num_att(X):
    l = [kurtosis([X[i1,j]*X[i2,j] for i1 in range(X.shape[0]-1) for i2 in range(i1+1, X.shape[0])]) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]

def skewness_internal_product_num_att(X):
    l = [skew([X[i1,j]*X[i2,j] for i1 in range(X.shape[0]-1) for i2 in range(i1+1, X.shape[0])]) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]

# covariance #####################################################
def covariance(X):
    l = np.cov(X, rowvar=False)[np.triu_indices(X.shape[1])] if X.shape[1] > 1 else [np.var(X)]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]

def euclidean_distance_distribution(X):
    D = base_metrics.get_metric("euclidean").fit(X).pairwise(X, n_jobs=-1)
    l = D[np.triu_indices(D.shape[0])]
    hist = np.histogram(l/max(l), bins=10)[0]
    return hist/sum(hist)

def cosine_distance_distribution(X):
    D = base_metrics.get_metric("cosine").fit(X).pairwise(X, n_jobs=-1)
    l = D[np.triu_indices(D.shape[0])]
    hist = np.histogram(l/max(l), bins=10)[0]
    return hist/sum(hist)

def canberra_distance_distribution(X):
    D = base_metrics.get_metric("canberra").fit(X).pairwise(X, n_jobs=-1)
    l = D[np.triu_indices(D.shape[0])]
    hist = np.histogram(l/max(l), bins=10)[0]
    return hist/sum(hist)

def lorentzian_distance_distribution(X):
    D = base_metrics.get_metric("lorentzian").fit(X).pairwise(X, n_jobs=-1)
    l = D[np.triu_indices(D.shape[0])]
    hist = np.histogram(l/max(l), bins=10)[0]
    return hist/sum(hist)

def divergence_distance_distribution(X):
    D = base_metrics.get_metric("divergence").fit(X).pairwise(X, n_jobs=-1)
    l = D[np.triu_indices(D.shape[0])]
    hist = np.histogram(l/max(l), bins=10)[0]
    return hist/sum(hist)

# def k_means_score(X):
#     n_clusters = len(set(y))
#     return get_score(y, KMeans(n_clusters=n_clusters).fit_predict(X))

def card_cat_att(X):
    l = [len(set(X[:,j])) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]
    
def myentropy(labels, base=None):
    _, counts = np.unique(labels, return_counts=True)
    return entropy(counts, base=base)

def entropy_cat_att(X):
    l = [myentropy(X[:,j]) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]

def mutual_info_cat_att(X):
    mi = pairwise_distances(X.T, metric=mutual_info_score)
    l = mi[np.triu_indices(mi.shape[0])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]

# std of the frequency of categories of a given attributes #######
def std_freq_cat_att(X):
    l = [np.std(np.unique(X[:,j], return_counts=True)[1]/X.shape[0]) for j in range(X.shape[1])]
    return [np.min(l), np.quantile(l, 0.25), np.mean(l), np.quantile(l, 0.75), np.max(l)]

class Node():
    def __init__(self, node_id=None, parent=None, elements=[], bounds=[], feature=None, split_value=None):
        self.id = node_id
        self.feature = feature
        self.split_value = split_value
        self.children_left = -1
        self.children_right = -1
        self.elements = elements
        self.parent = parent
        self.bounds = bounds

class Tree():
    def __init__(self, nodes={}):
        self.nodes = nodes

    def set_root(self, node):
        self.root = node
        self.nodes[node.id] = node

    def add_node(self, node):
        self.nodes[node.id] = node

    def get_node(self, node_id):
        return self.nodes[node_id]

def create_random_tree(X, l):
    n, m = X.shape
    t = Tree()
    root = Node(node_id=0, elements=list(range(n)), bounds=[(min(X[:,j]), max(X[:,j])) for j in range(m)])
    t.add_node(root)
    
    to_visit = []
    to_visit.append(0)
    next_id = 1
    while len(to_visit) < l:
        node_id = to_visit.pop(0)
        j = np.random.randint(m)
        s = np.random.uniform(*t.nodes[node_id].bounds[j])
        t.nodes[node_id].feature = j
        t.nodes[node_id].split_value = s

        children_left = Node(
            node_id=next_id, 
            parent=node_id, 
            elements=[i for i in range(n) if X[i, j] < s],
            bounds=[v if k != j else (v[0], s) for k, v in enumerate(t.nodes[node_id].bounds)]
        )
        t.add_node(children_left)
        t.nodes[node_id].children_left = next_id
        to_visit.append(next_id)
        next_id += 1

        children_right = Node(
            node_id=next_id, 
            parent=node_id, 
            elements=[i for i in range(n) if X[i, j] >= s],
            bounds=[v if k != j else (s, v[1]) for k, v in enumerate(t.nodes[node_id].bounds)]
        )
        t.add_node(children_right)
        t.nodes[node_id].children_right = next_id
        to_visit.append(next_id)
        next_id += 1
    t.leaf_nodes = sorted(to_visit)
    return t

def isolation_forest(Xnum=None, Xcat=None, n_components=10, n_trees=50, n_leaves=5):
    if Xnum is not None and Xcat is not None:
        Xnum = Xnum.reshape((Xnum.shape[0], -1))
        Xcat = Xcat.reshape((Xcat.shape[0], -1))
        X = np.c_[Xnum, OneHotEncoder().fit_transform(Xcat).toarray()]
    elif Xnum is not None:
        Xnum = Xnum.reshape((Xnum.shape[0], -1))
        X = Xnum
    elif Xcat is not None:
        Xcat = Xcat.reshape((Xcat.shape[0], -1))
        X = Xcat
    X_emb = np.zeros((X.shape[0], n_trees*n_leaves))

    for k in range(n_trees):
        # Creating the tree
        t = create_random_tree(X, n_leaves)
        for j, node_id in enumerate(t.leaf_nodes):
            for i in t.get_node(node_id).elements:
                X_emb[i, k*n_leaves + j] = 1
    svd = TruncatedSVD(n_components=n_components).fit(X_emb)
    res = np.zeros(n_components)
    res[:len(svd.singular_values_)] = svd.singular_values_
    return res
            


GENERAL_META_FEATURES_DICT = {
    "n_instances": n_instances, 
    "n_features": n_features,
    "dim": dim,
    "num_on_cat": num_on_cat,
    # "isolation_forest": isolation_forest,
}

NUM_META_FEATURES_DICT = {
    "n_num_att": n_features,

    "means_num_att": means_num_att,
    "std_num_att": std_num_att,
    # "kurtosis_num_att": kurtosis_num_att,
    # "skewness_num_att": skewness_num_att,

    "means_squared_num_att": means_squared_num_att,
    "std_squared_num_att": std_squared_num_att,
    # "kurtosis_squared_num_att": kurtosis_squared_num_att,
    # "skewness_squared_num_att": skewness_squared_num_att,

    "means_internal_product_num_att": means_internal_product_num_att,
    "std_internal_product_num_att": std_internal_product_num_att,
    # "kurtosis_internal_product_num_att": kurtosis_internal_product_num_att,
    # "skewness_internal_product_num_att": skewness_internal_product_num_att,
    
    "covariance": covariance,
    # "euclidean_distance_distribution": euclidean_distance_distribution,
    # "cosine_distance_distribution": cosine_distance_distribution,
    # "canberra_distance_distribution": canberra_distance_distribution,
    # "lorentzian_distance_distribution": canberra_distance_distribution,
    # "divergence_distance_distribution": canberra_distance_distribution,
    # "k_means_score": k_means_score,
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
        "means_num_att", "std_num_att", "kurtosis_num_att", "skewness_num_att",
        "means_squared_num_att", "std_squared_num_att", "kurtosis_squared_num_att", "skewness_squared_num_att",
        "means_internal_product_num_att", "std_internal_product_num_att", "kurtosis_internal_product_num_att", "skewness_internal_product_num_att",
        "covariance",
        "card_cat_att", "entropy_cat_att", "mutual_info_cat_att", "std_freq_cat_att"
        ]:
        return [f"{p}_{name}" for p in ["min", "q1", "mean", "q3", "max"]]
    elif name=="euclidean_distance_distribution":
        return [f"euclidean_{i}" for i in range(10)]
    elif name=="cosine_distance_distribution":
        return [f"cosine_{i}" for i in range(10)]
    elif name=="canberra_distance_distribution":
        return [f"canberra_{i}" for i in range(10)]
    elif name=="lorentzian_distance_distribution":
        return [f"lorentzian_{i}" for i in range(10)]
    elif name=="divergence_distance_distribution":
        return [f"divergence_{i}" for i in range(10)]
    elif name=="isolation_forest":
        return [f"isolation_forest_{i}" for i in range(10)]
    return

if __name__=="__main__":
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import minmax_scale
    X, y = make_blobs(n_samples=500, centers=5, n_features=8)
    X = minmax_scale(X)
    print(isolation_forest(Xnum=X, Xcat=y))