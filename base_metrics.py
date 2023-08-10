"Module that implement similarity measures"

import numpy.typing as npt
import abc
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import pairwise_distances


class Metric():
    @abc.abstractmethod
    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        pass

    @abc.abstractmethod
    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        pass

    def fit(self, X):
        return self

    def is_valid_data(self, x):
        return True

    def flex(self, x, y, **__) -> npt.NDArray:
        if len(x.shape) == 1 and len(y.shape) == 1:
            return self.dist(x, y)
        elif len(x.shape) == 1 and len(y.shape) > 1:
            return self.pairwise(np.reshape(x, (1, len(x))), y).flatten()
        elif len(x.shape) > 1 and len(y.shape) == 1:
            return self.pairwise(x, np.reshape(y, (1, len(y)))).flatten()
        else:
            return self.pairwise(x, y)


class EuclideanDistance(Metric):

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        return distance.euclidean(x, y)

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        return pairwise_distances(X, Y, metric="euclidean", n_jobs=n_jobs)


class ManhattanDistance(Metric):
    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        return distance.minkowski(x, y, p=1)

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        return pairwise_distances(X, Y, metric="manhattan", n_jobs=n_jobs)


class ChebyshevDistance(Metric):

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        return distance.chebyshev(x, y)

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        return pairwise_distances(X, Y, metric="chebyshev", n_jobs=n_jobs)


class SqeuclideanDistance(Metric):

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        return distance.sqeuclidean(x, y)

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        return pairwise_distances(X, Y, metric="sqeuclidean", n_jobs=n_jobs)

class CanberraDistance(Metric):

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        return distance.canberra(x, y)

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        return pairwise_distances(X, Y, metric="canberra", n_jobs=n_jobs)


class MahalanobisDistance(Metric):

    def __init__(self, V=None, VI=None):
        self.VI = VI
        if V is not None and VI is None:
            self.VI = np.linalg.inv(V)

    def fit(self, X):
        if X.shape[1] > 1:
            self.VI = np.linalg.inv(np.cov(X, rowvar=False)) 
        else:
            self.VI = 1/np.cov(X, rowvar=False)
        return self

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        if self.VI is None: 
            raise(Exception("Metric not fitted yet"))
        return distance.mahalanobis(x, y, VI=self.VI)

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        if self.VI is None: 
            raise(Exception("Metric not fitted yet"))
        return pairwise_distances(X, Y, metric="mahalanobis", n_jobs=n_jobs, VI=self.VI)


class CosineDistance(Metric):

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        return distance.cosine(x, y)

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        return pairwise_distances(X, Y, metric="cosine", n_jobs=n_jobs)


class PearsonDistance(Metric):

    def is_valid_data(self, x):
        x = np.array(x)
        if len(x.shape) == 1:
            return len(x) >= 2 and (x!=x[0]).any()
        else:
            if x.shape[1] <= 1: return False
            for xi in x:
                if (xi==xi[0]).all(): return False
            return True

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        return distance.correlation(x, y)

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        return pairwise_distances(X, Y, metric="correlation", n_jobs=n_jobs)


class LorentzianDistance(Metric):

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        return np.sum(np.log(1 + np.abs(np.array(x) - np.array(y))))

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        return pairwise_distances(X, Y=Y, metric=self.dist, n_jobs=n_jobs)


class DivergenceDistance(Metric):

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        x = np.array(x)
        y = np.array(y)
        xplusy = x + y
        return 2 * np.sum((x[xplusy != 0] - y[xplusy != 0])**2/xplusy[xplusy != 0]**2)

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        return pairwise_distances(X, Y=Y, metric=self.dist, n_jobs=n_jobs)

class HammingDistance(Metric):

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        return distance.hamming(x, y)

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        return pairwise_distances(X, Y=Y, metric="hamming", n_jobs=n_jobs)


class EskinDistance(Metric):

    def __init__(self, n_cat=None):
        super().__init__()
        self.n_cat = n_cat
        if n_cat is not None:
            self.n_cat2 = np.array(n_cat)**2

    def fit(self, X):
        X = np.array(X, dtype=int)
        self.n_cat = []
        for j in range(X.shape[1]):
            self.n_cat.append(len(set(X[:,j])))
        self.n_cat2 = np.array(self.n_cat)**2
        return self

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=int)
        if self.n_cat is None:
            raise(Exception("Metric not fitted yet"))
        return len(x)/sum([1 if xi == y[i] else self.n_cat2[i]/(self.n_cat2[i] + 2) for i, xi in enumerate(x)]) - 1

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        X = np.array(X, dtype=int)
        if Y is not None: Y = np.array(Y, dtype=int)
        if self.n_cat is None:
            raise(Exception("Metric not fitted yet"))
        return pairwise_distances(X, Y, metric=self.dist, n_jobs=n_jobs)


class IOFDistance(Metric):

    def __init__(self, f=None):
        super().__init__()
        self.f = f

    def fit(self, X):
        X = np.array(X, dtype=int)
        self.X = X.copy()
        self.f = []        
        for j in range(X.shape[1]):
            self.f.append(dict(zip(*np.unique(X[:,j], return_counts=True))))
        return self

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=int)
        if self.f is None:
            raise(Exception("Metric not fitted yet"))
        sims = np.array([1 if xi == y[i] else 1 / (1 + np.log(self.f[i]
                        [xi]) + np.log(self.f[i][y[i]])) for i, xi in enumerate(x)])
        s = sum(sims)/len(x)
        return 1/s - 1

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        X = np.array(X, dtype=int)
        if Y is not None: Y = np.array(Y, dtype=int)
        # print(self.f)
        if self.f is None:
            raise(Exception("Metric not fitted yet"))
        return pairwise_distances(X, Y, metric=self.dist, n_jobs=n_jobs)


class OFDistance(Metric):

    def __init__(self, f=None, N=None):
        self.f = f
        self.N = N

    def fit(self, X):
        X = np.array(X, dtype=int)
        self.f = []        
        for j in range(X.shape[1]):
            self.f.append(dict(zip(*np.unique(X[:,j], return_counts=True))))
        self.N = X.shape[0]
        return self

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=int)
        if self.f is None or self.N is None:
            raise(Exception("Metric not fitted yet"))
        sims = np.array([1 if xi == y[i] else 1 / (1 + np.log(self.N/self.f[i]
                        [xi]) + np.log(self.N/self.f[i][y[i]])) for i, xi in enumerate(x)])
        s = sum(sims)/len(x)
        return 1/s - 1

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        X = np.array(X, dtype=int)
        if Y is not None: Y = np.array(Y, dtype=int)
        if self.f is None or self.N is None:
            raise(Exception("Metric not fitted yet"))
        return pairwise_distances(X, Y, metric=self.dist, n_jobs=n_jobs)


class CoocDistance(Metric):

    def __init__(self):
        self.d = None

    def is_valid_data(self, x):
        x = np.array(x)
        if len(x.shape) == 1:
            return len(x) >= 2
        return x.shape[1] >= 2

    def fit(self, U):
        U = np.array(U, dtype=int)
        if not self.is_valid_data(U):
            raise(Exception(f"Input data should have at least two dimensions, {U.shape[1]} found"))
        self.U = U
        self.d = {i: {} for i in range(U.shape[1])}
        for i in range(U.shape[1]):
            values = list(set(U[:, i]))
            for x in values:
                self.d[i][x] = {}
                self.d[i][x][x] = 0
            for k in range(len(values) - 1):
                x = values[k]
                for l in range(k + 1, len(values)):
                    y = values[l]
                    s = 0
                    for j in range(U.shape[1]):
                        if i != j:
                            s += self.find_max(x, y, U[:, i], U[:, j])
                    self.d[i][x][y] = s/(U.shape[1] - 1)
                    self.d[i][y][x] = self.d[i][x][y]
        return self

    def find_max(self, x, y, Ai, Aj):
        dij = 0
        indexes_of_x = [i for i, val in enumerate(Ai) if val == x]
        indexes_of_y = [i for i, val in enumerate(Ai) if val == y]
        n_x = len(indexes_of_x)
        n_y = len(indexes_of_y)
        for u in set(Aj):
            p_u_x = len(list(filter(lambda v: v == u, Aj[indexes_of_x])))/n_x
            p_u_y = len(list(filter(lambda v: v == u, Aj[indexes_of_y])))/n_y
            if p_u_x >= p_u_y:
                dij += p_u_x
            else:
                dij += p_u_y
        return dij - 1

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=int)
        if self.d == None: 
            raise(Exception("Metric not fitted yet"))
        d_ = np.array([self.d[i][xi][y[i]] for i, xi in enumerate(x)])
        return sum(d_)/len(x)

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        X = np.array(X, dtype=int)
        if Y is not None: Y = np.array(Y, dtype=int)
        # print(self.d)
        if self.d == None: 
            raise(Exception("Metric not fitted yet"))
        return pairwise_distances(X, Y, metric=self.dist, n_jobs=n_jobs)


class BinJaccardDistance(Metric):

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=int)
        if ((x == 0) | (x == 1)).all() and ((y == 0) | (y == 1)).all():
            return distance.jaccard(x, y)
        raise(Exception("Input data contain non binary values"))

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        X = np.array(X, dtype=int)
        if Y is not None: Y = np.array(Y, dtype=int)
        if ((X == 0) | (X == 1)).all() and (Y is None or ((Y == 0) | (Y == 1)).all()):
            return pairwise_distances(X==1, Y=Y if Y is None else (Y==1), metric="jaccard", n_jobs=n_jobs)
        else:
            raise(Exception("Input data contain non binary values"))

class DiceDistance(Metric):

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=int)
        if ((x == 0) | (x == 1)).all() and ((y == 0) | (y == 1)).all():
            return distance.dice(x, y)
        raise(Exception("Input data contain non binary values"))

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        X = np.array(X, dtype=int)
        if Y is not None: Y = np.array(Y, dtype=int)
        if ((X == 0) | (X == 1)).all() and (Y is None or ((Y == 0) | (Y == 1)).all()):
            return pairwise_distances(X==1, Y=Y if Y is None else (Y==1), metric="dice", n_jobs=n_jobs)
        raise(Exception("Input data contain non binary values"))


class KulsinskiDistance(Metric):

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=int)
        if ((x == 0) | (x == 1)).all() and ((y == 0) | (y == 1)).all():
            return distance.kulsinski(x, y)
        raise(Exception("Input data contain non binary values"))

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        X = np.array(X, dtype=int)
        if Y is not None: Y = np.array(Y, dtype=int)
        if ((X == 0) | (X == 1)).all() and (Y is None or ((Y == 0) | (Y == 1)).all()):
            return pairwise_distances(X==1, Y=Y if Y is None else (Y==1), metric="kulsinski", n_jobs=n_jobs)
        raise(Exception("Input data contain non binary values"))


class RogerstanimotoDistance(Metric):

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=int)
        if ((x == 0) | (x == 1)).all() and ((y == 0) | (y == 1)).all():
            return distance.rogerstanimoto(x, y)
        raise(Exception("Input data contain non binary values"))

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        X = np.array(X, dtype=int)
        if Y is not None: Y = np.array(Y, dtype=int)
        if ((X == 0) | (X == 1)).all() and (Y is None or ((Y == 0) | (Y == 1)).all()):
            return pairwise_distances(X==1, Y=Y if Y is None else (Y==1), metric="rogerstanimoto", n_jobs=n_jobs)
        raise(Exception("Input data contain non binary values"))


class RussellraoDistance(Metric):

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=int)
        if ((x == 0) | (x == 1)).all() and ((y == 0) | (y == 1)).all():
            return distance.russellrao(x, y)
        raise(Exception("Input data contain non binary values"))

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        X = np.array(X, dtype=int)
        if Y is not None: Y = np.array(Y, dtype=int)
        if ((X == 0) | (X == 1)).all() and (Y is None or ((Y == 0) | (Y == 1)).all()):
            return pairwise_distances(X==1, Y=Y if Y is None else (Y==1), metric="russellrao", n_jobs=n_jobs)
        raise(Exception("Input data contain non binary values"))

class SokalmichenerDistance(Metric):

    def __init__(self, **_):
        self.enc = OneHotEncoder(handle_unknown='ignore')

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=int)
        if ((x == 0) | (x == 1)).all() and ((y == 0) | (y == 1)).all():
            return distance.sokalmichener(x, y)
        raise(Exception("Input data contain non binary values"))

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        X = np.array(X, dtype=int)
        if Y is not None: Y = np.array(Y, dtype=int)
        if ((X == 0) | (X == 1)).all() and (Y is None or ((Y == 0) | (Y == 1)).all()):
            return pairwise_distances(X==1, Y=Y if Y is None else (Y==1), metric="sokalmichener", n_jobs=n_jobs)
        raise(Exception("Input data contain non binary values"))


class SokalsneathDistance(Metric):

    def __init__(self, **_):
        self.enc = OneHotEncoder(handle_unknown='ignore')

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=int)
        if ((x == 0) | (x == 1)).all() and ((y == 0) | (y == 1)).all():
            return distance.sokalsneath(x, y)
        raise(Exception("Input data contain non binary values"))

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        X = np.array(X, dtype=int)
        if Y is not None: Y = np.array(Y, dtype=int)
        if ((X == 0) | (X == 1)).all() and (Y is None or ((Y == 0) | (Y == 1)).all()):
            return pairwise_distances(X==1, Y=Y if Y is None else (Y==1), metric="sokalsneath", n_jobs=n_jobs)
        raise(Exception("Input data contain non binary values"))


def get_metric(metric="euclidean", **metric_params) -> Metric:
    if metric == "euclidean":
        return EuclideanDistance()
    if metric == "manhattan":
        return ManhattanDistance()
    if metric == "chebyshev":
        return ChebyshevDistance()
    if metric == "sqeuclidean":
        return SqeuclideanDistance()
    if metric == "canberra":
        return CanberraDistance()
    if metric == "mahalanobis":
        return MahalanobisDistance(**metric_params)
    if metric == "cosine":
        return CosineDistance()
    if metric == "pearson":
        return PearsonDistance()
    if metric == "lorentzian":
        return LorentzianDistance()
    if metric == "divergence":
        return DivergenceDistance()
    # if metric == "jensenshannon":
    #     return JensenshannonDistance()

    if metric == "hamming":
        return HammingDistance()
    if metric == "eskin":
        return EskinDistance(**metric_params)
    if metric == "iof":
        return IOFDistance(**metric_params)
    if metric == "of":
        return OFDistance(**metric_params)
    if metric == "co-oc":
        return CoocDistance(**metric_params)

    if metric == "jaccard":
        return BinJaccardDistance()
    if metric == "dice":
        return DiceDistance()
    if metric == "kulsinski":
        return KulsinskiDistance()
    if metric == "rogerstanimoto":
        return RogerstanimotoDistance()
    if metric == "russellrao":
        return RussellraoDistance()
    if metric == "sokalmichener":
        return SokalmichenerDistance()
    if metric == "sokalsneath":
        return SokalsneathDistance()
    raise(Exception(f"Not known metric : {metric}"))


def get_available_metrics(data_type="numeric"):
    if data_type == "numeric":
        return ["euclidean", "manhattan", "chebyshev", "sqeuclidean", "canberra", "mahalanobis", "cosine", "pearson", "lorentzian", "divergence"]
    if data_type == "categorical":
        return ["hamming", "eskin", "iof", "of", "co-oc"]
    if data_type == "binary":
        return ["jaccard", "dice", "kulsinski", "rogerstanimoto", "russellrao", "sokalmichener", "sokalsneath"]


# def get_available_metrics(data_type="numeric"):
#     if data_type == "numeric":
#         return ["euclidean", "manhattan", "sqeuclidean", "canberra", "mahalanobis", "pearson"]
#     if data_type == "categorical":
#         return ["hamming", "iof", "of", "co-oc"]
#     if data_type == "binary":
#         return ["jaccard", "sokalsneath"]

def get_metric_params(metric="euclidean") -> list:
    if metric == "mahalanobis":
        return ["VI"]

    if metric == "eskin":
        return ["n_cat"]

    if metric == "iof":
        return ["f"]

    if metric == "of":
        return ["N", "f"]

    if metric == "co-oc":
        return ["U"]
    return []
