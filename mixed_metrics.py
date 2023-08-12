import abc
import base_metrics
import re
import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import OneHotEncoder

def get_valid_similarity_pairs(Xnum, Xcat):
    enc = OneHotEncoder(handle_unknown='ignore')
    Xdummy = enc.fit_transform(Xcat).toarray()
    l = []
    for num_metric in base_metrics.get_available_metrics():
        m = base_metrics.get_metric(num_metric)
        if m.is_valid_data(Xnum):
            ok = True
            if num_metric=="mahalanobis":
                try:
                    m.fit(Xnum)
                except:
                    ok = False
            if ok:
                for cat_metric in base_metrics.get_available_metrics(data_type="categorical"):
                    if base_metrics.get_metric(cat_metric).is_valid_data(Xcat):
                        l.append(f"{num_metric}_{cat_metric}")
                for bin_metric in base_metrics.get_available_metrics(data_type="binary"):
                    if base_metrics.get_metric(bin_metric).is_valid_data(Xdummy):
                        l.append(f"{num_metric}_{bin_metric}")
    return l


class MixedMetric(abc.ABC):
    @abc.abstractmethod
    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike, categorical=None) -> float:
        pass

    @abc.abstractmethod
    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None, categorical=None) -> npt.NDArray:
        pass

    def fit(self, X: npt.NDArray, categorical=None):
        return self

    def is_valid_data(self, x):
        return True
    
class WeightedAverage(MixedMetric):
    def __init__(self, pair_name=None, w=0.5, num_metric="euclidean", cat_metric="hamming", num_metric_params={}, cat_metric_params={}) -> None:
        super().__init__()
        if pair_name is not None:
            if re.match(r"[a-z\-]+_[a-z\-]", pair_name) is None:
                raise(Exception(f"Not known similarity measures pair ({pair_name})"))
            num_metric, cat_metric = pair_name.split("_")
            if num_metric not in base_metrics.get_available_metrics():
                raise(Exception(f"Not known numeric similarity measure ({num_metric}) in similarity measures pair ({pair_name})"))
            if cat_metric not in base_metrics.get_available_metrics(data_type="categorical") +\
                base_metrics.get_available_metrics(data_type="binary"):
                raise(Exception(f"Not known categorical similarity measure ({cat_metric}) in similarity measures pair ({pair_name})"))
            self.num_metric = base_metrics.get_metric(metric=num_metric, **num_metric_params)
            self.cat_metric = base_metrics.get_metric(metric=cat_metric, **cat_metric_params)
        else:
            if num_metric not in base_metrics.get_available_metrics():
                raise(Exception(f"Not known numeric similarity measure ({num_metric})"))
            if cat_metric not in base_metrics.get_available_metrics(data_type="categorical") +\
                base_metrics.get_available_metrics(data_type="binary"):
                raise(Exception(f"Not known categorical similarity measure ({cat_metric})"))
            self.num_metric = base_metrics.get_metric(metric=num_metric, **num_metric_params)
            self.cat_metric = base_metrics.get_metric(metric=cat_metric, **cat_metric_params)
        self.w = w

    def fit(self, X, categorical=None):
        numeric = [i for i in range(X.shape[1]) if i not in categorical]
        self.num_metric = self.num_metric.fit(X[:, numeric])
        self.cat_metric = self.cat_metric.fit(X[:, categorical])
        return self
    
    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike, categorical=None) -> float:
        numeric = [i for i in range(len(x)) if i not in categorical]
        return (1-self.w)*self.num_metric.dist(x[numeric], y[numeric]) + \
            self.w*self.cat_metric.dist(x[categorical], y[categorical])
    
    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None, categorical=None) -> npt.NDArray:
        numeric = [i for i in range(X.shape[1]) if i not in categorical]
        return (1-self.w)*self.num_metric.pairwise(X[:, numeric], Y if Y is None else Y[:, numeric]) + \
            self.w*self.cat_metric.pairwise(X[:, categorical], Y if Y is None else Y[:, categorical])
    
    def is_valid_data(self, x, categorical=None):
        x = np.array(x)
        if len(x.shape) == 1:
            numeric = [i for i in range(len(x)) if i not in categorical]
            return self.num_metric.is_valid_data(x[numeric]) and self.num_metric.is_valid_data(x[categorical])
        else:
            numeric = [i for i in range(x.shape[1]) if i not in categorical]
            return self.num_metric.is_valid_data(x[:, numeric]) and self.num_metric.is_valid_data(x[:, categorical])
        
if __name__ == "__main__":
    Xnum = np.random.rand(200, 10)
    Xcat = np.random.randint(8, size=(200, 5))
    X = np.c_[Xnum, Xcat]
    print(WeightedAverage().pairwise(X, categorical=np.arange(Xnum.shape[1], X.shape[1])))