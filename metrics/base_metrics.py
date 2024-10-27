"Module that implement similarity measures"

import numpy.typing as npt
import abc
import numpy as np
from scipy.spatial import distance
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import pairwise_distances
from scipy.stats import entropy
import math
import os
import pickle

# pylint: disable=assignment-from-no-return

class Metric():
    name: str
    def __init__(self, caching=False, cache_dir=None):
        self.caching = caching
        self.cache_dir = cache_dir
        if self.caching and self.cache_dir is None:
            self.cache_dir = "distance_caching_dir"
        self.current_dataset = None
        self.current_pairwise_distances_dataset = None
        self.pairwise_distances = None

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        pass

    def _pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        pass

    def pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None, dataset_name=None) -> npt.NDArray:
        if self.current_pairwise_distances_dataset is not None and \
            self.current_pairwise_distances_dataset == dataset_name:
            return self.pairwise_distances

        self.current_pairwise_distances_dataset = dataset_name

        if self.caching and self.cache_dir is not None and dataset_name is not None:
            if self.load_pairwise_distances(dataset_name): 
                return self.pairwise_distances

        self.pairwise_distances = self._pairwise(X, Y, n_jobs=n_jobs)

        if self.caching and self.cache_dir is not None and dataset_name is not None:
            self.save_pairwise_distances(dataset_name)
        
        return self.pairwise_distances

    def fit(self, X, dataset_name=None):
        return self

    def is_valid_data(self, x):
        return True

    def check_fitted(self): pass

    def flex(self, x, y, **__) -> npt.NDArray:
        if len(x.shape) == 1 and len(y.shape) == 1:
            return self.dist(x, y)
        elif len(x.shape) == 1 and len(y.shape) > 1:
            return self._pairwise(np.reshape(x, (1, len(x))), y).flatten()
        elif len(x.shape) > 1 and len(y.shape) == 1:
            return self._pairwise(x, np.reshape(y, (1, len(y)))).flatten()
        else:
            return self._pairwise(x, y)

    def save_infos(self, dataset_name): pass

    def load_infos(self, dataset_name): pass

    def save_pairwise_distances(self, dataset_name):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        filename = os.path.join(self.cache_dir, f"{self.name}_{dataset_name}_pairwise_distances.pickle")
        with open(filename, "wb") as fp:
            pickle.dump(self.pairwise_distances, fp)

    def load_pairwise_distances(self, dataset_name):
        filename = os.path.join(self.cache_dir, f"{self.name}_{dataset_name}_pairwise_distances.pickle")
        if os.path.isfile(filename):
            with open(filename, "rb") as fp:
                self.pairwise_distances = pickle.load(fp)
            return True
        return False


class EuclideanDistance(Metric):
    name = "euclidean"
    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        return distance.euclidean(x, y)

    def _pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        return pairwise_distances(X, Y, metric="euclidean")    #, n_jobs=n_jobs


class ManhattanDistance(Metric):
    name = "manhattan"
    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        return distance.minkowski(x, y, p=1)

    def _pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        return pairwise_distances(X, Y, metric="manhattan", n_jobs=n_jobs)


class ChebyshevDistance(Metric):
    name = "chebishev"
    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        return distance.chebyshev(x, y)

    def _pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        return pairwise_distances(X, Y, metric="chebyshev", n_jobs=n_jobs)


class SqeuclideanDistance(Metric):
    name = "sqeuclidean"
    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        return distance.sqeuclidean(x, y)

    def _pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        return pairwise_distances(X, Y, metric="sqeuclidean")    #, n_jobs=n_jobs

class CanberraDistance(Metric):
    name = "canberra"
    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        return distance.canberra(x, y)

    def _pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        return pairwise_distances(X, Y, metric="canberra", n_jobs=n_jobs)


class MahalanobisDistance(Metric):
    name = "mahalanobis"
    def __init__(self, caching=False, cache_dir=None):
        super().__init__(caching=caching, cache_dir=cache_dir)
        self.VI = None

    def fit(self, X, dataset_name=None):
        if self.current_dataset is not None and self.current_dataset == dataset_name:
            return self
        self.current_dataset = dataset_name

        if self.caching and self.cache_dir is not None and dataset_name is not None:
            if self.load_infos(dataset_name): return self

        if X.shape[1] > 1:
            self.VI = np.linalg.inv(np.cov(X, rowvar=False)) 
        else:
            self.VI = 1/np.cov(X, rowvar=False)

        if self.caching and self.cache_dir is not None and dataset_name is not None:
            self.save_infos(dataset_name)
        return self

    def check_fitted(self):
        if self.VI is None: 
            raise(Exception("Metric not fitted yet"))


    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        self.check_fitted()
        return distance.mahalanobis(x, y, VI=self.VI)

    def _pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        self.check_fitted()
        return pairwise_distances(X, Y, metric="mahalanobis", VI=self.VI, n_jobs=n_jobs)
    
    def save_infos(self, dataset_name):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        filename = os.path.join(self.cache_dir, f"{self.name}_{dataset_name}_infos.pickle")
        with open(filename, "wb") as fp:
            pickle.dump({"VI": self.VI}, fp)
    
    def load_infos(self, dataset_name):
        filename = os.path.join(self.cache_dir, f"{self.name}_{dataset_name}_infos.pickle")
        if os.path.isfile(filename):
            with open(filename, "rb") as fp:
                obj = pickle.load(fp)
            for k, v in obj.items():
                self.__setattr__(k, v)
            return True
        return False


class CosineDistance(Metric):
    name = "cosine"
    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        return distance.cosine(x, y)

    def _pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        return pairwise_distances(X, Y, metric="cosine")    #, n_jobs=n_jobs


class PearsonDistance(Metric):
    name = "pearson"
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

    def _pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        return pairwise_distances(X, Y, metric="correlation", n_jobs=n_jobs)


class LorentzianDistance(Metric):
    name = "lorentzian"
    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        return np.sum(np.log(1 + np.abs(np.array(x) - np.array(y))))

    def _pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        return pairwise_distances(X, Y=Y, metric=self.dist, n_jobs=n_jobs)


class DivergenceDistance(Metric):
    name = "divergence"
    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        x = np.array(x)
        y = np.array(y)
        xplusy = x + y
        return 2 * np.sum((x[xplusy != 0] - y[xplusy != 0])**2/xplusy[xplusy != 0]**2)

    def _pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        return pairwise_distances(X, Y=Y, metric=self.dist, n_jobs=n_jobs)


class CategoricalDistance(Metric):

    def __init__(self, caching=False, cache_dir=None):
        super().__init__(caching=caching, cache_dir=cache_dir)
        self.per_attribute_dissimilarity_matrices = []

    def check_fitted(self):
        if not self.per_attribute_dissimilarity_matrices:
            raise(Exception("Metric not fitted yet"))

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        self.check_fitted()
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=int)
        return sum([self.per_attribute_dissimilarity_matrices[j][xj, y[j]] for j, xj in enumerate(x)])

    def _pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        self.check_fitted()
        X = np.array(X, dtype=int)
        if Y is not None: Y = np.array(Y, dtype=int)
        return pairwise_distances(X, Y, metric=self.dist, n_jobs=n_jobs)
    
    def save_infos(self, dataset_name):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        filename = os.path.join(self.cache_dir, f"{self.name}_{dataset_name}_infos.pickle")
        with open(filename, "wb") as fp:
            pickle.dump({"per_attribute_dissimilarity_matrices": self.per_attribute_dissimilarity_matrices}, fp)
    
    def load_infos(self, dataset_name):
        filename = os.path.join(self.cache_dir, f"{self.name}_{dataset_name}_infos.pickle")
        if os.path.isfile(filename):
            with open(filename, "rb") as fp:
                obj = pickle.load(fp)
            for k, v in obj.items():
                self.__setattr__(k, v)
            return True
        return False

class HammingDistance(CategoricalDistance):
    name = "hamming"
    def fit(self, X, dataset_name=None):
        if self.current_dataset is not None and self.current_dataset == dataset_name:
            return self
        self.current_dataset = dataset_name

        if self.caching and self.cache_dir is not None and dataset_name is not None:
            if self.load_infos(dataset_name): return self

        X = np.array(X, dtype=int)
        self.per_attribute_dissimilarity_matrices = []
        for j in range(X.shape[1]):
            self.per_attribute_dissimilarity_matrices.append(1 - np.identity(len(np.unique(X[:,j]))))

        if self.caching and self.cache_dir is not None and dataset_name is not None:
            self.save_infos(dataset_name)
        return self

    # def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
    #     return distance.hamming(x, y)*len(x)

    # def _pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
    #     return pairwise_distances(X, Y=Y, metric="hamming", n_jobs=n_jobs)*X.shape[1]


class EskinDistance(CategoricalDistance):
    name = "eskin"
    def __init__(self, caching=False, cache_dir=None):
        super().__init__(caching=caching, cache_dir=cache_dir)
        self.n_cat = None
        self.n_cat2 = None

    def fit(self, X, dataset_name=None):
        if self.current_dataset is not None and self.current_dataset == dataset_name:
            return self
        self.current_dataset = dataset_name

        if self.caching and self.cache_dir is not None and dataset_name is not None:
            if self.load_infos(dataset_name): return self

        X = np.array(X, dtype=int)
        self.n_cat = []
        for j in range(X.shape[1]):
            self.n_cat.append(len(set(X[:,j])))
        self.n_cat2 = np.array(self.n_cat)**2
        
        self.per_attribute_dissimilarity_matrices = []
        for j in range(X.shape[1]):
            self.per_attribute_dissimilarity_matrices.append((1 - np.identity(len(np.unique(X[:,j]))))*2/self.n_cat2[j])

        if self.caching and self.cache_dir is not None and dataset_name is not None:
            self.save_infos(dataset_name)
        return self


class IOFDistance(CategoricalDistance):
    name = "iof"
    def __init__(self, caching=False, cache_dir=None):
        super().__init__(caching=caching, cache_dir=cache_dir)
        self.f = None

    def fit(self, X, dataset_name=None):
        if self.current_dataset is not None and self.current_dataset == dataset_name:
            return self
        self.current_dataset = dataset_name

        if self.caching and self.cache_dir is not None and dataset_name is not None:
            if self.load_infos(dataset_name): return self

        X = np.array(X, dtype=int)
        self.f = []
        self.per_attribute_dissimilarity_matrices = []
        for j in range(X.shape[1]):
            Xj, counts = np.unique(X[:,j], return_counts=True)
            self.f.append(dict(zip(Xj, counts)))

            dissimilarity_matrix = np.zeros((len(Xj), len(Xj)))
            for k, u in enumerate(Xj[:-1]):
                for v in Xj[k+1:]:
                    dissimilarity_matrix[u, v] = dissimilarity_matrix[v, u] = math.log(self.f[j][u]) * math.log(self.f[j][v])
            self.per_attribute_dissimilarity_matrices.append(dissimilarity_matrix)

        if self.caching and self.cache_dir is not None and dataset_name is not None:
            self.save_infos(dataset_name)
        return self


class OFDistance(CategoricalDistance):
    name = "of"
    def __init__(self, caching=False, cache_dir=None):
        super().__init__(caching=caching, cache_dir=cache_dir)
        self.f = None
        self.N = None

    def fit(self, X, dataset_name=None):
        if self.current_dataset is not None and self.current_dataset == dataset_name:
            return self
        self.current_dataset = dataset_name
        
        if self.caching and self.cache_dir is not None and dataset_name is not None:
            if self.load_infos(dataset_name): return self

        X = np.array(X, dtype=int)
        self.N = X.shape[0]
        self.f = []
        self.per_attribute_dissimilarity_matrices = []
        for j in range(X.shape[1]):
            Xj, counts = np.unique(X[:,j], return_counts=True)
            self.f.append(dict(zip(Xj, counts)))

            dissimilarity_matrix = np.zeros((len(Xj), len(Xj)))
            for k, u in enumerate(Xj[:-1]):
                for v in Xj[k+1:]:
                    dissimilarity_matrix[u, v] = dissimilarity_matrix[v, u] = math.log(self.N/self.f[j][u]) * math.log(self.N/self.f[j][v])
            self.per_attribute_dissimilarity_matrices.append(dissimilarity_matrix)

        if self.caching and self.cache_dir is not None and dataset_name is not None:
            self.save_infos(dataset_name)
        return self


class CoocDistance(CategoricalDistance):
    name = "co-oc"
    def is_valid_data(self, x):
        x = np.array(x)
        if len(x.shape) == 1:
            return len(x) >= 2
        return x.shape[1] >= 2

    def fit(self, X, dataset_name=None):
        if self.current_dataset is not None and self.current_dataset == dataset_name:
            return self
        self.current_dataset = dataset_name
        
        if self.caching and self.cache_dir is not None and dataset_name is not None:
            if self.load_infos(dataset_name): return self

        X = np.array(X, dtype=int)
        if not self.is_valid_data(X):
            raise(Exception(f"Input data should have at least two dimensions, {X.shape[1]} found"))
        self.per_attribute_dissimilarity_matrices = []
        for j in range(X.shape[1]):
            Xj = np.unique(X[:,j])
            dissimilarity_matrix = np.zeros((len(Xj), len(Xj)))
            for k, u in enumerate(Xj[:-1]):
                for v in Xj[k+1:]:
                    d = 0
                    for l in range(X.shape[1]):
                        if j != l:
                            d += self.find_max(u, v, X[:, j], X[:, l])
                    d /= X.shape[1] - 1
                    dissimilarity_matrix[u, v] = dissimilarity_matrix[v, u] = d
            self.per_attribute_dissimilarity_matrices.append(dissimilarity_matrix)

        if self.caching and self.cache_dir is not None and dataset_name is not None:
            self.save_infos(dataset_name)
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

class DILCA(CategoricalDistance):
    name = "dilca"
    def is_valid_data(self, x):
        x = np.array(x)
        if len(x.shape) == 1:
            return len(x) >= 2
        return x.shape[1] >= 2
        
    def ProconditionMatrixYX(self, Y, X, Y_unique, X_unique):
        lenData = len(Y)
        lenX = len(X_unique)
        lenY = len(Y_unique)
        HY = 0
        count_x = 0;
        count_y=0;
        MATRIX = [[0 for i in range(lenX)] for j in  range(lenY)]
        for x in range(lenX):
            for y in range(lenY):
                count_x = count_y =0
                for i in range(lenData):
                    if X[i] == x:
                        count_x = count_x+1
                        if(Y[i] == y):
                            count_y = count_y+1;
                MATRIX[y][x] = count_y/count_x if count_x > 0 else 0
                asd = 12332
        return MATRIX

    def compute_per_attribute_dissimilarity_matrices(self, X):
        N = len(X)
        D = len(X[0])
        context_array = [[] for _ in range(D)]
        self.contextMatrix = []
        self.conditionProMatrix= {}
        self.probabilityMatrix = []
        #Compute cross contextMatrix
        max_columns = []
        for i in range(D):
            Y_ = X[:,i]
            Y_unique, Y_freq = np.unique(Y_, return_counts=True)
            max_columns.append(max(Y_unique))
            X_probability  = [ii/len(Y_) for ii in Y_freq]
            self.probabilityMatrix.append(X_probability)
            HY = entropy(X_probability,base=2)
            SUY_dict = {}
            for j in range(D):
                if i !=j:
                    X_ = X[:,j]
                    X_unique, X_freq = np.unique(X_, return_counts=True)
                    X_property = [ii/len(X_) for ii in X_freq]
                    HX = entropy(X_property,base=2)
                    conditionMatrix = self.ProconditionMatrixYX(Y_,X_,Y_unique, X_unique)
                    self.conditionProMatrix[(i,j)] = conditionMatrix
                    HYX = 0
                    for k in range(len(X_unique)):
                        sum_tmp =0;
                        for k2 in range(len(Y_unique)):
                            if conditionMatrix[k2][k] !=0:
                                sum_tmp = sum_tmp +  conditionMatrix[k2][k] * math.log2(conditionMatrix[k2][k])
                        HYX = HYX + sum_tmp*X_property[k]
                    HYX= -HYX 
                    IGYX = HY - HYX
                    if(HX + HY ==0):
                        SUYX=0
                    else:
                        SUYX = 2*IGYX/(HY + HX)
                    SUY_dict[j] = SUYX
            values = list(SUY_dict.values())
            o = 1
            mean = np.mean(values)
            context_Y = [ key for (key,value) in  SUY_dict.items() if value >= o*mean]
            self.contextMatrix.append(context_Y)
        #Compute distMatrix
        self.per_attribute_dissimilarity_matrices = [];
        for d in range(D):
            matrix = [] # 2D array for 1 dimension
            for i in range(max_columns[d]+1):
                matrix_tmp = []; #1D array for 1 values on the attribute d
                for j in range(max_columns[d]+1):
                    dist_sum_all =0;
                    dist_sum = 0;
                    dist_sum2 = 0;
                    for d2 in self.contextMatrix[d]:
                        dist_sum_tmp =0;
                        conditionMatrix = self.conditionProMatrix[(d,d2)]
                        for i_k in range(max_columns[d2]):
                            dist_sum_tmp2 =  (conditionMatrix[i][i_k] -  conditionMatrix[j][i_k])**2
                            dist_sum_tmp = dist_sum_tmp +  dist_sum_tmp2*self.probabilityMatrix[d2][i_k];
                        dist_sum = dist_sum+ dist_sum_tmp
                        dist_sum2 = dist_sum2 + max_columns[d2]+1
                    if dist_sum2==0: # toanstt 
                        dist_sum2=1;
                    dist_sum_all = np.sqrt(dist_sum/dist_sum2)
                    matrix_tmp.append(dist_sum_all)
                matrix.append(matrix_tmp)
            self.per_attribute_dissimilarity_matrices.append(np.array(matrix))

    def fit(self, X, dataset_name=None):
        if self.current_dataset is not None and self.current_dataset == dataset_name:
            return self
        self.current_dataset = dataset_name
        
        if self.caching and self.cache_dir is not None and dataset_name is not None:
            if self.load_infos(dataset_name): return self

        X = np.array(X, dtype=int)
        if not self.is_valid_data(X):
            raise(Exception(f"Input data should have at least two dimensions, {X.shape[1]} found"))
        self.compute_per_attribute_dissimilarity_matrices(X)

        if self.caching and self.cache_dir is not None and dataset_name is not None:
            self.save_infos(dataset_name)
        return self


class BinaryDistance(Metric):

    def check(self, x: npt.ArrayLike):
        x = np.array(x, dtype=int)
        if ((x != 0) & (x != 1)).any():
            raise(Exception("Input data contain non binary values"))
        return x==1 # conversion to boolean to avoid warning from sklearn

class BinJaccardDistance(BinaryDistance):
    name = "jaccard"

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        x = self.check(x)
        y = self.check(y)
        return distance.jaccard(x, y)

    def _pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        X = self.check(X)
        if Y is not None: Y = self.check(Y)
        return pairwise_distances(X, Y, metric="jaccard", n_jobs=n_jobs)

class DiceDistance(BinaryDistance):
    name = "dice"

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        x = self.check(x)
        y = self.check(y)
        return distance.dice(x, y)

    def _pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        X = self.check(X)
        if Y is not None: Y = self.check(Y)
        return pairwise_distances(X, Y, metric="dice", n_jobs=n_jobs)


class KulsinskiDistance(BinaryDistance):
    name = "kulsinski"

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        x = self.check(x)
        y = self.check(y)
        return distance.kulsinski(x, y)

    def _pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        X = self.check(X)
        if Y is not None: Y = self.check(Y)
        return pairwise_distances(X, Y , metric="kulsinski", n_jobs=n_jobs)


class RogerstanimotoDistance(BinaryDistance):
    name = "rogerstanimoto"

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        x = self.check(x)
        y = self.check(y)
        return distance.rogerstanimoto(x, y)

    def _pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        X = self.check(X)
        if Y is not None: Y = self.check(Y)
        return pairwise_distances(X, Y, metric="rogerstanimoto", n_jobs=n_jobs)


class RussellraoDistance(BinaryDistance):
    name = "russellrao"

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        x = self.check(x)
        y = self.check(y)
        return distance.russellrao(x, y)

    def _pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        X = self.check(X)
        if Y is not None: Y = self.check(Y)
        return pairwise_distances(X, Y, metric="russellrao", n_jobs=n_jobs)

class SokalmichenerDistance(BinaryDistance):
    name = "sokalmichener"

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        x = self.check(x)
        y = self.check(y)
        return distance.sokalmichener(x, y)

    def _pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        X = self.check(X)
        if Y is not None: Y = self.check(Y)
        return pairwise_distances(X, Y, metric="sokalmichener", n_jobs=n_jobs)


class SokalsneathDistance(BinaryDistance):
    name = "sokalsneath"

    def dist(self, x: npt.ArrayLike, y: npt.ArrayLike) -> float:
        x = self.check(x)
        y = self.check(y)
        return distance.sokalsneath(x, y)

    def _pairwise(self, X: npt.NDArray, Y: npt.NDArray = None, n_jobs=None) -> npt.NDArray:
        X = self.check(X)
        if Y is not None: Y = self.check(Y)
        return pairwise_distances(X, Y, metric="sokalsneath", n_jobs=n_jobs)


def get_metric(metric="euclidean", **metric_params) -> Metric:
    if metric.lower() == "euclidean":
        return EuclideanDistance(**metric_params)
    if metric.lower() == "manhattan":
        return ManhattanDistance(**metric_params)
    if metric.lower() == "chebyshev":
        return ChebyshevDistance(**metric_params)
    if metric.lower() == "sqeuclidean":
        return SqeuclideanDistance(**metric_params)
    if metric.lower() == "canberra":
        return CanberraDistance(**metric_params)
    if metric.lower() == "mahalanobis":
        return MahalanobisDistance(**metric_params)
    if metric.lower() == "cosine":
        return CosineDistance(**metric_params)
    if metric.lower() == "pearson":
        return PearsonDistance(**metric_params)
    if metric.lower() == "lorentzian":
        return LorentzianDistance(**metric_params)
    if metric.lower() == "divergence":
        return DivergenceDistance(**metric_params)
    # if metric.lower() == "jensenshannon":
    #     return JensenshannonDistance()

    if metric.lower() == "hamming":
        return HammingDistance(**metric_params)
    if metric.lower() == "eskin":
        return EskinDistance(**metric_params)
    if metric.lower() == "iof":
        return IOFDistance(**metric_params)
    if metric.lower() == "of":
        return OFDistance(**metric_params)
    if metric.lower() == "co-oc":
        return CoocDistance(**metric_params)
    if metric.lower() == "dilca":
        return DILCA(**metric_params)

    if metric.lower() == "jaccard":
        return BinJaccardDistance(**metric_params)
    if metric.lower() == "dice":
        return DiceDistance(**metric_params)
    if metric.lower() == "kulsinski":
        return KulsinskiDistance(**metric_params)
    if metric.lower() == "rogerstanimoto":
        return RogerstanimotoDistance(**metric_params)
    if metric.lower() == "russellrao":
        return RussellraoDistance(**metric_params)
    if metric.lower() == "sokalmichener":
        return SokalmichenerDistance(**metric_params)
    if metric.lower() == "sokalsneath":
        return SokalsneathDistance(**metric_params)
    raise(Exception(f"Not known metric : {metric}"))


def get_available_metrics(data_type="numeric"):
    if data_type == "numeric":
        return ["euclidean", "manhattan", "chebyshev", "sqeuclidean", "canberra", "mahalanobis", "cosine", "pearson", "lorentzian", "divergence"]
    if data_type == "categorical":
        return ["hamming", "eskin", "iof", "of", "co-oc", "dilca"]
    if data_type == "binary":
        return ["jaccard", "dice", "kulsinski", "rogerstanimoto", "russellrao", "sokalmichener", "sokalsneath"]


# def get_available_metrics(data_type="numeric"):
#     if data_type == "numeric":
#         return ["euclidean", "manhattan", "sqeuclidean", "canberra", "mahalanobis", "pearson"]
#     if data_type == "categorical":
#         return ["hamming", "iof", "of", "co-oc"]
#     if data_type == "binary":
#         return ["jaccard", "sokalsneath"]