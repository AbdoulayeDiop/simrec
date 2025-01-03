import os
import os.path
import sys
from sys import platform
sys.path.append(".")
from metrics import base_metrics
import numpy as np
from sklearn.utils.validation import check_array
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import silhouette_score
import timeit
import csv
from csv import writer
import random
import math
from LSHkRepresentatives.Measure import CustomMeasure


def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)


class ClusteringAlgorithm:
    ALGORITHM_LIST = ['kModes', 'kRepresentatives']
    # def __init__(self, X, y,n_init=5,n_clusters=-1,max_iter=100,dbname='dpname',verbose =0,random_state = None):
    #     self.random_state= random_state
    #     self.verbose = verbose
    #     self.measurename = 'None'
    #     self.dicts = [];self.dicts2 = []
    #     self.iter=-1
    #     self.dbname = dbname
    #     self.time_lsh=-1
    #     self.X = X
    #     self.y = y
    #     self.n = len(self.X)
    #     self.d = len(self.X[0])
    #     self.k = n_clusters if n_clusters > 0 else len(np.unique(y))
    #     self.n_init = n_init
    #     self.n_iter = max_iter
    #     self.scorebest = -2

    def __init__(self, n_clusters, n_init=5, max_iter=100, dbname="None", verbose=0, random_state=None, num_metric="euclidean", cat_metric="dilca", categorical_weight=1, numerical_weight=1):
        self.y = None
        self.random_state = random_state
        self.verbose = verbose
        self.measurename = 'None'
        self.dicts = []
        self.dicts2 = []
        self.iter = -1
        self.dbname = dbname
        self.time_lsh = -1

        self.k = n_clusters if n_clusters > 0 else len(np.unique(y))
        self.n_init = n_init
        self.n_iter = max_iter
        self.scorebest = -2

        self.num_metric = self.check_metric(num_metric)
        self.cat_metric = self.check_metric(cat_metric)
        self.weight_CATE = categorical_weight
        self.weight_NUM = numerical_weight
            

    def check_metric(self, metric):
        if isinstance(metric, base_metrics.Metric):
            return metric
        if isinstance(metric, str):
            return base_metrics.get_metric(metric)
        raise(Exception("metric should be of type str or Metric"))
        

    # attributeMasks: masking the types of attributes: 0 -> Categorical ; 1 -> Numeric  : Example: [0,1,0,1] means the 1st and 3rd attributes are the categircal attribues while 2st and 4rd attributes are the numeric attribues
    def fit(self, X, attributeMasks=None, dataset_name=None):
        """
        X: dataset
        attributeMasks: masking the types of attributes: 
            0 -> Categorical ; 1 -> Numeric  : 
            Example: [0,1,0,1] means the 1st and 3rd attributes are the categorical attribues while 2st and 4rd attributes are the numeric attribues
        categorical_weight: weight of categorical attributes
        numerical_weight: weight of numericcal attributes

        """
        if type(attributeMasks) == list:
            attributeMasks = np.array(attributeMasks)
        if attributeMasks is None:
            attributeMasks = np.zeros((len(X[0])))
        self.attributeMasks = attributeMasks

        self.indexes_CATE = indexes_CATE = np.argwhere(attributeMasks == 0)[
            :, 0]
        self.indexes_NUM = indexes_NUM = np.argwhere(attributeMasks > 0)[:, 0]

        self.X_CATE = X[:, indexes_CATE].astype(int)
        self.X_NUM = X[:, indexes_NUM].astype(float)

        self.d_NUM = len(self.X_NUM[0])

        self.n = len(self.X_CATE)
        self.d_CATE = len(self.X_CATE[0])
        self.uniques_CATE = []
        for i in range(self.d_CATE):
            unique = np.unique(self.X_CATE[:, i])
            self.uniques_CATE.append(unique)
        self.D_CATE = [len(self.uniques_CATE[i]) for i in range(self.d_CATE)]
        self.dataset_name = dataset_name
        
        # if self.caching and self.cache_dir is None:
        #     self.cache_dir = "distance_caching_dir"
        #     if not os.path.exists(self.cache_dir):
        #         os.makedirs(self.cache_dir)

        self.SetupMeasures()
        self.SetupLSH()
        self.DoCluster()
        return self.labels

    def SetupMeasures(self):
        self.num_metric.fit(self.X_NUM, dataset_name=self.dataset_name)
        self.measure = CustomMeasure(self.cat_metric)
        self.measure.setUp(self.X_CATE, self.y, dataset_name=self.dataset_name)

    def Overlap(self, x, y):
        n = len(x)
        sum = 0
        for i in range(n):
            if x[i] != y[i]:
                sum += 1
        return sum

    def DoCluster(self):
        print("Do something")
        return -1

    def _labels_cost(self, X, centroids, dissim, membship=None):
        X = check_array(X)
        n_points = X.shape[0]
        cost = 0.
        labels = np.empty(n_points, dtype=np.uint16)
        for ipoint, curpoint in enumerate(X):
            diss = self.ComputeDistances(centroids, curpoint)
            clust = np.argmin(diss)
            labels[ipoint] = clust
            cost += diss[clust]
        return labels, cost

    def ComputeDistances(self, X, mode):
        return [self.measure.calculate(i, mode) for i in X]

    def CheckCLusteringPurityByHeuristic(self, labels_, km_labels_):
        # start = timeit.default_timer()
        unique_ = np.unique(labels_)
        n_clusters = len(unique_)
        matching_matrix = [-1 for i in range(n_clusters)]
        n = len(labels_)
        n_range = range(n)
        # Computer matching matrix
        count_item = 0
        for i in range(n_clusters):
            max_count = -1
            max_index = 0
            for j in range(n_clusters):
                if j in matching_matrix:
                    continue
                count = sum([labels_[k] == i and km_labels_[
                            k] == j for k in n_range])
                if count > max_count:
                    max_count = count
                    max_index = j
            matching_matrix[i] = max_index
            count_item = count_item + max_count
        # Compute score
        score2 = count_item/len(labels_)
        return score2

    def AcPrRc(self, A, B):
        # B = np.array([0,0,0,0,0,1,2,2,   0,1,1,1,1,   1,2,2,2 ])
        # B = np.array([1,1,1,1,1,2,0,0,   1,2,2,2,2,   2,0,0,0 ])

        k = len(np.unique(A))
        n = len(A)
        MAP = [0, 1, 2]

        clustersA = [np.where(A == i)[0] for i in range(k)]
        clustersB = [np.where(B == i)[0] for i in range(k)]

        TP_FP = 0
        for i in range(k):
            if len(clustersB[i]) > 1:
                TP_FP += nCr(len(clustersB[i]), 2)
        TP = 0
        for i in range(k):
            for j in range(k):
                num = sum(A[clustersB[i]] == j)
                if num > 1:
                    TP += nCr(num, 2)

        FP = TP_FP-TP

        TN_FN = int(n*(n-1)/2) - TP_FP
        FN = 0

        for i in range(k):
            for j in range(k):
                num_ = sum(A[clustersB[i]] == j)
                sum_ = 0
                for i2 in range(i+1, k):
                    sum_ += sum(A[clustersB[i2]] == j)
                FN += num_*sum_
        TN = TN_FN - FN
        # print('TP=',TP,'TP_FP=',TP_FP ,'TN_FN=',TN_FN,'FN=',FN)
        PR = TP/(TP+FP)
        RC = TP/(TP+FN)
        AC = (TP+TN)/(TP+TN+FP+FN)
        # print('PR=',PR,'RC=',RC,'AC=',AC )
        return AC, PR, RC

    def CalcScore(self, y, verbose=True):
        self.y = y
        starttime = timeit.default_timer()
        s = ""
        if self.n*self.k <= 8000000:
            self.purity_score = self.CheckCLusteringPurityByHeuristic(
                self.y, self.labels)
        else:
            self.purity_score = -2
        s += str(timeit.default_timer()-starttime)+"|"
        starttime = timeit.default_timer()
        # tulti.CheckClusteringNMI(self.y, self.labels)
        self.NMI_score = normalized_mutual_info_score(self.y, self.labels)
        s += str(timeit.default_timer()-starttime)+"|"
        starttime = timeit.default_timer()
        # tulti.CheckClusteringARI(self.y, self.labels)
        self.ARI_score = adjusted_rand_score(self.labels, self.y)
        s += str(timeit.default_timer()-starttime)+"|"
        starttime = timeit.default_timer()
        self.AMI_score = adjusted_mutual_info_score(self.labels, self.y)
        s += str(timeit.default_timer()-starttime)+"|"
        starttime = timeit.default_timer()
        self.HOMO_score = homogeneity_score(self.labels, self.y)
        s += str(timeit.default_timer()-starttime)+"|"
        starttime = timeit.default_timer()
        if self.n*self.k <= 800000:
            try:
                self.SILHOUETTE_score = silhouette_score(
                    self.X, self.labels, metric=self.Overlap)
            except:
                self.SILHOUETTE_score = -1
        else:
            self.SILHOUETTE_score = -2

        s += str(timeit.default_timer()-starttime)+"|"
        starttime = timeit.default_timer()
        if self.n*self.k <= 8000000:
            self.Ac_score, self.Pr_score, self.Rc_score = self.AcPrRc(
                self.y, self.labels)
        else:
            self.Ac_score = self.Pr_score = self.Rc_score = -2

        s += str(timeit.default_timer()-starttime)+"|"
        starttime = timeit.default_timer()
        self.AddVariableToPrint("Scoringtime", s)
        self.WriteResultToCSV()
        if verbose:
            print("Purity:", "%.2f" % self.purity_score, "NMI:", "%.2f" % self.NMI_score, "ARI:", "%.2f" % self.ARI_score, "Sil: ", "%.2f" % self.SILHOUETTE_score, "Acc:", "%.2f" % self.Ac_score,
                  "Recall:", "%.2f" % self.Rc_score, "Precision:", "%.2f" % self.Pr_score)
        return (self.purity_score, self.NMI_score, self.ARI_score, self.AMI_score, self.HOMO_score, self.SILHOUETTE_score, self.time_score, self.time_lsh,
                self.Ac_score, self.Pr_score, self.Rc_score)

    def append_list_as_row(self, file_name, list_of_elem):
        with open(file_name, 'a+', newline='') as write_obj:
            csv_writer = writer(write_obj)
            csv_writer.writerow(list_of_elem)

    def AddVariableToPrint(self, name, val):
        self.dicts2.append((name, val))

    def WriteResultToCSV(self, file=''):
        return 0
        if not os.path.exists(TDef.folder):
            os.makedirs(TDef.folder)
        if file == '':
            file = TDef.folder + '/' + self.name + TDef.fname + ".csv"

        self.dbname = self.dbname.replace(
            "_c", "").replace(".csv", "").capitalize()
        self.dicts.append(('dbname', self.dbname))
        self.dicts.append(('n', self.n))
        self.dicts.append(('d', self.d))
        self.dicts.append(('k', self.k))
        self.dicts.append(('range', '-1'))
        self.dicts.append(('sigma_ratio', -1))
        self.dicts.append(('Measure',  self.measurename))
        self.dicts.append(('n_init', self.n_init))
        self.dicts.append(('n_iter', self.n_iter))
        self.dicts.append(('iter', self.iter))
        self.dicts.append(('Purity', self.purity_score))
        self.dicts.append(('NMI', self.NMI_score))
        self.dicts.append(('ARI', self.ARI_score))
        self.dicts.append(('AMI', self.AMI_score))
        self.dicts.append(('Homogeneity', self.HOMO_score))
        self.dicts.append(('Silhouette', self.SILHOUETTE_score))
        self.dicts.append(('Accuracy', self.Ac_score))
        self.dicts.append(('Precision', self.Pr_score))
        self.dicts.append(('Recall', self.Rc_score))
        self.dicts.append(('Time', self.time_score))
        self.dicts.append(('LSH_time', self.time_lsh))
        self.dicts.append(('Score', self.scorebest))

        dicts = self.dicts+self.dicts2
        try:
            if os.path.isfile(file) == False:
                colnames = [i[0] for i in dicts]
                self.append_list_as_row(file, colnames)
            vals = [i[1] for i in dicts]
            self.append_list_as_row(file, vals)
        except Exception as ex:
            # self.exe(file,ex)
            print('Cannot write to file ', file, '', ex)
            self.WriteResultToCSV(
                file + str(random.randint(0, 1000000)) + '.csv')

    def AddValuesToMyTable(self, mytable, is_first=False, dbname=''):
        if is_first:
            mytable.AddValue("Purity", 'Dataset', dbname)
            mytable.AddValue("NMI", 'Dataset', dbname)
            mytable.AddValue("ARI", 'Dataset', dbname)
            mytable.AddValue("AMI", 'Dataset', dbname)
            mytable.AddValue("Homogeneity", 'Dataset', dbname)
            mytable.AddValue("Silhouette", 'Dataset', dbname)
            mytable.AddValue("iter", 'Dataset', dbname)
            mytable.AddValue("Precision", 'Dataset', dbname)
            mytable.AddValue("Recall", 'Dataset', dbname)
            mytable.AddValue("Accuracy", 'Dataset', dbname)
            mytable.AddValue("LSH_time", 'Dataset', dbname)
            mytable.AddValue("Time", 'Dataset', dbname)
            mytable.AddValue("Time", 'Dataset', dbname)

        mytable.AddValue("Purity", self.measurename, self.purity_score)
        mytable.AddValue("NMI", self.measurename, self.NMI_score)
        mytable.AddValue("ARI", self.measurename, self.ARI_score)
        mytable.AddValue("AMI", self.measurename, self.AMI_score)
        mytable.AddValue("Homogeneity", self.measurename, self.HOMO_score)
        mytable.AddValue("Silhouette", self.measurename, self.SILHOUETTE_score)
        mytable.AddValue("iter", self.measurename, self.iter)
        mytable.AddValue("Precision", self.measurename, self.Pr_score)
        mytable.AddValue("Recall", self.measurename, self.Rc_score)
        mytable.AddValue("Accuracy", self.measurename, self.Ac_score)
        mytable.AddValue("LSH_time", self.measurename, self.time_lsh)
        mytable.AddValue("Time", self.measurename, self.time_score)
