import timeit
import os
import os.path
import  csv
import json
import sys
sys.path.append(".")
from metrics import base_metrics
from LSHkRepresentatives.MeasureManager import MeasureManager
import numpy as np
from termcolor import colored

class Measure(object):
    def calculate(self):
        return 'Finish caculating'
    def LoaddistMatrixAuto(self):
        if MeasureManager.IS_LOAD_AUTO == False or MeasureManager.CURRENT_DATASET=="None":
            print( f'SKIP LOADING distMatrix because IS_LOAD_AUTO={MeasureManager.IS_LOAD_AUTO} or dbname={MeasureManager.CURRENT_DATASET}; If you cluster a dataset multiple times, make sure to include the \'dbname\' parameter during initialization to cache the distance matrix of the dataset. Calculating distance matrix can take a lot of time with high categirical values dataset.' )
            return False
        path = 'saved_dist_matrices/json/' + self.name+"_" + MeasureManager.CURRENT_DATASET+ ".json"
        if os.path.isfile(path):
            with open(path, "r") as fp:
                self.distMatrix = json.load(fp)
        else: 
            print(colored('CANNOT OPEN FILE: ' +path),'yellow' )
            return False

        print("Loaded dist matrix: " , path)
        return True
    def SavedistMatrix(self):
        if not os.path.exists('saved_dist_matrices'):
            os.makedirs('saved_dist_matrices')
        if not os.path.exists('saved_dist_matrices/json'):
            os.makedirs('saved_dist_matrices/json')
        path = 'saved_dist_matrices/json/' + self.name+"_" + MeasureManager.CURRENT_DATASET+ ".json"
        with open('saved_dist_matrices/' + self.name+"_" + MeasureManager.CURRENT_DATASET, 'w') as f:
            wr = csv.writer(f)
            wr.writerows(self.distMatrix)
        with open(path, "w") as fp:
            json.dump(self.distMatrix, fp)
        print("Saving",self.name,"to:" ,path)
    def setUp(self, X, y):
        self.X_ = X
        self.y_ = y
        return 0
    def test(self):
        print('Test OK')
    def GeneratedistMatrix(self):
        D = len(self.X_[0])
        self.max = []
        for i in range(len(self.X_[0])):
            self.max.append(max(self.X_[:,i]))
        self.distMatrix = [];
        for d in range(D):
            matrix2D = [] # 2D array for 1 dimension
            for i in range(self.max[d]+1):
                matrix1D = [] # 1D array for 1 dimension
                for j in range(self.max[d]+1): 
                    matrix_tmp = self.CalcdistanceArrayForDimension(d,i,j)
                    matrix1D.append(matrix_tmp)
                matrix2D.append(matrix1D)
            self.distMatrix.append(matrix2D)

    def GeneratesimMatrix(self):
        self.d = d = len(self.X_[0])
        self.simMatrix = [];
        self.D = D = [len(np.unique(self.X_[:,i])) for i in range(d) ]
        self.medianValues = {}
        for di in range(d):
            matrix2D = [] # 2D array for 1 dimension
            for i in range(D[di]):
                matrix1D = [] # 1D array for 1 dimension
                for j in range(D[di]):
                    #matrix_tmp = 1-self.distMatrix[di][i][j]
                    if self.distMatrix[di][i][j] ==0: matrix_tmp = 10000;
                    else : matrix_tmp= 1/self.distMatrix[di][i][j]

                    matrix1D.append(matrix_tmp)
                matrix2D.append(matrix1D)
            self.simMatrix.append(matrix2D)


class CustomMeasure(Measure):
    def __init__(self, metric):
        self.m = metric
        self.name = metric.name

    def setUp(self, X, y, dataset_name=None):
        start = timeit.default_timer()
        self.X_ = X
        self.m.fit(X, dataset_name=dataset_name)
        self.distMatrix = self.m.per_attribute_dissimilarity_matrices
        return timeit.default_timer() - start

    def calculate(self, instance1, instance2):
        return self.m.dist(instance1, instance2)