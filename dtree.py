# imports
from __future__ import annotations
from typing import Tuple
from abc import ABC,abstractmethod
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer,make_regression
from sklearn.metrics import accuracy_score,precision_score,recall_score,mean_squared_error,mean_absolute_error,r2_score
from scipy.stats import kendalltau, rankdata, spearmanr
from joblib import Parallel, delayed

class Node(object):
    """
    Class to define & control tree nodes
    """
    
    def __init__(self) -> None:
        """
        Initializer for a Node class instance
        """
        self.__split    = None
        self.__feature  = None
        self.__left     = None
        self.__right    = None
        self.leaf_value = None
        self.purity = None

    def set_params(self, split: float, feature: int) -> None:
        """
        Set the split & feature parameters for this node
        
        Input:
            split   -> value to split feature on
            feature -> index of feature to be used in splitting 
        """
        self.__split   = split
        self.__feature = feature
        
    def get_params(self) -> Tuple[float,int]:
        """
        Get the split & feature parameters for this node
        
        Output:
            Tuple containing (split,feature) pair
        """
        return(self.__split, self.__feature)    
        
    def set_children(self, left: Node, right: Node) -> None:
        """
        Set the left/right child nodes for the current node
        
        Inputs:
            left  -> LHS child node
            right -> RHS child node
        """
        self.__left  = left
        self.__right = right
        
    def get_left_node(self) -> Node:
        """
        Get the left child node
        
        Output:
            LHS child node
        """
        return(self.__left)
    
    def get_right_node(self) -> Node:
        """
        Get the RHS child node
        
        Output:
            RHS child node
        """
        return(self.__right)
    
class DecisionTree(ABC):
    """
    Base class to encompass the CART algorithm
    """
    
    def __init__(self, max_depth: int=None, min_samples_split: int=2, gamma=0.98) -> None:
        """
        Initializer
        
        Inputs:
            max_depth         -> maximum depth the tree can grow
            min_samples_split -> minimum number of samples required to split a node
        """
        self.tree              = None
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.gamma             = gamma

    @abstractmethod
    def _purity(self, D: dict) -> None:
        """
        Protected function to define the purity
        """
        pass
        
    @abstractmethod
    def _leaf_value(self, D: dict) -> None:
        """
        Protected function to compute the value at a leaf node
        """
        pass

    def __grow(self, node: Node, D: dict, level: int) -> None:
        """
        Private recursive function to grow the tree during training
        
        Inputs:
            node  -> input tree node
            D     -> sample of data at node 
            level -> depth level in the tree for node
        """
        
        if node.purity is None:
            node.purity = self._purity(D)
        # are we in a leaf node?
        depth = (self.max_depth is None) or (self.max_depth >= (level+1))
        msamp = (self.min_samples_split <= D["X"].shape[0])
        
        # not a leaf node
        if depth and msamp:
            # initialize the function parameters
            p_node = None
            feature = None
            split   = None
            left_D  = None
            right_D = None
            left_p  = None
            right_p = None
            # iterate through the possible feature/split combinations
            for f in range(D["X"].shape[1]):
                for s in np.unique(D["X"][:,f]):
                    # for the current (f,s) combination, split the dataset
                    ind_l = D["X"][:,f]<=s
                    ind_r = D["X"][:,f]>s
                    D_l = {"X": D["X"][ind_l], "Y": D["Y"][ind_l]}
                    D_r = {"X": D["X"][ind_r], "Y": D["Y"][ind_r]}
                    # ensure we have non-empty arrays
                    if D_l["X"].size and D_r["X"].size:
                        # calculate the purity
                        p_l = self._purity(D_l)
                        p_r = self._purity(D_r)
                        p  = (D_l["X"].shape[0]/D["X"].shape[0])*p_l + (D_r["X"].shape[0]/D["X"].shape[0])*p_r
                        # now update the purity and choice of (f,s)
                        if (p_node is None) or (p > p_node):
                            p_node = p
                            feature = f
                            split   = s
                            left_D  = D_l
                            right_D = D_r
                            left_p  = p_l
                            right_p = p_r

            if node.purity >= self.gamma*(p_node):
                # set the current node's parameters
                node.set_params(split, feature)
                # declare child nodes
                left_node  = Node()
                right_node = Node()
                left_node.purity = left_p
                right_node.purity = right_p
                node.set_children(left_node, right_node)
                # investigate child nodes
                self.__grow(node.get_left_node(), left_D, level+1)
                self.__grow(node.get_right_node(), right_D, level+1)
            else :
                # set the node value & return
                node.leaf_value = self._leaf_value(D)
                return      
        # is a leaf node
        else:
            # set the node value & return
            node.leaf_value = self._leaf_value(D)
            return
        
    def __traverse(self, node: Node, Xrow: np.array) -> int | float:
        """
        Private recursive function to traverse the (trained) tree
        
        Inputs:
            node -> current node in the tree
            Xrow -> data sample being considered
        Output:
            leaf value corresponding to Xrow
        """        
        # check if we're in a leaf node?
        if node.leaf_value is None:
            # get parameters at the node
            (s,f) = node.get_params()
            # decide to go left or right?
            if (Xrow[f] <= s):
                return(self.__traverse(node.get_left_node(),Xrow))
            else:
                return(self.__traverse(node.get_right_node(),Xrow))
        else:
            # return the leaf value
            return(node.leaf_value)
        
    def fit(self, Xin: np.array, Yin: np.array) -> None:
        """
        Train the CART model
        
        Inputs:
            Xin -> input set of predictor features
            Yin -> input set of labels
        """        
        # prepare the input data
        D = {"X": Xin, "Y": Yin}
        # set the root node of the tree
        self.tree = Node()
        # build the tree
        self.__grow(self.tree,D,1)
        return self
        
    def predict(self, Xin: np.array) -> np.array:
        """
        Make predictions from the trained CART model
        
        Input:
            Xin -> input set of predictor features
        Output:
            array of prediction values
        """
        # iterate through the rows of Xin
        p = []
        for r in range(Xin.shape[0]):
            p.append(self.__traverse(self.tree, Xin[r,:]))
        # return predictions
        return(np.array(p))
    

class RankingTree(DecisionTree):
    """
    Decision Tree Classifier
    """
    
    def __init__(self, max_depth: int=None, min_samples_split: int=2, rank_sim: str='kendall', n_jobs=-1) -> None:
        """
        Initializer
        
        Inputs:
            max_depth         -> maximum depth the tree can grow
            min_samples_split -> minimum number of samples required to split a node
            loss              -> loss function to use during training
        """
        DecisionTree.__init__(self,max_depth,min_samples_split)
        self.rank_sim = rank_sim
        self.n_jobs=n_jobs
    
    def _leaf_value(self, D: dict) -> int:
        """
        Protected function to compute the value at a leaf node
        
        Input:
            D -> data to compute the leaf value
        Output:
            Mode of D         
        """ 
        return np.mean(D["Y"], axis=0)
    
    def __kendall(self, D: dict) -> float:
        """
        Input:
            D -> data to compute the Kendall over
        Output:
            Mean Kendall correlation over D
        """
        # compute the mean kendall correlation for the node
        y_pred = self._leaf_value(D)
        def tau(x, y):
            return kendalltau(x, y)[0]
        
        # res = Parallel(n_jobs=self.n_jobs)(delayed(tau)(D["Y"][i], D["Y"][j]) \
        #     for i in range(D["Y"].shape[0]) for j in range(i, D["Y"].shape[0]))
        res = Parallel(n_jobs=self.n_jobs)(delayed(tau)(D["Y"][i], y_pred) for i in range(D["Y"].shape[0]))
        return np.mean(res)
    
    def __spearman(self, D: dict) -> float:
        pass
    
    def __ndcg(self, D: dict) -> float:
        y_pred = self._leaf_value(D)
        def base_ndcg(y, y_pred, p=None):
            sorted_ind_true = np.argsort(-y)
            sorted_ind_pred = np.argsort(-y_pred)
            if p is None:
                p = sum(y > 0)
            idcg = np.sum(y[sorted_ind_true[:p]]/np.log2(np.arange(2, p+2)))
            ind = [k for k in sorted_ind_pred if y[k] > 0]
            dcg = np.sum(y[ind[:p]]/np.log2(np.arange(2, p+2)))
            return dcg/idcg
        # res = Parallel(n_jobs=self.n_jobs)(delayed(base_ndcg)(D["Y"][i], D["Y"][j], p=5) \
        #     for i in range(D["Y"].shape[0]) for j in range(D["Y"].shape[0]))
        res = Parallel(n_jobs=self.n_jobs)(delayed(base_ndcg)(D["Y"][i], y_pred, p=5) for i in range(D["Y"].shape[0]))
        return np.mean(res)
    
    def _purity(self, D: dict) -> float:
        """
        Protected function to define the purity
        
        Input:
            D -> data to compute the purity metric over
        Output:
            Purity metric for D        
        """            
        # use the selected ranking similarity measure to calculate the node Purity
        p = None
        if self.rank_sim == 'kendall':
            p = self.__kendall(D)
        elif self.rank_sim == 'spearman':
            p = self.__kendall(D)
        elif self.rank_sim == 'ndcg':
            p = self.__ndcg(D)
        return p
    
if __name__ == "__main__":
    X = np.random.uniform(size=(100, 10))
    Y = np.random.uniform(size=(100, 30))
    clf = RankingTree(rank_sim='ndcg')
    clf = clf.fit(X, Y)
    print(clf.predict(np.random.uniform(size=(5, 10))))