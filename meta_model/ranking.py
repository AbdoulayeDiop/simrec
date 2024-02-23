import abc
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model as linear
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from pytorchltr.loss import PairwiseLogisticLoss as ltrloss
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
# from svm_label_ranking.model import SVMLR
# from svm_label_ranking.arff import ArffFile
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from torch.optim import Adam
from tqdm import tqdm
from utils import ndcg
from ranking_tree import RankingTree
from scipy.stats import wasserstein_distance

def mse(y_true, y_pred):
    return np.mean((y_true[y_true > 0]-y_pred[y_true > 0])**2)

scorer = make_scorer(lambda yt, yp: np.mean([y[y>-1][np.argmax(yp[i][y>-1])] for i, y in enumerate(yt)]))
scorer_func = lambda yt, yp: np.mean([y[y>-1][np.argmax(yp[i][y>-1])] for i, y in enumerate(yt)])

# scorer = make_scorer(lambda yt, yp: np.mean(ndcg(yt, yp, p=10)))
# scorer_func = lambda yt, yp: np.mean(ndcg(yt, yp, p=10))

class LRRanker(linear.ElasticNet):
    def __init__(self, alpha=1, l1_ratio=0.5, **params) -> None:
        super().__init__(alpha=alpha, l1_ratio=l1_ratio, **params)

    def cross_val_fit(self, X, Y, groups=None, n_splits=5, scorer=scorer, return_cv_scores=False, verbose=0, n_jobs=-1):
        parameters = {
            'alpha': [0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 5, 10],
            'l1_ratio': np.linspace(0.1, 1, 10),
        }
        gridcv = GridSearchCV(self, parameters,
                              scoring=scorer,
                              cv=GroupKFold(n_splits=n_splits) if groups is not None else KFold(
                                  n_splits=n_splits),
                              verbose=verbose,
                              error_score='raise',
                              n_jobs=n_jobs
                              ).fit(X, Y, groups=groups)
        if return_cv_scores:
            return gridcv.best_estimator_, gridcv.best_score_
        return gridcv.best_estimator_


class KNNRanker(KNeighborsRegressor):
    def __init__(self, n_neighbors=5, metric="euclidean", weights="uniform", **params) -> None:
        super().__init__(n_neighbors=n_neighbors, metric=metric, weights=weights, **params)

    def cross_val_fit(self, X, Y, groups=None, n_splits=5, scorer=scorer, return_cv_scores=False, verbose=0, n_jobs=-1):
        parameters = {
            'n_neighbors': [v for v in range(1, 26) if v <= X.shape[0]/2],
            'metric': ["euclidean", "manhattan", "cosine"],
            'weights': ["uniform", "distance"]
        }
        gridcv = GridSearchCV(self, parameters,
                              scoring=scorer,
                              cv=GroupKFold(n_splits=n_splits) if groups is not None else KFold(
                                  n_splits=n_splits),
                              verbose=verbose,
                              error_score='raise',
                              n_jobs=n_jobs
                              ).fit(X, Y, groups=groups)
        if return_cv_scores:
            return gridcv.best_estimator_, gridcv.best_score_
        return gridcv.best_estimator_

    def grid_search(self, X, Y, groups=None, n_splits=5, scorer=scorer, verbose=0, n_jobs=-1, refit=True):
        parameters = {
            'n_neighbors': [v for v in range(1, 32, 2) if v <= X.shape[0]/2],
            'metric': ["euclidean", "manhattan", "cosine"],
            'weights': ["uniform", "distance"]
        }
        gridcv = GridSearchCV(self, parameters,
                              scoring=scorer,
                              cv=GroupKFold(n_splits=n_splits) if groups is not None else KFold(
                                  n_splits=n_splits),
                              verbose=verbose,
                              error_score='raise',
                              refit=refit,
                              n_jobs=n_jobs
                              ).fit(X, Y, groups=groups)
        return gridcv


class DTreeRanker(DecisionTreeRegressor):
    def __init__(self, min_samples_leaf=1, max_depth=None, max_features=None, **params) -> None:
        super().__init__(min_samples_leaf=min_samples_leaf,
                         max_depth=max_depth, max_features=max_features, **params)

    def cross_val_fit(self, X, Y, groups=None, n_splits=5, scorer=scorer, return_cv_scores=False, verbose=0, n_jobs=-1):
        parameters = {
            'min_samples_leaf': [v for v in [1, 5, 10, 20, 30] if v <= X.shape[0]/2],
            'max_depth': [None, 5, 10],
            'max_features': [None, "sqrt", "log2"]
        }
        gridcv = GridSearchCV(self, parameters,
                              scoring=scorer,
                              cv=GroupKFold(n_splits=n_splits) if groups is not None else KFold(
                                  n_splits=n_splits),
                              verbose=verbose,
                              error_score='raise',
                              n_jobs=n_jobs
                              ).fit(X, Y, groups=groups)
        if return_cv_scores:
            return gridcv.best_estimator_, gridcv.best_score_
        return gridcv.best_estimator_


class NDCGDTreeRanker(BaseEstimator):
    def __init__(self, max_depth=None, min_samples_split=2, **params) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.other_params = params
    
    def fit(self, X, Y):
        self.model = RankingTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, rank_sim="ndcg")
        self.model = self.model.fit(X, Y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def cross_val_fit(self, X, Y, groups=None, n_splits=5, scorer=scorer, return_cv_scores=False, verbose=0, n_jobs=-1):
        parameters = {
            'min_samples_split': [v for v in [2, 5, 10, 20] if v <= X.shape[0]/2],
            'max_depth': [None, 5, 10]
        }
        gridcv = GridSearchCV(self, parameters,
                              scoring=scorer,
                              cv=GroupKFold(n_splits=n_splits) if groups is not None else KFold(
                                  n_splits=n_splits),
                              verbose=verbose,
                              error_score='raise',
                              n_jobs=n_jobs
                              ).fit(X, Y, groups=groups)
        if return_cv_scores:
            return gridcv.best_estimator_, gridcv.best_score_
        return gridcv.best_estimator_


class RFRanker(RandomForestRegressor):
    def __init__(self, n_estimators=100, min_samples_leaf=1, max_depth=None, max_features=None, **params) -> None:
        super().__init__(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                         max_depth=max_depth, max_features=max_features, **params)

    def cross_val_fit(self, X, Y, groups=None, n_splits=5, scorer=scorer, return_cv_scores=False, verbose=0, n_jobs=-1):
        parameters = {
            'n_estimators': [50, 100, 200],
            'min_samples_leaf': [v for v in [1, 5, 10, 20, 30] if v <= X.shape[0]/2],
            'max_depth': [None],
            'max_features': [None, "sqrt"]
        }
        gridcv = GridSearchCV(self, parameters,
                              scoring=scorer,
                              cv=GroupKFold(n_splits=n_splits) if groups is not None else KFold(
                                  n_splits=n_splits),
                              verbose=verbose,
                              error_score='raise',
                              n_jobs=n_jobs
                              ).fit(X, Y, groups=groups)
        if return_cv_scores:
            return gridcv.best_estimator_, gridcv.best_score_
        return gridcv.best_estimator_


class MultipleRegressors(ABC):
    def fit(self, X, Y, lb=-1):
        self.n_outputs = Y.shape[1]
        base_model = self.create_regressor()

        def fit_(y):
            model = clone(base_model)
            try:
                model = model.fit(X[y > lb], y[y > lb])
            except:
                model = None
            return model
        self.all_regressors = Parallel(
            n_jobs=-1)(delayed(fit_)(Y[:, k]) for k in range(self.n_outputs))
        return self

    def predict(self, X):
        Y = np.zeros((X.shape[0], self.n_outputs))

        def predict_(model):
            return model.predict(X) if model is not None else None
        ret = Parallel(n_jobs=-1)(delayed(predict_)(model)
                                  for model in self.all_regressors)
        for k, yk in enumerate(ret):
            if yk is not None:
                Y[:, k] = yk
        return Y

    @abstractmethod
    def create_regressor(self):
        pass


class MDTree(BaseEstimator, MultipleRegressors):
    def __init__(self, min_samples_leaf=1, max_depth=None, max_features=None, **params) -> None:
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.other_params = params

    def create_regressor(self):
        return DecisionTreeRegressor(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            max_features=self.max_features,
            **self.other_params
        )

    def cross_val_fit(self, X, Y, groups=None, n_splits=5, scorer=scorer, return_cv_scores=False, verbose=0, n_jobs=-1):
        parameters = {
            'min_samples_leaf': [v for v in [1, 5, 10, 20, 30] if v <= X.shape[0]/2],
            'max_depth': [None, 5, 10],
            'max_features': [None, "sqrt", "log2"]
        }
        gridcv = GridSearchCV(self, parameters,
                              scoring=scorer,
                              cv=GroupKFold(n_splits=n_splits) if groups is not None else KFold(
                                  n_splits=n_splits),
                              verbose=verbose,
                              error_score='raise',
                              n_jobs=n_jobs
                              ).fit(X, Y, groups=groups)
        if return_cv_scores:
            return gridcv.best_estimator_, gridcv.best_score_
        return gridcv.best_estimator_


class MKNN(BaseEstimator, MultipleRegressors):
    def __init__(self, n_neighbors=8, metric="euclidean", weights="uniform", **params) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.other_params = params

    def create_regressor(self):
        return KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            weights=self.weights,
            **self.other_params
        )

    def cross_val_fit(self, X, Y, groups=None, n_splits=5, scorer=scorer, return_cv_scores=False, verbose=0, n_jobs=-1):
        parameters = {
            'n_neighbors': [v for v in range(1, 32, 2) if v <= X.shape[0]/2],#[v for v in [1, 5, 10, 20, 30] if v <= X.shape[0]/2],
            'metric': ["euclidean", "manhattan", "cosine"],
            'weights': ["uniform", "distance"]
        }
        gridcv = GridSearchCV(self, parameters,
                              scoring=scorer,
                              cv=GroupKFold(n_splits=n_splits) if groups is not None else KFold(
                                  n_splits=n_splits),
                              verbose=verbose,
                              error_score='raise',
                              n_jobs=n_jobs
                              ).fit(X, Y, groups=groups)
        if return_cv_scores:
            return gridcv.best_estimator_, gridcv.best_score_
        return gridcv.best_estimator_


class PairwiseRanker(ABC):
    def _get_pairs(self, Y, lb):
        n = Y.shape[1]
        list_i = []
        list_j = []
        list_samples = []
        for i in range(n-1):
            for j in range(i+1, n):
                list_i.append(i)
                list_j.append(j)
                list_samples.append([ind for ind in range(
                    Y.shape[0]) if Y[ind, i] > lb and Y[ind, j] > lb])
        return list_i, list_j, list_samples

    def fit(self, X, Y, lb=-1, verbose=0):
        self.n_outputs = Y.shape[1]
        self.list_i, self.list_j, self.list_samples = self._get_pairs(Y, lb)
        n_pairs = len(self.list_i)
        pairwiseY = Y[:, self.list_i] - Y[:, self.list_j]
        base_model = self.create_model()

        def fit_(k):
            y = pairwiseY[:, k][self.list_samples[k]]
            model = clone(base_model)
            try:
                model = model.fit(X[self.list_samples[k]], y)
            except:
                model = None
            return model
        # self.all_models = Parallel(n_jobs=-1, verbose=verbose)(delayed(fit_)(k) for k in range(n_pairs))
        self.all_models = [fit_(k) for k in (
            range(n_pairs) if verbose == 0 else tqdm(range(n_pairs)))]

        return self

    def predict(self, X):
        Y = np.zeros((X.shape[0], self.n_outputs))

        def predict_(clf):
            return clf.predict(X) if clf is not None else 0
        pairwise_pred = Parallel(n_jobs=-1)(delayed(predict_)(clf)
                                            for clf in self.all_models)
        for k, yk in enumerate(pairwise_pred):
            i, j = self.list_i[k], self.list_j[k]
            Y[:, i] += yk
            Y[:, j] += -yk
        return Y

    @abstractmethod
    def create_model(self):
        pass


class PairwiseLRRanker(BaseEstimator, PairwiseRanker):
    def __init__(self, **params) -> None:
        self.other_params = params

    def create_model(self):
        return linear.LinearRegression(
            **self.other_params
        )


class PairwiseKNNRanker(BaseEstimator, PairwiseRanker):
    def __init__(self, n_neighbors=8, metric="euclidean", weights="uniform", **params) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.other_params = params

    def create_model(self):
        return KNeighborsRegressor(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            weights=self.weights,
            **self.other_params
        )

    # def cross_val_fit(self, X, Y, groups=None, n_splits=5, scorer=scorer, return_cv_scores=False, verbose=0, n_jobs=1):
    #     parameters = {
    #         'n_neighbors': [1, 3, 5, 8, 12, 18, 25],
    #         'metric': ["euclidean", "manhattan", "cosine"],
    #         'weights': ["uniform", "distance"]
    #     }
    #     best_score = -1
    #     best_n_neighbors = None
    #     best_metric = None
    #     best_weights = None
    #     cv=GroupKFold(n_splits=n_splits) if groups is not None else KFold(n_splits=n_splits)
    #     n_param_sets = len(parameters["n_neighbors"])*len(parameters["metric"])*len(parameters["weights"])
    #     for n_neighbors in parameters["n_neighbors"]:
    #         for metric in parameters["metric"]:
    #             for weights in parameters["weights"]:
    #                 if n_param_sets%10==0: print(f"{n_param_sets} remaining...")
    #                 score = 0
    #                 for train_index, test_index in cv.split(X, Y, groups=groups):
    #                     model = clone(self)
    #                     model.set_params(n_neighbors=n_neighbors, metric=metric, weights=weights)
    #                     model = model.fit(X[train_index], Y[train_index])
    #                     score += np.mean(ndcg(Y[test_index], model.predict(X[test_index])))
    #                 score /= n_splits
    #                 if score > best_score:
    #                     best_score = score
    #                     best_n_neighbors = n_neighbors
    #                     best_metric = metric
    #                     best_weights = weights
    #                 n_param_sets -= 1
    #     model = clone(self)
    #     model.set_params(n_neighbors=best_n_neighbors, metric=best_metric, weights=best_weights)
    #     model = model.fit(X[train_index], Y[train_index])
    #     return model

    def cross_val_fit(self, X, Y, groups=None, n_splits=5, scorer=scorer, return_cv_scores=False, verbose=0, n_jobs=8):
        parameters = {
            'n_neighbors': [v for v in [1, 5, 10, 20, 30] if v <= X.shape[0]/2],
            'metric': ["euclidean", "manhattan", "cosine"],
            'weights': ["uniform", "distance"]
        }
        gridcv = GridSearchCV(self, parameters,
                              scoring=scorer,
                              cv=GroupKFold(n_splits=n_splits) if groups is not None else KFold(
                                  n_splits=n_splits),
                              verbose=verbose,
                              error_score='raise',
                              n_jobs=n_jobs
                              ).fit(X, Y, groups=groups)
        if return_cv_scores:
            return gridcv.best_estimator_, gridcv.best_score_
        return gridcv.best_estimator_


class PairwiseDTreeRanker(BaseEstimator, PairwiseRanker):
    def __init__(self, min_samples_leaf=1, max_depth=None, max_features=None, **params) -> None:
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_features = max_features
        self.other_params = params

    def create_model(self):
        return DecisionTreeRegressor(
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            max_features=self.max_features,
            **self.other_params
        )

    def cross_val_fit(self, X, Y, groups=None, n_splits=5, scorer=scorer, return_cv_scores=False, verbose=0, n_jobs=15):
        parameters = {
            'min_samples_leaf': [v for v in [1, 5, 10, 20, 30] if v <= X.shape[0]/2],
            'max_depth': [None, 5, 10],
            'max_features': [None, "sqrt"]
        }
        gridcv = GridSearchCV(self, parameters,
                              scoring=scorer,
                              cv=GroupKFold(n_splits=n_splits) if groups is not None else KFold(
                                  n_splits=n_splits),
                              verbose=verbose,
                              error_score='raise',
                              n_jobs=n_jobs
                              ).fit(X, Y, groups=groups)
        if return_cv_scores:
            return gridcv.best_estimator_, gridcv.best_score_
        return gridcv.best_estimator_


class RankNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=(100,), multiple=False, gamma=0, loss="mse", optimizer='adam', learnin_rate=1e-3, wd=0, device='cpu') -> None:
        super().__init__()
        self.hidden_layers = hidden_layers
        self.gamma = gamma
        self.loss = loss
        self.multiple = multiple
        if self.multiple:
            self.network = nn.ModuleList()
            for _ in range(output_dim):
                subnet = nn.Sequential()
                for i in range(len(hidden_layers)):
                    n_input = input_dim if i == 0 else hidden_layers[i-1]
                    n_output = hidden_layers[i]
                    subnet.append(nn.Linear(n_input, n_output))
                    subnet.append(nn.ReLU())
                subnet.append(nn.Linear(hidden_layers[-1], 1))
                subnet.append(nn.Sigmoid())
                self.network.append(subnet)
        else:
            self.network = nn.Sequential()
            for i in range(len(hidden_layers)):
                n_input = input_dim if i == 0 else hidden_layers[i-1]
                n_output = hidden_layers[i]
                self.network.append(nn.Linear(n_input, n_output))
                self.network.append(nn.ReLU())
            self.network.append(nn.Linear(hidden_layers[-1], output_dim))
            self.network.append(nn.Sigmoid())

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=learnin_rate, weight_decay=wd)
        self.device = device
        self.to(device)

    def forward(self, X):
        return torch.concat([subnet(X) for subnet in self.network], dim=1) if self.multiple else self.network(X)

    def fit(self, X, Y, epochs=20, batch_size=None,
            X_test=None, Y_test=None, shuffle=True):

        if batch_size is None:
            batch_size = len(X)

        history = {
            "train": {
                "loss": [],
            },
            "test": {
                "loss": [],
            }
        }

        X_traint = torch.tensor(X, device=self.device,
                                requires_grad=False).float()
        Y_traint = torch.tensor(Y, device=self.device,
                                requires_grad=False).float()
        n_traint = torch.tensor(np.ones(
            Y.shape[0])*Y.shape[1], device=self.device, requires_grad=False).float()

        train_dataset = torch.utils.data.TensorDataset(
            X_traint, Y_traint, n_traint)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

        if X_test is not None:
            X_testt = torch.tensor(
                X_test, device=self.device, requires_grad=False).float()
            Y_testt = torch.tensor(
                Y_test, device=self.device, requires_grad=False).float()
            n_testt = torch.tensor(np.ones(
                Y_test.shape[0])*Y_test.shape[1], device=self.device, requires_grad=False).float()

        loss_function = nn.MSELoss() if self.loss == 'mse' else ltrloss()
        for _ in tqdm(range(epochs)):
            train_loss = 0
            count = 0
            for X_, Y_, n in train_loader:
                Yp = self.forward(X_)
                # Calculating the loss function
                loss = loss_function(
                    Yp, Y_) if self.loss == 'mse' else loss_function(Yp, Y_, n).mean()
                train_loss += loss.item()
                count += 1

                l2_norm = sum(p.pow(2.0).sum()
                              for p in self.network.parameters())
                loss += self.gamma*l2_norm

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            history["train"]["loss"].append(train_loss/count)
            if X_test is not None:
                with torch.no_grad():
                    # train_loss = loss_function(Y_traint, model.forward(X_traint), n_traint).mean().item()
                    test_loss = loss_function(self.forward(X_testt), Y_testt).mean().item() if self.loss == 'mse' \
                        else loss_function(self.forward(X_testt), Y_testt, n_testt).mean().item()
                    history["test"]["loss"].append(test_loss)
        self.history = history
        return self

    def show_history(self):
        plt.figure(figsize=(4, 3))
        plt.plot(self.history["train"]["loss"], label="train")
        plt.plot(self.history["test"]["loss"], label="test")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def predict(self, X):
        Xt = torch.tensor(X, device=self.device, requires_grad=False).float()
        return self.forward(Xt).cpu().detach().numpy()


ALL_MODELS = {
    # "LR": linear.LinearRegression,
    "ElasticNet": LRRanker,
    "KNN": KNNRanker,
    "DTree": DTreeRanker,
    "RT": NDCGDTreeRanker,
    "MDTree": MDTree,
    "MKNN": MKNN,
    "RF": RFRanker,
    "PR-LR": PairwiseLRRanker,
    "PR-KNN": PairwiseKNNRanker,
    "PR-DTree": PairwiseDTreeRanker,
    # "RForest": ensemble.RandomForestRegressor,
    "RankNet": RankNet,
}

NN_MODELS = {
    "RankNet": RankNet,
}

if __name__ == "__main__":
    import time
    X = np.random.uniform(size=(100, 8))
    Y = np.random.uniform(size=(100, 120))
    ranker = ALL_MODELS["RT"]()
    print("fitting...", end="")
    start = time.time()
    ranker = ranker.cross_val_fit(X, Y, verbose=10)
    end = time.time()
    print(f"END. {end - start} s.")

    # ranker = SVMLR2()
    # ranker = ranker.fit(X, Y)

    # ranker = ResNet(X.shape[1], Y.shape[1], device="cuda:0", multiple=False)
    # ranker = ranker.fit(X, Y)
    # ranker.show_history()

    # ranker = RankNet(X.shape[1], Y.shape[1], hidden_layers=(256, 256, 256, 256, 256, 256, 128, 128, 128, 128, 128, 128, 64, 64, 64, 64, 64, 64), device="cuda:0", multiple=False)
    # ranker = ranker.fit(X, Y)
    # ranker.show_history()
    print(ranker.predict(np.random.uniform(size=(5, 8))))
