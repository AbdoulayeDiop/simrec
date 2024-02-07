import sys
sys.path.append(".")
from metric_learning import ContrastiveML, AE
import ranking
from utils import ndcg_sim
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import torch
import torch.nn as nn

class MetaModel():
    def __init__(self, mapping_function=None, ranking_model="KNN", margin=1, gamma=0, device="cpu"):
        self.metric_learner = None if mapping_function is None else \
            ContrastiveML(mapping_function, margin, gamma=gamma)
        self.ranker = ranking.ALL_MODELS[ranking_model]()
        self.device = device

    def _create_similarity_matrix(self, Y1, Y2=None):
        return pairwise_distances(Y1, Y2, metric=lambda y1,y2: ndcg_sim(y1,y2,p=10), n_jobs=-1)

    def train_metric_learner(self, Xtrain, Ytrain, metric_learning_params):
        if self.metric_learner is None:
            return self
        sim_matrix = self._create_similarity_matrix(Ytrain)
        nearest_neighbors = np.argsort(sim_matrix, axis=1)
        X1, X2, y = [], [], []
        k = 10
        for i in range(Xtrain.shape[0]-1):
            for j in range(i+1, Xtrain.shape[0]):
                X1.append(Xtrain[i])
                X2.append(Xtrain[j])
                # similar = i in nearest_neighbors[j][:k] and\
                #     j in nearest_neighbors[i][:k]
                similar = sim_matrix[i, j] > 0.9
                y.append(0 if similar else 1)
        X1, X2, y = np.array(X1), np.array(X2), np.array(y)
        print()
        print(np.unique(y, return_counts=True))
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X1, device=self.device).float(),
            torch.tensor(X2, device=self.device).float(),
            torch.tensor(y, device=self.device).float()
        )
        
        self.metric_learner.to(self.device)
        optimizer = torch.optim.Adam(self.metric_learner.parameters(), lr=metric_learning_params["lr"])
        metric_learning_params.pop("lr")
        self.metric_learner.fit(train_dataset, optimizer, **metric_learning_params)
        return self

    def embbed(self, X):
        if self.metric_learner is None:
            return X
        with torch.no_grad():
            Z = self.metric_learner(torch.tensor(X, device=self.device).float()).cpu().detach().numpy()
        return Z

    def train_ranker(self, Ztrain, Ytrain):
        self.ranker = self.ranker.cross_val_fit(Ztrain, Ytrain)
        return self


    def fit(self, Xtrain, Ytrain, metric_learning_params):
        self.train_metric_learner(Xtrain, Ytrain, metric_learning_params)
        Ztrain = self.embbed(Xtrain)
        self.train_ranker(Ztrain, Ytrain)
        return self

    def predict(self, X):
        return self.ranker.predict(self.embbed(X))

class AEKNN():
    def __init__(self, encoder=None, decoder=None, n_neighbors=5, metric="euclidean", weights="uniform", device="cpu"):
        self.ae = AE(encoder, decoder)
        self.knn = KNeighborsRegressor(n_neighbors=n_neighbors, metric=metric, weights=weights)
        self.device = device

    def train_ae(self, X_train, params, X_val=None):
        train_set = torch.utils.data.TensorDataset(torch.tensor(X_train, device=self.device).float())
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=params["batch_size_train"])
        
        val_loader = None
        if X_val is not None:
            val_set = torch.utils.data.TensorDataset(torch.tensor(X_val, device=self.device).float())
            val_loader = torch.utils.data.DataLoader(val_set, batch_size=params["batch_size_val"])
        
        self.ae.to(self.device)
        optimizer = torch.optim.Adam(
            self.ae.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"]
        )
        self.ae.fit(
            train_loader, optimizer,
            val_loader=val_loader,
            epochs=params["epochs"]
        )
        return self

    def embbed(self, X):
        if self.ae is None:
            return X
        with torch.no_grad():
            Z = self.ae(torch.tensor(X, device=self.device).float()).cpu().detach().numpy()
        return Z

    def train_knn(self, Ztrain, Y_train):
        self.knn = self.knn.fit(Ztrain, Y_train)
        return self


    def fit(self, X_train, Y_train, params, X_val=None):
        self.train_ae(X_train, params, X_val=X_val)
        Ztrain = self.embbed(X_train)
        self.train_knn(Ztrain, Y_train)
        return self

    def predict(self, X):
        return self.knn.predict(self.embbed(X))

if __name__=="__main__":
    from sklearn.datasets import make_moons, make_circles
    from torchsummary import summary
    import matplotlib.pyplot as plt
    if torch.cuda.is_available():  
        device = "cuda:0"
    else:  
        device = "cpu"
    print("device:", device)
    n_samples = 100
    X, y = make_circles(n_samples=n_samples)
    new_y = np.array([np.array([3, 1, 1]) + 0.5*np.random.rand(3) if v==0 else \
        np.array([1, 2, 3]) + 0.5*np.random.rand(3) for v in y])

    network = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    )
    network.to(device)
    print(summary(network, (2,)))

    model = MetaModel(network, margin=1, device=device, gamma=1e-4)
    metric_learning_params = {
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 64
    }
    model.fit(X, new_y, metric_learning_params)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(model.metric_learner.history["train"]["loss"], label="loss")
    plt.plot(model.metric_learner.history["train"]["distance_similar"], label="distance_similar")
    plt.plot(model.metric_learner.history["train"]["distance_dissimilar"], label="distance_dissimilar")
    plt.legend()
    plt.subplot(1, 2, 2)
    Z = model.embbed(X)
    plt.scatter(Z[:, 0], Z[:, 1], s=10, c=y)
    plt.savefig("fig.png")