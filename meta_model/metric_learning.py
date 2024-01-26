import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import os

class ContrastiveML(nn.Module):
    def __init__(self, mapping_function, margin) -> None:
        super().__init__()
        self.mapping_function = mapping_function
        self.margin = margin

    def forward(self, X):
        return self.mapping_function(X)
    
    def _cost(self, mapping1, mapping2):
        return torch.norm(mapping1 - mapping2, dim=1)
    
    def _contrastive_loss(self, distance, y):
        ls = 0.5*torch.pow(distance, 2)
        ld = torch.maximum(torch.zeros_like(distance), self.margin - distance)
        ld = 0.5*torch.pow(ld, 2)
        loss = (1 - y)*ls + y*ld
        return torch.mean(loss)
    
    def fit(self, train_loader, optimizer, epochs=20, \
            val_loader=None, shuffle=True, checkpoint_epochs=None, checkpoint_dir=None):
        history = {
            "train": {
                "loss": [],
                "distance_similar": [],
                "distance_dissimilar": [],
            },
            "val": {
                "loss": [],
                "distance_similar": [],
                "distance_dissimilar": [],
            }
        }
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}:", end=" ")
            losses = []
            ds = []
            ns = 0
            dd = []
            nd = 0
            for X1, X2, y in train_loader:
                mapping1 = self.forward(X1)
                mapping2 = self.forward(X2)
                distance = self._cost(mapping1, mapping2)
                loss = self._contrastive_loss(distance, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    ds.append(torch.sum((1-y)*distance).item())
                    dd.append(torch.sum(y*distance).item())
                    ns += torch.sum(1-y).item()
                    nd += torch.sum(y).item()
                    losses.append(loss.item())

            history["train"]["loss"].append(np.mean(losses))
            history["train"]["distance_similar"].append(np.sum(ds)/ns)
            history["train"]["distance_dissimilar"].append(np.sum(dd)/nd)
            print(f"train_loss={history['train']['loss'][-1]:.3g}", end=" ")
            print(f"sim={history['train']['distance_similar'][-1]:.3g}", end=" ")
            print(f"dis={history['train']['distance_dissimilar'][-1]:.3g}", end=", " if val_loader else "\n")

            if val_loader is not None:
                with torch.no_grad():
                    losses = []
                    ds = []
                    ns = 0
                    dd = []
                    nd = 0
                    for X1, X2, y in val_loader:
                        mapping1 = self.forward(X1)
                        mapping2 = self.forward(X2)
                        distance = self._cost(mapping1, mapping2)
                        loss = self._contrastive_loss(distance, y)
                        ds.append(torch.sum((1-y)*distance).item())
                        dd.append(torch.sum(y*distance).item())
                        ns += torch.sum(1-y).item()
                        nd += torch.sum(y).item()
                        losses.append(loss.item())

                    history["val"]["loss"].append(np.mean(losses))
                    history["val"]["distance_similar"].append(np.sum(ds)/ns)
                    history["val"]["distance_dissimilar"].append(np.sum(dd)/nd)
                    print(f"val_loss={history['val']['loss'][-1]:.3g}", end=" ")
                    print(f"sim={history['val']['distance_similar'][-1]:.3g}", end=" ")
                    print(f"dis={history['val']['distance_dissimilar'][-1]:.3g}")
                scheduler.step(np.mean(losses))

            if checkpoint_epochs != None and epoch % checkpoint_epochs == 0:
                torch.save(self, os.path.join(checkpoint_dir, f"checkpoint{epoch}.pth"))
        self.history = history
        return self

class TripletML(nn.Module):
    def __init__(self, mapping_function, margin, gamma=0) -> None:
        super().__init__()
        self.mapping_function = mapping_function
        self.margin = margin
        self.gamma = gamma

    def forward(self, X):
        return self.mapping_function(X)
    
    def _cost(self, mapping_a, mapping_p, mapping_n):
        return torch.norm(mapping_a - mapping_p, dim=1), torch.norm(mapping_a - mapping_n, dim=1)
    
    def _triplet_loss(self, dp, dn):
        loss = torch.maximum(torch.zeros_like(dp), dp - dn + self.margin)
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        return torch.mean(loss) + self.gamma*l2_norm
    
    def fit(self, train_dataset, optimizer, epochs=20, batch_size=None, \
            val_dataset=None, shuffle=True):
        if batch_size is None: batch_size = len(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            dataset = train_dataset,
            batch_size = batch_size,
            shuffle = shuffle
        )

        history = {
            "train_loss": []
        }
        for epoch in tqdm(range(epochs), desc=f"Epochs"):
            # for X1, X2, y in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
            losses = []
            for Xa, Xp, Xn in train_loader:
                mapping_a = self.forward(Xa)
                mapping_p = self.forward(Xp)
                mapping_n = self.forward(Xn)

                dp, dn = self._cost(mapping_a, mapping_p, mapping_n)
                loss = self._triplet_loss(dp, dn)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            history["train_loss"].append(np.mean(losses))

            if val_dataset is not None:
                if "val_loss" not in history: history["val_loss"] = []
                val_loader = torch.utils.data.DataLoader(
                    dataset = val_dataset,
                    batch_size = len(val_dataset),
                    shuffle = shuffle
                )
                with torch.no_grad():
                    for Xa, Xp, Xn in val_loader:
                        mapping_a = self.forward(Xa)
                        mapping_p = self.forward(Xp)
                        mapping_n = self.forward(Xn)

                        dp, dn = self._cost(mapping_a, mapping_p, mapping_n)
                        loss = self._triplet_loss(dp, dn)
                    history["val_loss"].append(loss.item())
            # print(f"Loss: {loss.item():.2f}")
        self.history = history
        return self
    
class AEML(nn.Module):
    def __init__(self, encoder, decoder, margin, w1=1, w2=1) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.margin = margin
        self.w1 = w1
        self.w2 = w2

    def forward(self, X):
        return self.encoder(X)
    
    def _cost(self, mapping1, mapping2):
        return torch.norm(mapping1 - mapping2, dim=1)
    
    def _contrastive_loss(self, distance, y):
        ls = 0.5*torch.pow(distance, 2)
        ld = torch.maximum(torch.zeros_like(distance), self.margin - distance)
        ld = 0.5*torch.pow(ld, 2)
        loss = (1 - y)*ls + y*ld
        return torch.mean(loss)
    
    def fit(self, train_loader, optimizer, epochs=20, \
            val_loader=None, shuffle=True):

        history = {
            "train": {
                "contrastive_loss": [],
                "reconstruction_loss": [],
                "loss": [],
                "distance_similar": [],
                "distance_dissimilar": [],
            },
            "val": {
                "contrastive_loss": [],
                "reconstruction_loss": [],
                "loss": [],
                "distance_similar": [],
                "distance_dissimilar": [],
            }
        }
        mseloss = nn.MSELoss()
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}:", end=" ")
            closses = []
            rlosses = []
            tlosses = []
            ds = []
            dd = []
            for X1, X2, y in train_loader:
                mapping1 = self.forward(X1)
                mapping2 = self.forward(X2)
                Xr1 = self.decoder(mapping1)
                Xr2 = self.decoder(mapping2)

                distance = self._cost(mapping1, mapping2)
                contastive_loss = self._contrastive_loss(distance, y)
                reconstruction_loss = 0.5*mseloss(X1, Xr1) + 0.5*mseloss(X2, Xr2)
                loss = self.w1*contastive_loss + self.w2*reconstruction_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    ds.append(torch.sum((1-y)*distance).item())
                    dd.append(torch.sum((y)*distance).item())
                    closses.append(contastive_loss.item())
                    rlosses.append(reconstruction_loss.item())
                    tlosses.append(loss.item())

            history["train"]["contrastive_loss"].append(np.mean(closses))
            history["train"]["reconstruction_loss"].append(np.mean(rlosses))
            history["train"]["loss"].append(np.mean(tlosses))
            history["train"]["distance_similar"].append(np.sum(ds)/len(train_dataset))
            history["train"]["distance_dissimilar"].append(np.sum(dd)/len(train_dataset))
            print(f"train_loss={history['train']['loss'][-1]:.3g}", end=", " if val_loader else "\n")

            if val_loader is not None:
                with torch.no_grad():
                    for X1, X2, y in val_loader:
                        mapping1 = self.forward(X1)
                        mapping2 = self.forward(X2)
                        Xr1 = self.decoder(mapping1)
                        Xr2 = self.decoder(mapping2)
                        distance = self._cost(mapping1, mapping2)
                        contastive_loss = self._contrastive_loss(distance, y)
                        reconstruction_loss = self._reconstruction_loss(X1, X2, Xr1, Xr2)
                        loss = self.w1*contastive_loss + self.w2*reconstruction_loss
                    history["val"]["contrastive_loss"].append(contastive_loss.item())
                    history["val"]["reconstruction_loss"].append(reconstruction_loss.item())
                    history["val"]["loss"].append(loss.item())
                    history["val"]["distance_similar"].append(torch.mean((1-y)*distance).item())
                    history["val"]["distance_dissimilar"].append(torch.mean((y)*distance).item())
                    print(f"val_loss={history['train']['loss'][-1]:.3g}")
        self.history = history
        return self

class AE(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X):
        return self.encoder(X)
    
    def fit(self, train_loader, optimizer, epochs=20, \
            val_loader=None, shuffle=True, verbose=0):

        history = {
            "train": {
                "loss": []
            },
            "val": {
                "loss": []
            }
        }
        loss_function = nn.MSELoss()
        for epoch in range(epochs):
            if verbose > 0:
                print(f"Epoch {epoch+1}/{epochs}:", end=" ")
            losses = []
            for X, in train_loader:
                mapping = self.forward(X)
                Xr = self.decoder(mapping)
                loss = loss_function(X, Xr)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            history["train"]["loss"].append(np.mean(losses))
            if verbose > 0:
                print(f"train_loss={history['train']['loss'][-1]:.3g}", end=", " if val_loader else "\n")
            if val_loader is not None:
                with torch.no_grad():
                    losses = []
                    for X, in val_loader:
                        mapping = self.forward(X)
                        Xr = self.decoder(mapping)
                        loss = loss_function(X, Xr)
                        losses.append(loss.item())
                    history["val"]["loss"].append(np.mean(losses))
                if verbose > 0:
                    print(f"val_loss={history['val']['loss'][-1]:.3g}")
        self.history = history
        return self
    
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import pairwise_distances
    from sklearn.preprocessing import minmax_scale

    if torch.cuda.is_available():  
        device = "cuda:0"
    else:  
        device = "cpu"

    # X, y = make_blobs(n_samples=1000, n_features=16, centers=5, random_state=0)
    # X = np.c_[X, np.random.rand(X.shape[0], 32)]
    # X = minmax_scale(X)
    # X = torch.tensor(X, device=device).float()

    # dataset = torch.utils.data.TensorDataset(X)
    # train_set, val_set = torch.utils.data.random_split(dataset, (0.8, 0.2))
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=32)
    # val_loader = torch.utils.data.DataLoader(val_set, batch_size=32)

    # network = nn.Sequential(
    #     nn.Linear(X.shape[1], 16),
    #     nn.ReLU(),
    #     nn.Linear(16, 8),
    #     nn.ReLU(),
    #     nn.Linear(8, 4),
    #     nn.ReLU(),
    #     nn.Linear(4, 2)
    # )
    # network.to(device)

    # decoder = nn.Sequential(
    #     nn.Linear(2, 4),
    #     nn.ReLU(),
    #     nn.Linear(4, 8),
    #     nn.ReLU(),
    #     nn.Linear(8, 16),
    #     nn.ReLU(),
    #     nn.Linear(16, X.shape[1])
    # )
    # decoder.to(device)

    # model = AE(network, decoder)
    # optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    # model.fit(train_loader, optimizer, epochs=100, val_loader=val_loader)

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(7, 3))
    # plt.subplot(1, 2, 1)
    # plt.plot(model.history["train"]["loss"], label="train")
    # plt.plot(model.history["val"]["loss"], label="val")
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # mapping = model(X).cpu().detach().numpy()
    # mapping_train, y_train = mapping[train_set.indices], y[train_set.indices]
    # mapping_val, y_val = mapping[val_set.indices], y[val_set.indices]
    # plt.scatter(mapping_train[:, 0], mapping_train[:, 1], marker='o', s=5, c=y_train, alpha=0.5)
    # plt.scatter(mapping_val[:, 0], mapping_val[:, 1], marker='^', s=5, c=y_val)
    # plt.tight_layout()
    # plt.show()

    k = 5
    n_features = 16
    X, y = make_blobs(n_samples=500, n_features=n_features, centers=5, random_state=0)
    X = np.c_[X, np.random.rand(X.shape[0], 32)]
    X = minmax_scale(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    X1, X2, y = [], [], []
    k = 5
    D = pairwise_distances(X_train[:, :n_features])
    np.fill_diagonal(D, np.inf)
    nearest_neighbors = np.argsort(D, axis=1)[:,:k]
    for i in range(X_train.shape[0]):
        for j in nearest_neighbors[i]:
            X1.append(X_train[i])
            X2.append(X_train[j])
            y.append(0)
    for i in range(X_train.shape[0]-1):
        for j in range(i, X_train.shape[0]):
            if i not in nearest_neighbors[j] and j not in nearest_neighbors[i]:
                X1.append(X_train[i])
                X2.append(X_train[j])
                y.append(1)
    X1, X2, y = np.array(X1), np.array(X2), np.array(y)
    print("classes distribution in train set:", np.unique(y, return_counts=True))
    train_set = torch.utils.data.TensorDataset(
        torch.tensor(X1, device=device).float(),
        torch.tensor(X2, device=device).float(),
        torch.tensor(y, device=device).float()
    )

    X1, X2, y = [], [], []
    D = pairwise_distances(X_val[:, :n_features], X_train[:, :n_features])
    nearest_neighbors = np.argsort(D, axis=1)[:,:k]
    for i in range(X_val.shape[0]):
        for j in range(X_train.shape[0]):
            X1.append(X_val[i])
            X2.append(X_train[j])
            y.append(0 if j in nearest_neighbors[i] else 1)
    X1, X2, y = np.array(X1), np.array(X2), np.array(y)
    print("classes distribution in train set:", np.unique(y, return_counts=True))
    val_set = torch.utils.data.TensorDataset(
        torch.tensor(X1, device=device).float(),
        torch.tensor(X2, device=device).float(),
        torch.tensor(y, device=device).float()
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=512)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=512)

    network = nn.Sequential(
        nn.Linear(X.shape[1], 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 2)
    )
    network.to(device)

    model = ContrastiveML(network, margin=1)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    model.fit(train_loader, optimizer, epochs=10, val_loader=val_loader)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 3))
    plt.subplot(1, 2, 1)
    plt.plot(model.history["train"]["loss"], label="train")
    plt.plot(model.history["val"]["loss"], label="val")
    plt.legend()

    plt.subplot(1, 2, 2)
    mapping_train = model(torch.tensor(X_train, device=device).float()).cpu().detach().numpy()
    mapping_val = model(torch.tensor(X_val, device=device).float()).cpu().detach().numpy()
    plt.scatter(mapping_train[:, 0], mapping_train[:, 1], marker='o', s=5, c=y_train, alpha=0.5)
    plt.scatter(mapping_val[:, 0], mapping_val[:, 1], marker='^', s=5, c=y_val)
    plt.tight_layout()
    plt.savefig("fig.png")
    plt.show()