import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import os

class ContrastiveML(nn.Module):
    def __init__(self, mapping_function, margin, gamma=0) -> None:
        super().__init__()
        self.mapping_function = mapping_function
        self.margin = margin
        self.gamma = gamma

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
    
    def fit(self, train_dataset, optimizer, epochs=20, batch_size=None, \
            test_dataset=None, shuffle=True, checkpoint_epochs=None, checkpoint_dir=None):
        if batch_size is None: batch_size = len(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            dataset = train_dataset,
            batch_size = batch_size,
            shuffle = shuffle
        )

        history = {
            "train": {
                "loss": [],
                "distance_similar": [],
                "distance_dissimilar": [],
            },
            "test": {
                "loss": [],
                "distance_similar": [],
                "distance_dissimilar": [],
            }
        }
        for epoch in range(1, epochs+1):
            print(f"Epoch {epoch}/{epochs}", end="\t-\t")
            # for X1, X2, y in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
            losses = []
            ds = []
            ns = 0
            dd = []
            nd = 0
            for X1, X2, y in train_loader:
                mapping1 = self.forward(X1)
                mapping2 = self.forward(X2)
                distance = self._cost(mapping1, mapping2)

                l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
                loss = self._contrastive_loss(distance, y) + self.gamma*l2_norm

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
            print(f"train:  loss={history['train']['loss'][-1]:.2g}", end=", ")
            print(f"dist_sim={history['train']['distance_similar'][-1]:.2g}", end=", ")
            print(f"dist_dissim={history['train']['distance_dissimilar'][-1]:.2g}", end="\t-\t")
            print()

            if test_dataset is not None:
                test_loader = torch.utils.data.DataLoader(
                    dataset = test_dataset,
                    batch_size = batch_size,
                    shuffle = shuffle
                )
                losses = []
                ds = []
                ns = 0
                dd = []
                nd = 0
                with torch.no_grad():
                    for X1, X2, y in test_loader:
                        mapping1 = self.forward(X1)
                        mapping2 = self.forward(X2)
                        distance = self._cost(mapping1, mapping2)

                        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
                        loss = self._contrastive_loss(distance, y) + self.gamma*l2_norm

                        ds.append(torch.sum((1-y)*distance).item())
                        dd.append(torch.sum(y*distance).item())
                        ns += torch.sum(1-y).item()
                        nd += torch.sum(y).item()
                        losses.append(loss.item())

                    history["test"]["loss"].append(np.mean(losses))
                    history["test"]["distance_similar"].append(np.sum(ds)/ns)
                    history["test"]["distance_dissimilar"].append(np.sum(dd)/nd)
                    print(f"test:  loss={history['test']['loss'][-1]:.2g}", end=", ")
                    print(f"dist_sim={history['test']['distance_similar'][-1]:.2g}", end=", ")
                    print(f"dist_dissim={history['test']['distance_dissimilar'][-1]:.2g}")

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
            test_dataset=None, shuffle=True):
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

            if test_dataset is not None:
                if "test_loss" not in history: history["test_loss"] = []
                test_loader = torch.utils.data.DataLoader(
                    dataset = test_dataset,
                    batch_size = len(test_dataset),
                    shuffle = shuffle
                )
                with torch.no_grad():
                    for Xa, Xp, Xn in test_loader:
                        mapping_a = self.forward(Xa)
                        mapping_p = self.forward(Xp)
                        mapping_n = self.forward(Xn)

                        dp, dn = self._cost(mapping_a, mapping_p, mapping_n)
                        loss = self._triplet_loss(dp, dn)
                    history["test_loss"].append(loss.item())
            # print(f"Loss: {loss.item():.2f}")
        self.history = history
        return self
    
class AEML(nn.Module):
    def __init__(self, encoder, decoder, margin, w1=1, w2=1, gamma=0) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.margin = margin
        self.w1 = w1
        self.w2 = w2
        self.gamma = gamma

    def forward(self, X):
        return self.encoder(X)
    
    def _cost(self, mapping1, mapping2):
        return torch.norm(mapping1 - mapping2, dim=1)
    
    def _contrastive_loss(self, distance, y):
        ls = 0.5*torch.pow(distance, 2)
        ld = torch.maximum(torch.zeros_like(distance), self.margin - distance)
        ld = 0.5*torch.pow(ld, 2)
        loss = (1 - y)*ls + y*ld
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())
        return torch.mean(loss) + self.gamma*l2_norm
    
    def _reconstruction_loss(self, X1, X2, Xr1, Xr2):
        loss = 0.5*torch.sum((X1 - Xr1)**2, dim=1) + 0.5*torch.sum((X2 - Xr2)**2, dim=1)
        return torch.mean(loss)
    
    def fit(self, train_dataset, optimizer, epochs=20, batch_size=None, \
            test_dataset=None, shuffle=True):
        if batch_size is None: batch_size = len(train_dataset)
        train_loader = torch.utils.data.DataLoader(
            dataset = train_dataset,
            batch_size = batch_size,
            shuffle = shuffle
        )

        history = {
            "train": {
                "contrastive_loss": [],
                "reconstruction_loss": [],
                "total_loss": [],
                "distance_similar": [],
                "distance_dissimilar": [],
            },
            "test": {
                "contrastive_loss": [],
                "reconstruction_loss": [],
                "total_loss": [],
                "distance_similar": [],
                "distance_dissimilar": [],
            }
        }
        for epoch in tqdm(range(epochs), desc=f"Epochs"):
            # for X1, X2, y in tqdm(loader, desc=f"Epoch {epoch}/{epochs}"):
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
                ds.append(torch.sum((1-y)*distance).item())
                dd.append(torch.sum((y)*distance).item())

                contastive_loss = self._contrastive_loss(distance, y)
                reconstruction_loss = self._reconstruction_loss(X1, X2, Xr1, Xr2)
                closses.append(contastive_loss.item())
                rlosses.append(reconstruction_loss.item())

                loss = self.w1*contastive_loss + self.w2*reconstruction_loss
                tlosses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            history["train"]["contrastive_loss"].append(np.mean(closses))
            history["train"]["reconstruction_loss"].append(np.mean(rlosses))
            history["train"]["distance_similar"].append(np.sum(ds)/len(train_dataset))
            history["train"]["distance_dissimilar"].append(np.sum(dd)/len(train_dataset))

            if test_dataset is not None:
                test_loader = torch.utils.data.DataLoader(
                    dataset = test_dataset,
                    batch_size = len(test_dataset),
                    shuffle = shuffle
                )
                with torch.no_grad():
                    for X1, X2, y in test_loader:
                        mapping1 = self.forward(X1)
                        mapping2 = self.forward(X2)
                        Xr1 = self.decoder(mapping1)
                        Xr2 = self.decoder(mapping2)
                        distance = self._cost(mapping1, mapping2)
                        contastive_loss = self._contrastive_loss(distance, y)
                        reconstruction_loss = self._reconstruction_loss(X1, X2, Xr1, Xr2)
                        loss = self.w1*contastive_loss + self.w2*reconstruction_loss
                    history["test"]["contrastive_loss"].append(contastive_loss.item())
                    history["test"]["reconstruction_loss"].append(reconstruction_loss.item())
                    history["test"]["total_loss"].append(loss.item())
                    history["test"]["distance_similar"].append(torch.mean((1-y)*distance).item())
                    history["test"]["distance_dissimilar"].append(torch.mean((y)*distance).item())
            # print(f"Loss: {loss.item():.2f}")
        self.history = history
        return self
    
if __name__ == "__main__":
    from torchsummary import summary
    if torch.cuda.is_available():  
        device = "cuda:0"
    else:  
        device = "cpu"
    X1 = torch.randn(128, 1, 28, 28, device=device)
    X2 = torch.randn(128, 1, 28, 28, device=device)
    y = torch.randint(2, size=(128,), device=device).flatten()
    train_dataset = torch.utils.data.TensorDataset(X1, X2, y)

    network = nn.Sequential(
        nn.Conv2d(1, 15, 5, padding=0, stride=1),
        nn.MaxPool2d(3),
        nn.Conv2d(15, 30, 8),
        nn.Flatten(),
        nn.Linear(30, 2)
    )
    network.to(device)
    print(summary(network, (1, 28, 28)))

    model = ContrastiveML(network, margin=10)
    optimizer = Adam(model.parameters(), lr=1e-3)
    model.fit(train_dataset, optimizer, batch_size=32)
    import matplotlib.pyplot as plt
    plt.plot(model.history["loss"])
    plt.show()