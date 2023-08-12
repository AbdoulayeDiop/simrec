# simrec
**SIM**ilarity measures **REC**ommendation in mixed data clustering.

This repository implements a meta-learning approach for similarity measures recommendation. It mainly consist in building, for each considered mixed data clustering algorithm, a machine learning model able to predict the ranking of similarity measures pairs according to the characteristics of the input dataset. These characteristics are called **meta-features**. 

There are two main steps for building such a model :

1. Create a **meta-dataset**: a meta-dataset is a dataset of datasets. Each record of the meta-dataset is a dataset represented by a vector that contains its meta-features (the predictive attributes) and the performances of the similarity measures pairs (target attributes).
2. Learn to predict ranking of similarity measure pair: in this step a machine learning model, called **meta-learner**, is trained on the meta-dataset to predict the ranking of similarity measures pairs according to the meta-features of the datasets.

The trained meta-learners can then be used on new datasets for similarity measures recommendation by predicting the ranking of the similarity measures pairs.

## We provide

- Needed materials in order to reproduce these two steps for any clustering algorithm (including the ones we already considered). Find more details in the folder [experiments](experiments/).
- Several meta-learners already trained (you can find in the folder [models](models/)) that can be used directly for similarity measures recommendation. For now, these meta-learners are available for two mixed data clustering algorithms:
    - K-Medoids
    - Hierarchical clustering with average linkage

A detailed example for practical usage of these meta-learners is available in the [example notebook](example.ipynb).

## A simple example
Here is an example of using a pre-trained meta-learner (_MDTRee_) in order to predict the ranking of similarity measures pairs on a new dataset for K-Medoids algorithm.

``` python
import numpy as np
import pickle
from mixed_metrics import get_valid_similarity_pairs
from meta_features import compute_meta_features
from sklearn.preprocessing import minmax_scale

# load the model
with open("models/KMedoids/KNN.pickle", "rb") as f:
    ranker = pickle.load(f)

# load the scaler
with open("models/KMedoids/scaler.pickle", "rb") as f:
    scaler = pickle.load(f)

# load your dataset. Here we create a random mixed dataset with 10 numeric attributes and 5 categorical attributes
Xnum = np.random.rand(200, 10)
Xcat = np.random.randint(8, size=(200, 5))

# Important: Normalize the numeric part before computing the meta-features or performing clustering
Xnum = minmax_scale(Xnum)

# create the meta-features vector of your dataset
mf_vector = compute_meta_features(Xnum, Xcat)

# predict the ranks of all similarity measures pairs
y_pred = ranker.predict(scaler.transform([mf_vector]))[0]

# get a ranked list of similarity measures pairs
ranked_pairs = ranker.similarity_pairs_[np.argsort(-y_pred)]

# keep only valid similarity measures pairs for your dataset
valid_pairs = get_valid_similarity_pairs(Xnum, Xcat)
ranked_pairs = [sim_pair for sim_pair in ranked_pairs if sim_pair in valid_pairs]

# show the top-5 similarity measures pairs
print(ranked_pairs[:5])
```

Here is the output:

``` python
['canberra_russellrao', 'canberra_eskin', 'canberra_kulsinski', 'canberra_hamming', 'canberra_dice']
```
