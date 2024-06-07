# SIMREC
SIMREC is a **SIM**ilarity measures **REC**ommendation system for mixed data clustering algorithms. Given an input query composed of a mixed dataset, a mixed data clustering (MDC) algorithm, and a cluster validity index (CVI) to be optimized, SIMREC predicts a ranking of the similarity measure pairs according to their performances for the input query. We present here an overview of how SIMREC works, give some example of practical use cases, and describe the organization of this repository.

## Overview of SIMREC


## A simple example
Here is an example of using a pre-trained meta-learner (_KNN_) in order to predict the ranking of similarity measures pairs on a new dataset for K-Prototypes algorithm.

``` python
import simrec as np
import pickle
from mixed_metrics import get_valid_similarity_pairs
from meta_features import compute_meta_features
from sklearn.preprocessing import minmax_scale

# load the model
with open("models/KPrototypes/KNN.pickle", "rb") as f:
    ranker = pickle.load(f)

# load the scaler
with open("models/KPrototypes/scaler.pickle", "rb") as f:
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
['mahalanobis_of', 'mahalanobis_russellrao', 'mahalanobis_hamming', 'mahalanobis_kulsinski', 'divergence_of']
```
