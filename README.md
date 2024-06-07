# SIMREC
SIMREC is a **SIM**ilarity measures **REC**ommendation system for mixed data clustering algorithms. Given an input query composed of a mixed dataset, a mixed data clustering (MDC) algorithm, and a cluster validity index (CVI) to be optimized, SIMREC predicts a ranking of the similarity measure pairs according to their performances for the input query. We present here an overview of how SIMREC works, give some example of practical use cases, and describe the organization of this repository.

## Overview of SIMREC

SIMREC is composed of two modules:

1. A meta-feature extraction module that compute the meta-feature vector of the input dataset.
2. An ranking module that takes as input the meta-feature vector computed by the meta-feature extraction module and predicts the ranking of the similarity measure pairs for the input algorithm and CVI.

## A simple example
Here is an example of using SIMREC to recommend suitable similarity measure pairs on a new dataset for the K-Prototypes algorithm and the clustering accuracy.

``` python
import simrec
import numpy as np

# load your dataset. Here we create a random mixed dataset with 10 numeric attributes and 5 categorical attributes
Xnum = np.random.rand(200, 10)
Xcat = np.random.randint(8, size=(200, 5))

# define the models directory, algorithm and the CVI to be optimized.
models_dir = "meta_model_training/data/saved_models/"
algorithm = "kprototypes"
cvi = "acc"

# recommend the 5 top performing similarity pairs
recommendation = simrec.recommend(Xnum, Xcat, models_dir, algorithm=algorithm, cvi=cvi, k=5)

# show the recommendation
print(recommendation)
```

Here is the output:

``` python
[('lorentzian_of', 0.7497378130452476),
 ('euclidean_eskin', 0.7496881002259083),
 ('sqeuclidean_sokalsneath', 0.7491985225953831),
 ('manhattan_eskin', 0.7486684895711725),
 ('manhattan_of', 0.748400729930295)]
```
