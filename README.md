# SIMREC
SIMREC is a **SIM**ilarity measure **REC**ommendation system for mixed data clustering algorithms. Given an input query composed of a mixed dataset, a mixed data clustering (MDC) algorithm, and a cluster validity index (CVI) to be optimized, SIMREC predicts a ranking of the similarity measure pairs according to their performances for the input query. We present here an overview of how SIMREC works, give some examples of practical use cases, and describe the organization of this repository.

## Overview of SIMREC

SIMREC is composed of two modules:

1. A meta-feature extraction module that computes the meta-feature vector of the input dataset.
2. A ranking module that takes as input the meta-feature vector computed by the meta-feature extraction module and predicts the ranking of the similarity measure pairs for the input algorithm and CVI.

For more information about how SIMREC works, please refer to the paper [here]().

## A simple example
To get started, here is an example of using SIMREC to recommend suitable similarity measure pairs on a new dataset for the K-Prototypes algorithm and the clustering accuracy.

``` python
import simrec # the simrec module which implements the recommendation function
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

We use the function `recommend(...)` from the `simrec` module to perform the recommendations. This function takes as input:
- the mixed dataset (as 2 matrices `Xnum` and `Xcat` representing the numeric and categorical parts of the dataset), MDC algorithm, and CVI for which the recommendation needs to be done.
- the path to the directory where the recommendation models are saved.
- the number $k$ of similarity pairs to recommend.

## Organization of the repository

:file_folder:simrec
├── :file_folder:[clustering_algorithms](clustering_algorithms/)
├── :file_folder:[example_datasets](example_datasets/)
├── :file_folder:[loading_datasets](loading_datasets/)
├── :file_folder:[meta_dataset_creation](meta_dataset_creation/)
├── :file_folder:[meta_model_training](meta_model_training/)
├── :file_folder:[metrics](metrics/)
├── :file_folder:[paper](paper/)
├── :notebook:example.ipynb
├── :page_facing_up:meta_features.py
├── :page_facing_up:meta_model.py
├── :page_facing_up:simrec.py
├── :page_facing_up:utils.py

- The `clustering_algorithms` folder contains the implementations of the considered MDC algorithms.
- The `example_datasets` folder contains some example mixed datasets that can be used for illustration purposes and to get started with the system.
- The `loading_datasets` folder contains code for loading datasets from tho [OpenML repository](https://www.openml.org/).
- The `meta_dataset_creation` folder contains the code for the creation of the meta-dataset. This includes the computation of the meta-features of all retrieved datasets and evaluation of the different clustering algorithms and similarity measure pairs on these datasets with different CVIs.
- The `meta_model_training` folder contains the code for training the recommendation models using the created meta-dataset.
- `metrics` implements the different numeric and categorical similarity measures.
- `paper` contains the code to generate the illustration presented in the paper.
- `example.ipynb` is a notebook containing an illustrative example of how to perform similarity measure recommendation with SIMREC.
- `meta_features.py` implements the meta-features.
- `meta_model.py` implements the recommendation models. These models are responsible for predicting the ranking of the similarity measure pairs given the meta-features of a dataset.
- `simrec.py` is the main module of the recommendation system. It implements the function `recommend(...)` which produces the recommendations given a dataset, algorithm, and CVI.
- `utils.py` contains utility functions.
