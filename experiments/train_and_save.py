import json
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def create_metadataset(metafeatfile, benchmarkfile):
    with open(benchmarkfile, "r", encoding="utf-8") as f:
        benchmark_dict = json.load(f)
        benchmark_dict = {k: {k2: v2["score"] for k2, v2 in v.items(
        ) if v2["score"] > 0} for k, v in benchmark_dict.items()}
        benchmark_dict = {k: v for k,
                          v in benchmark_dict.items() if len(v) > 0}
        
    benchmark_df = pd.DataFrame.from_dict(benchmark_dict, orient="index").sample(len(benchmark_dict))
    benchmark_df = benchmark_df.fillna(-1)
    benchmark_df = benchmark_df[benchmark_df.max(axis=1) >= 0.7]

    meta_df = pd.read_csv(metafeatfile, index_col="id")
    meta_df = meta_df.loc[benchmark_df.index, :]

    return meta_df.to_numpy(), benchmark_df.to_numpy(), meta_df.index.values, meta_df.columns.values, benchmark_df.columns.values


if __name__ == "__main__":
    import argparse
    import os
    import pickle
    import sys

    import torch

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from meta_learners import ALL_MODELS

    np.random.seed(12333)

    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    parser = argparse.ArgumentParser(
        description='Train the meta-learners given a benchmark file (containing similarity measures pairs performances) and a meta-features file containing datasets meta-features')
    parser.add_argument("-m", "--metafeatfile",
                        help="Path to the file containing datasets meta-features")
    parser.add_argument("-b", "--benchmarkfile",
                        help="Path to the file containing similarity measures pairs performances")
    parser.add_argument("-o", "--outputdir",
                        help="Path to the output directory where model will be save")
    args = parser.parse_args()

    X, Y, ids, meta_features, similarity_pairs = create_metadataset(
        args.metafeatfile, args.benchmarkfile)
    print("Number of instances:", X.shape[0])
    print("Number of meta-features:", X.shape[1])
    print("Number of similarity measures pairs:", Y.shape[1])
    groups = np.array([dataset_id.split("_")[0] for dataset_id in ids])

    sc = StandardScaler().fit(X)
    print(sc.get_params())
    X = sc.transform(X)

    with open(os.path.join(args.outputdir, f"scaler.pickle"), "wb") as f:
        pickle.dump(sc, f)

    model_names = ["KNN", "PR-DTree"]
    model_types = dict(zip(model_names, ["KNN", "PR-DTree"]))

    for model_name in model_names:
        print()
        print(f"Training of model: {model_name}")
        model = ALL_MODELS[model_types[model_name]]()        
        model = model.cross_val_fit(X, Y, n_splits=5, groups=groups)
        print(model.get_params())

        model.similarity_pairs_ = similarity_pairs
        model.meta_features_ = meta_features
        with open(os.path.join(args.outputdir, f"{model_name}.pickle"), "wb") as f:
            pickle.dump(model, f)
