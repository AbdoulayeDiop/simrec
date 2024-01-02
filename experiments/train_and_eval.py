import json
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

dataset_statistics = ["n_instances", "n_features", "dim", "num_on_cat", "n_num_att", "n_cat_att"]
attributes_statistics = [
    f"{p}_{name}" for p in ["min", "q1", "mean", "q3", "max"] for name in [
        "means_num_att", "std_num_att", #"kurtosis_num_att", "skewness_num_att",
        "card_cat_att", "entropy_cat_att",
        "covariance",
    ]
]

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
    # Use meta_df = meta_df.loc[benchmark_df.index, dataset_statistics+attributes_statistics]
    # to consider only the meta-features from the literature (i.e. without the proposed meta-features)

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
    parser.add_argument("-o", "--outputfile",
                        help="Path/name of the output file")
    parser.add_argument(
        "-r", "--replace", help="Update or replace result if exist", action="store_true")
    args = parser.parse_args()

    obj = {}
    if os.path.isfile(args.outputfile):
        with open(args.outputfile, "rb") as f:
            obj = pickle.load(f)

    if "X" not in obj:
        X, Y, ids, meta_features, similarity_pairs = create_metadataset(
            args.metafeatfile, args.benchmarkfile)
        print("Number of instances:", X.shape[0])
        print("Number of meta-features:", X.shape[1])
        print("Number of similarity pairs:", Y.shape[1])
        groups = np.array([dataset_id.split("_")[0] for dataset_id in ids])
        obj["X"] = X.copy()
        obj["Y"] = Y
        obj["ids"] = ids
        obj["meta_features"] = meta_features
        obj["similarity_pairs"] = similarity_pairs
        obj["groups"] = groups
    else:
        X = obj["X"].copy()
        Y = obj["Y"]
        ids = obj["ids"]
        meta_features = obj["meta_features"]
        similarity_pairs = obj["similarity_pairs"]
        groups = obj["groups"]

    sc = StandardScaler().fit(X)
    X = sc.transform(X)

    base_datasets = [i for i in range(X.shape[0]) if ids[i].split('_')[1]=='0']

    if "train_results" not in obj:
        obj["train_results"] = {}

    model_names = ["ElasticNet", "KNN", "DTree", "MKNN", "MDTree", "RF", "PR-DTree"] #"RankNetLTR", "LR", "ElasticNet", "KNN", "DTree", "MKNN", "MDTree", "RT-K", "RT-NDCG", "PR-LR", "PR-KNN", "PR-DTree", "Mtrl+RankNet", "Mtrl+KNN"
    model_types = dict(zip(model_names, ["ElasticNet", "KNN", "DTree", "MKNN", "MDTree", "RF", "PR-DTree"])) #"RankNet", "LR", "ElasticNet", "KNN", "DTree", "MKNN", "MDTree", "RT", "RT", "PR-LR", "PR-KNN", "PR-DTree", None, None
    
    default_params = {
        "RT-K": {"min_samples_split":3, "n_jobs":-1},
        "RT-NDCG": {"min_samples_split": 3, 'rank_sim': 'ndcg', "n_jobs":-1},
        "RankNetMSE": {
            "input_dim": X.shape[1],
            'output_dim': Y.shape[1], 
            'hidden_layers': (1024, 512),
            'gamma': 1e-3,
            'learning_rate': 1e-3,
            'device': device
        },
        "RankNetLTR": {
            "input_dim": X.shape[1], 
            'output_dim': Y.shape[1], 
            'hidden_layers': (1024, 512),
            'gamma': 15,
            'loss': 'ltr',
            'learning_rate': 1e-3,
            'device': device
        }
    }
    
    epsilon = 0
    for iter, i in enumerate(base_datasets):
        train_index = [j for j in range(X.shape[0]) if groups[j] != groups[i]]
        test_index = [i]
        print()
        print("##############################################")
        if f"split_{iter}" not in obj["train_results"]:
            obj["train_results"][f"split_{iter}"] = {}
            obj["train_results"][f"split_{iter}"]["train_index"] = train_index
            obj["train_results"][f"split_{iter}"]["test_index"] = test_index
            obj["train_results"][f"split_{iter}"]["predictions"] = {}
            obj["train_results"][f"split_{iter}"]["model_params"] = {}
            obj["train_results"][f"split_{iter}"]["times"] = {}
        
        new_train_index = [train_index[j] for j, y in enumerate(Y[train_index]) if max(y[y > 0]) - np.quantile(y[y > 0], 0.5) >= epsilon*max(y[y > 0])]
        print("split {} | train: {}, test: {}".format(iter, len(new_train_index), len(test_index)))
        X_train, X_test = X[new_train_index], X[test_index]
        Y_trainr, Y_testr = Y[new_train_index], Y[test_index]
        Y_train, Y_test = Y[new_train_index], Y[test_index]
        Xe, Xe_train, Xe_test = None, None, None
        for model_name in model_names:
            if model_name not in obj["train_results"][f"split_{iter}"]["predictions"]:
                print(f"Training of model: {model_name}")
                model = ALL_MODELS[model_types[model_name]]() if model_name not in default_params \
                    else ALL_MODELS[model_types[model_name]](**default_params[model_name])
                start = time.time()
                
                if model_name == "RankNetMSE":
                    model = model.fit(X_train, Y_train, X_test=X_test, Y_test=Y_test, batch_size=32, epochs=100)

                elif model_name == "RankNetLTR":
                    model = model.fit(X_train, Y_train, X_test=X_test, Y_test=Y_test, batch_size=32, epochs=100)

                elif model_name in ["LR", "RT-K", "RT-NDCG"]:
                    model = model.fit(X_train, Y_train)
                else:
                    print("Grid Search CV...")
                    model = model.cross_val_fit(X_train, Y_train, n_splits=4, groups=groups[new_train_index], verbose=60) # 
                    print(model.get_params())
                end = time.time()
                
                obj["train_results"][f"split_{iter}"]["predictions"][model_name] = model.predict(X_test)
                obj["train_results"][f"split_{iter}"]["model_params"][model_name] = model.get_params() if model_name not in default_params else default_params[model_name]
                obj["train_results"][f"split_{iter}"]["times"][model_name] = end - start
                with open(args.outputfile, "wb") as f:
                    pickle.dump(obj, f)
