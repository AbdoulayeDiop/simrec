import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json
import time

dataset_statistics = ["n_instances", "n_features", "dim", "num_on_cat", "n_num_att", "n_cat_att"]
attributes_statistics = [
    f"{p}_{name}" for p in ["min", "q1", "mean", "q3", "max"] for name in [
        "means_num_att", "std_num_att", #"kurtosis_num_att", "skewness_num_att",
        "card_cat_att", "entropy_cat_att",
        "covariance",
    ]
]
isolation_forest = [f"isolation_forest_{i}" for i in range(10)]
proposed_attributes_statistics = [
    f"{p}_{name}" for p in ["min", "q1", "mean", "q3", "max"] for name in [
        "means_squared_num_att", "std_squared_num_att", #"kurtosis_squared_num_att", "skewness_squared_num_att",
        "means_internal_product_num_att", "std_internal_product_num_att", #"kurtosis_internal_product_num_att", "skewness_internal_product_num_att",
        "mutual_info_cat_att",
        "std_freq_cat_att"
    ]
]

def create_metadataset(metafeatfile, benchmarkfile):
    with open(benchmarkfile, "r", encoding="utf-8") as f:
        benchmark_dict = json.load(f)
        benchmark_dict = {k: {k2: v2["score"] for k2, v2 in v.items() if v2["score"]>0} for k, v in benchmark_dict.items()}
        # benchmark_dict = {k: v for k, v in benchmark_dict.items() if int(k.split("_")[1]) == 0}
        benchmark_dict = {k: v for k, v in benchmark_dict.items() if len(v) > 0}
    benchmark_df = pd.DataFrame.from_dict(benchmark_dict, orient="index").sample(len(benchmark_dict))
    benchmark_df = benchmark_df.fillna(-1)

    meta_df = pd.read_csv(metafeatfile, index_col="id")
    meta_df = meta_df.loc[benchmark_df.index, dataset_statistics + attributes_statistics + proposed_attributes_statistics]

    return meta_df.to_numpy(), benchmark_df.to_numpy(), meta_df.index.values, meta_df.columns.values, benchmark_df.columns.values

if __name__=="__main__":
    import argparse
    from meta_learners import ALL_MODELS
    import os
    import pickle
    import torch

    np.random.seed(12333)

    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    parser = argparse.ArgumentParser(description='Train the meta-learners given a benchmark file (containing similarity measures pairs performances) and a meta-features file containing datasets meta-features')
    parser.add_argument("-m", "--metafeatfile", help="Path to the file containing datasets meta-features")
    parser.add_argument("-b", "--benchmarkfile", help="Path to the file containing similarity measures pairs performances")
    parser.add_argument("-o", "--outputdir", help="Path to the output directory where model will be save")
    args = parser.parse_args()

    X, Y, ids, meta_features, similarity_pairs = create_metadataset(args.metafeatfile, args.benchmarkfile)
    print("Number of instances:", X.shape[0])
    print("Number of meta-features:", X.shape[1])
    print("Number of similarity measures pairs:", Y.shape[1])
    groups = np.array([dataset_id.split("_")[0] for dataset_id in ids])
    
    sc = StandardScaler().fit(X)
    X = sc.transform(X)

    model_names = ["LR", "KNN", "DTree", "MKNN", "MDTree", "RankNetMSE", "PR-LR", "PR-KNN", "PR-DTree"] #, "RT-K", "RT-NDCG"
    model_types = dict(zip(model_names, ["LR", "KNN", "DTree", "MKNN", "MDTree", "RankNet", "PR-LR", "PR-KNN", "PR-DTree"])) #, "RT", "RT"
    
    default_params = {
        "RT-K": {"min_samples_split":3, "n_jobs":-1},
        "RT-NDCG": {"min_samples_split": 3, 'rank_sim': 'ndcg', "n_jobs":-1},
        "RankNetMSE": {
            "input_dim": X.shape[1], 
            'output_dim': Y.shape[1], 
            'hidden_layers': (1024, 2048, 1024), 
            'device': device
        }
    }
    
    for model_name in model_names:
        print()
        print(f"Training of model: {model_name}")
        model = ALL_MODELS[model_types[model_name]]() if model_name not in default_params \
            else ALL_MODELS[model_types[model_name]](**default_params[model_name])
        start = time.time()
        if model_name in ["RankNetMSE"]:
            model = model.fit(X, Y, batch_size=64, epochs=150)
        elif model_name in ["LR", "PR-LR", "RT-K", "RT-NDCG"]:
            model = model.fit(X, Y)
        else:
            print("Grid Search CV...")
            model = model.cross_val_fit(X, Y, n_splits=4, groups=groups) # 
            print(model.get_params())
        
        with open(os.path.join(args.outputdir, f"{model_name}.pickle"), "wb") as f:
            pickle.dump(model, f)