import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json
from scipy.stats import iqr
from tqdm import tqdm
import time
sys.path.append("../..")

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
    from meta_learners import ALL_MODELS, NN_MODELS, ndcg
    import os
    import pickle
    from sklearn.model_selection import GroupShuffleSplit, KFold, GroupKFold
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
    parser.add_argument("-o", "--outputfile", help="Path/name of the output file")
    parser.add_argument("-r", "--replace", help="Update or replace result if exist", action="store_true")
    args = parser.parse_args()
    
    obj = {}
    if os.path.isfile(args.outputfile):
        with open(args.outputfile, "rb") as f:
            obj = pickle.load(f)

    if "X" not in obj:
        X, Y, ids, meta_features, similarity_pairs = create_metadataset(args.metafeatfile, args.benchmarkfile)
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

    if "train_results" not in obj:
        obj["train_results"] = {}

    model_names = ["LR", "KNN", "DTree", "MKNN", "MDTree", "RankNetMSE", "PR-LR", "PR-KNN", "PR-DTree"] #, "RT-K", "RT-NDCG"
    model_types = dict(zip(model_names, ["LR", "KNN", "DTree", "MKNN", "MDTree", "RankNet", "PR-LR", "PR-KNN", "PR-DTree"])) #, "RT", "RT"
    
    default_params = {
        "RT-K": {"min_samples_split":3, "n_jobs":-1},
        "RT-NDCG": {"min_samples_split": 3, 'rank_sim': 'ndcg', "n_jobs":-1},
        "RankNetMSE": {
            "input_dim": X.shape[1], 
            'output_dim': Y.shape[1], 
            'hidden_layers': (2048, 1024, 512), 
            'device': device
        }
    }
    
    # kf = KFold(n_splits=5, shuffle=True, random_state=1234)
    gkf = GroupKFold(n_splits=5)
    i = 0
    epsilon = 0
    # for train_index, test_index in kf.split(X):
    for train_index, test_index in gkf.split(X, groups=groups):
        print()
        print("##############################################")
        if f"split_{i}" not in obj["train_results"]:
            obj["train_results"][f"split_{i}"] = {}
            obj["train_results"][f"split_{i}"]["train_index"] = train_index
            obj["train_results"][f"split_{i}"]["test_index"] = test_index
            obj["train_results"][f"split_{i}"]["predictions"] = {}
            obj["train_results"][f"split_{i}"]["model_params"] = {}
            obj["train_results"][f"split_{i}"]["times"] = {}
        
        new_train_index = [train_index[i] for i, y in enumerate(Y[train_index]) if max(y[y > 0]) - np.quantile(y[y > 0], 0.5) >= epsilon*max(y[y > 0])]
        print("split {} | train: {}, test: {}".format(i, len(new_train_index), len(test_index)))
        X_train, X_test = X[new_train_index], X[test_index]
        Y_trainr, Y_testr = Y[new_train_index], Y[test_index]
        Y_train, Y_test = Y[new_train_index], Y[test_index]
        for model_name in model_names:
            if model_name not in obj["train_results"][f"split_{i}"]["predictions"]:
                print()
                print(f"Training of model: {model_name}")
                model = ALL_MODELS[model_types[model_name]]() if model_name not in default_params \
                    else ALL_MODELS[model_types[model_name]](**default_params[model_name])
                start = time.time()
                if model_name in ["RankNetMSE"]:
                    model = model.fit(X_train, Y_train, X_test=X_test, Y_test=Y_test, batch_size=32, epochs=150)
                elif model_name in ["LR", "PR-LR", "RT-K", "RT-NDCG"]:
                    model = model.fit(X_train, Y_train)
                else:
                    print("Grid Search CV...")
                    model = model.cross_val_fit(X_train, Y_train, n_splits=4, groups=groups[new_train_index]) # 
                    print(model.get_params())
                end = time.time()
                
                obj["train_results"][f"split_{i}"]["predictions"][model_name] = model.predict(X_test)
                obj["train_results"][f"split_{i}"]["model_params"][model_name] = model.get_params() if model_name not in default_params else default_params[model_name]
                obj["train_results"][f"split_{i}"]["times"][model_name] = end - start
                with open(args.outputfile, "wb") as f:
                    pickle.dump(obj, f)
        i += 1

# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import json
# from scipy.stats import iqr
# from tqdm import tqdm
# sys.path.append("../..")

# dataset_statistics = ["n_instances", "n_features", "dim", "num_on_cat", "n_num_att", "n_cat_att"]
# attributes_statistics = [
#     f"{p}_{name}" for p in ["min", "q1", "mean", "q3", "max"] for name in [
#         "means_num_att", "std_num_att", #"kurtosis_num_att", "skewness_num_att",
#         "card_cat_att", "entropy_cat_att",
#         "covariance",
#     ]
# ]
# isolation_forest = [f"isolation_forest_{i}" for i in range(10)]
# proposed_attributes_statistics = [
#     f"{p}_{name}" for p in ["min", "q1", "mean", "q3", "max"] for name in [
#         "means_squared_num_att", "std_squared_num_att", #"kurtosis_squared_num_att", "skewness_squared_num_att",
#         "means_internal_product_num_att", "std_internal_product_num_att", #"kurtosis_internal_product_num_att", "skewness_internal_product_num_att",
#         "mutual_info_cat_att",
#         "std_freq_cat_att"
#     ]
# ]

# def create_metadataset(metafeatfile, benchmarkfile):
#     with open(benchmarkfile, "r", encoding="utf-8") as f:
#         benchmark_dict = json.load(f)
#         benchmark_dict = {k: {k2: v2["score"] for k2, v2 in v.items() if v2["score"]>0} for k, v in benchmark_dict.items()}
#         # benchmark_dict = {k: v for k, v in benchmark_dict.items() if int(k.split("_")[1]) == 0}
#         benchmark_dict = {k: v for k, v in benchmark_dict.items() if len(v) > 0}
#     benchmark_df = pd.DataFrame.from_dict(benchmark_dict, orient="index").sample(len(benchmark_dict))
#     benchmark_df = benchmark_df.fillna(-1)

#     meta_df = pd.read_csv(metafeatfile, index_col="id")
#     meta_df = meta_df.loc[benchmark_df.index, dataset_statistics + attributes_statistics + proposed_attributes_statistics]

#     return meta_df.to_numpy(), benchmark_df.to_numpy(), meta_df.index.values, meta_df.columns.values, benchmark_df.columns.values

# if __name__=="__main__":
#     import argparse
#     from meta_learners import ALL_MODELS, NN_MODELS, ndcg
#     import os
#     import pickle
#     from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
#     import torch
#     import time

#     np.random.seed(12333)

#     print("GPU available:", torch.cuda.is_available())
#     if torch.cuda.is_available():
#         device = "cuda:0"
#     else:
#         device = "cpu"

#     parser = argparse.ArgumentParser(description='Train the meta-learners given a benchmark file (containing similarity measures pairs performances) and a meta-features file containing datasets meta-features')
#     parser.add_argument("-m", "--metafeatfile", help="Path to the file containing datasets meta-features")
#     parser.add_argument("-b", "--benchmarkfile", help="Path to the file containing similarity measures pairs performances")
#     parser.add_argument("-o", "--outputfile", help="Path/name of the output file")
#     parser.add_argument("-r", "--replace", help="Update or replace result if exist", action="store_true")
#     args = parser.parse_args()
    
#     obj = {}
#     if os.path.isfile(args.outputfile):
#         with open(args.outputfile, "rb") as f:
#             obj = pickle.load(f)

#     if "X" not in obj:
#         X, Y, ids, meta_features, similarity_pairs = create_metadataset(args.metafeatfile, args.benchmarkfile)
#         sc = StandardScaler().fit(X)
#         print("Number of instances:", X.shape[0])
#         print("Number of meta-features:", X.shape[1])
#         print("Number of similarity pairs:", Y.shape[1])
#         groups = np.array([dataset_id.split("_")[0] for dataset_id in ids])
#         gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1234)
#         train_index, test_index = list(gss.split(X, groups=groups))[0]
#         obj["X"] = X.copy()
#         obj["Y"] = Y
#         obj["ids"] = ids
#         obj["meta_features"] = meta_features
#         obj["similarity_pairs"] = similarity_pairs
#         obj["groups"] = groups
#         obj["train_index"] = train_index
#         obj["test_index"] = test_index
#     else:
#         X = obj["X"].copy()
#         Y = obj["Y"]
#         ids = obj["ids"]
#         meta_features = obj["meta_features"]
#         similarity_pairs = obj["similarity_pairs"]
#         groups = obj["groups"]
#         train_index = obj["train_index"]
#         test_index = obj["test_index"]
    
#     X = StandardScaler().fit_transform(X)

#     if "train_results" not in obj:
#         obj["train_results"] = {}
#     for epsilon in np.arange(0, 0.11, 0.025):
#         print("##############################################")
#         print()
        
#         if epsilon in obj["train_results"] and "new_train_index" in obj["train_results"][epsilon]:
#             new_train_index = obj["train_results"][epsilon]["new_train_index"]

#         else:
#             new_train_index = [train_index[i] for i, y in enumerate(Y[train_index]) if max(y[y > 0]) - np.quantile(y[y > 0], 0.5) >= epsilon*max(y[y > 0])]
#             if len(new_train_index) < 200 :
#                 print(f"new_train_index={len(new_train_index)} < 20. END !")
#                 break
#             obj["train_results"][epsilon] = {}
#             obj["train_results"][epsilon]["new_train_index"] = new_train_index
        
#         print(f"epsilon:", epsilon)
#         print("train: {}, test: {}".format(len(new_train_index), len(test_index)))
#         X_train, X_test = X[new_train_index], X[test_index]
#         Y_trainr, Y_testr = Y[new_train_index], Y[test_index]
#         Y_train, Y_test = Y[new_train_index], Y[test_index]

#         model_names = ["LR", "KNN", "DTree", "PR-LR", "PR-KNN", "PR-DTree", "RankNetMSE", "RankNetLTR", "RT-K", "RT-NDCG"] #
#         model_types = dict(zip(model_names, ["LR", "KNN", "DTree", "PR-LR", "PR-KNN", "PR-DTree", "RankNet", "RankNet", "RT", "RT"])) #
        
#         default_params = {
#             "RT-K": {"min_samples_split":5, "n_jobs":-1},
#             "RT-NDCG": {"min_samples_split": 5, 'rank_sim': 'ndcg', "n_jobs":-1},
#             "RankNetMSE": {
#                 "input_dim": X.shape[1], 
#                 'output_dim': Y.shape[1], 
#                 'hidden_layers': (1024, 1024, 512), 
#                 'device': device
#             },
#             "RankNetLTR": {
#                 "input_dim": X.shape[1], 
#                 'output_dim': Y.shape[1], 
#                 'hidden_layers': (1024, 1024, 512), 
#                 'loss': "ltr",
#                 'device': device
#             }
#         }

#         if "model_names" in obj["train_results"][epsilon]:
#             model_names = [k for k in model_names if k not in obj["train_results"][epsilon]["model_names"]]
#         else :
#             obj["train_results"][epsilon]["model_names"] = []
#             obj["train_results"][epsilon]["model_params"] = {}
#             obj["train_results"][epsilon]["predictions"] = {}
#             obj["train_results"][epsilon]["times"] = {}
        
#         predictions = {}
#         for model_name in model_names:
#             print()
#             print(f"Training of model: {model_name}")
#             model = ALL_MODELS[model_types[model_name]]() if model_name not in default_params \
#                 else ALL_MODELS[model_types[model_name]](**default_params[model_name])
#             start = time.time()
#             if model_name in ["RankNetMSE", "RankNetLTR"]:
#                 model = model.fit(X_train, Y_train, X_test=X_test, Y_test=Y_test, batch_size=32, epochs=150)
#                 model.show_history()
#             elif model_name in ["LR", "PR-LR", "RT-K", "RT-NDCG"]:
#                 model = model.fit(X_train, Y_train)
#             else:
#                 print("Grid Search CV...")
#                 model = model.cross_val_fit(X_train, Y_train, n_splits=4) # , groups=groups[new_train_index]
#                 print(model.get_params())
                
#             end = time.time()
#             obj["train_results"][epsilon]["predictions"][model_name] = model.predict(X_test)
#             obj["train_results"][epsilon]["times"][model_name] = end - start

#             if model_name in default_params:
#                 obj["train_results"][epsilon]["model_params"][model_name] = default_params[model_name]
#             else:
#                 obj["train_results"][epsilon]["model_params"][model_name] = model.get_params()
            
#             obj["train_results"][epsilon]["model_names"].append(model_name)
    
#         with open(args.outputfile, "wb") as f:
#             pickle.dump(obj, f)