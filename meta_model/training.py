import numpy as np
import pandas as pd
import openml
import pickle
import os
import sys
from utils import ndcg_sim, ndcg, custom_sim
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale, StandardScaler
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from ranking import ALL_MODELS, scorer, scorer_func
from ga import mfs_plus_hpo_knn, mfs_plus_hpo_dtree, create_encoder_decoder, aeknn_hpo
import torch
from meta_model import AEKNN
from utils import load_meta_dataset
from joblib import Parallel, delayed

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"

np.random.seed(1234)

dataset_statistics = ["n_instances", "n_features",
                      "dim", "num_on_cat", "n_num_att", "n_cat_att"]
attributes_statistics = [
    f"{p}_{name}" for p in ["min", "q1", "mean", "q3", "max"] for name in [
        "means_num_att", "std_num_att",
        "card_cat_att", "entropy_cat_att",
        "covariance",
    ]
]
proposed_attributes_statistics = [
    f"{p}_{name}" for p in ["min", "q1", "mean", "q3", "max"] for name in [
        "means_squared_num_att", "std_squared_num_att",
        "means_internal_product_num_att", "std_internal_product_num_att",
        "mutual_info_cat_att",
        "std_freq_cat_att"
    ]
]


# def grid_search_cv_predict_knn(X, Y, Yn, n_splits=5, scorer=scorer, verbose=0, n_jobs=-1):
#     parameters = {
#         'n_neighbors': [v for v in range(1, 32, 2) if v <= X.shape[0]/2],
#         'metric': ["euclidean", "manhattan", "cosine"],
#         'weights': ["uniform", "distance"]
#     }
#     knn = ALL_MODELS["KNN"]().cross_val_fit(X, Y, n_splits=n_splits)
#     knn = ALL_MODELS["KNN"](**knn.get_params())
#     return cross_val_predict(knn, X, Y, cv=n_splits, n_jobs=-1), knn.get_params()

def grid_search_cv_predict_knn(X, Y, Yn, n_splits=5, scorer=scorer, verbose=0, n_jobs=-1):
    parameters = {
        'n_neighbors': [v for v in range(1, 32, 2) if v <= X.shape[0]/2],
        'metric': ["euclidean", "manhattan", "cosine"],
        'weights': ["uniform", "distance"]
    }
    def evaluate(n_neighbors, metric, w):
        knn = ALL_MODELS["KNN"](n_neighbors=n_neighbors, metric=metric, weights=w)
        Y_pred = cross_val_predict(knn, X, Y, cv=n_splits)
        fitness = scorer_func(Yn, Y_pred)
        return fitness
    list_params = [(n_neighbors, metric, w) for n_neighbors in parameters["n_neighbors"] for metric in parameters["metric"] for w in parameters["weights"]]
    list_ret = Parallel(n_jobs=n_jobs)(delayed(evaluate)(*t) for t in list_params)
    best_params = dict(zip(["n_neighbors", "metric", "weights"], list_params[np.argmax(list_ret)]))
    knn = ALL_MODELS["KNN"](**best_params)
    return cross_val_predict(knn, X, Y, cv=n_splits, n_jobs=-1), best_params

def grid_search_cv_predict_dtree(X, Y, n_splits=5, scorer=scorer, verbose=0, n_jobs=-1):
    parameters = {
        'min_samples_leaf': [v for v in range(1, 32, 2) if v <= X.shape[0]/2],
        'max_features': [None, "sqrt", "log2"],
    }
    dtree = DecisionTreeRegressor(random_state=0)
    gridcv = GridSearchCV(dtree, parameters,
                          scoring=scorer,
                          cv=n_splits,
                          verbose=verbose,
                          error_score='raise',
                          refit=False,
                          n_jobs=n_jobs
                          ).fit(X, Y)
    dtree = DecisionTreeRegressor(**gridcv.best_params_, random_state=0)
    return cross_val_predict(dtree, X, Y, cv=n_splits, n_jobs=-1), gridcv.best_params_

# def grid_search_cv_predict_rf(X, Y, n_splits=5, scorer=scorer, verbose=0, n_jobs=-1):
#     parameters = {
#         'min_samples_leaf': [v for v in range(1, 32, 2) if v <= X.shape[0]/2],
#         'max_features': [None, "sqrt", "log2"],
#     }
#     dtree = DecisionTreeRegressor()
#     gridcv = GridSearchCV(dtree, parameters,
#                           scoring=scorer,
#                           cv=n_splits,
#                           verbose=verbose,
#                           error_score='raise',
#                           refit=False,
#                           n_jobs=n_jobs
#                           ).fit(X, Y)
#     dtree = DecisionTreeRegressor(**gridcv.best_params_)
#     return cross_val_predict(dtree, X, Y, cv=n_splits, n_jobs=-1), gridcv.best_params_


OUTPUT_DIR = "data/training_wo_filtering"
BENCHMARK_RESULTS_DIR = "../meta_dataset_creation/data/benchmark_results_prev/"
META_FEATURES_FILE = "../meta_dataset_creation/data/meta_features/original/meta_features.csv"
N_SPLITS = 10

ae_fit_params = {
    "lr": 1e-3,
    "weight_decay": 0,
    "epochs": 400,
    "batch_size_train": 16,
    "batch_size_val": 16,
}
nn_structs = [
    [32],
    [16],
    [32, 16],
    [32, 16, 8],
]
n_neighbors_values = [1, 5, 10, 15, 20, 30]
metrics = ["euclidean", "manhattan", "cosine"]
weights = ["uniform", "distance"]

for algorithm in ['kprototypes', 'fasterpam', 'haverage']: #, 'fasterpam', 'haverage'
    for eval_metric in [ "sil", "ari", "acc"]:
        print(algorithm, eval_metric,
            "##################################################")
        obj = {}
        filename = os.path.join(OUTPUT_DIR, f"{algorithm}_{eval_metric}.pickle")
        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                obj = pickle.load(f)
        if "meta_dataset" not in obj:
            scores_dir = os.path.join(
                BENCHMARK_RESULTS_DIR, algorithm, "original/scores")
            obj["meta_dataset"] = load_meta_dataset(
                META_FEATURES_FILE, scores_dir, algorithm, eval_metric)
            with open(filename, "wb") as f:
                pickle.dump(obj, f)

        mixed_meta_df, benchmark_results = obj["meta_dataset"]
        Y = benchmark_results.to_numpy()
        if eval_metric in ["acc", "purity"]:
            Yn = np.array([y/max(y) for y in Y])
            Yn[Yn > 0] **= 4
            Yn[Yn <= 0] = -1
        elif eval_metric == "ari":
            Yn = np.array([(y+0.5)/max(y+0.5) for y in Y])
            Yn[Yn > 0] **= 4
            Yn[Yn <= 0] = -1
        else:
            Yn = np.array([(y+1)/max(y+1) for y in Y])
            Yn[Yn > 0] **= 4
            Yn[Yn <= 0] = -1

        X = mixed_meta_df.to_numpy()
        sc = StandardScaler().fit(X)
        X = sc.transform(X)
        X2 = X[:, [i for i in range(X.shape[1]) if mixed_meta_df.columns.values[i]
                in dataset_statistics+attributes_statistics]]

        print(f"X: {X.shape}, X2: {X2.shape}, Y: {Y.shape}")

        if "train_results" not in obj:
            obj["train_results"] = {}
        ###############################################################
        model_name = "AR"
        if model_name not in obj["train_results"]:
            print(model_name)
            obj["train_results"][model_name] = {}
            obj["train_results"][model_name]["pred"] = np.zeros(shape=Y.shape)
            # obj["train_results"][model_name]["pred"] = np.array([[np.mean(yj[yj>-1]) for yj in Y.T] for _ in Y])
            for train, test in KFold(n_splits=N_SPLITS).split(X):
                obj["train_results"][model_name]["pred"][test] = np.array(
                    [[np.mean(yj) for yj in Y[train].T] for _ in test])
            score = np.mean([y[y > -1][np.argmax(obj["train_results"]
                            [model_name]["pred"][i][y > -1])] for i, y in enumerate(Y)])
            print(f"mean {eval_metric}:",  score)
            print()

        with open(filename, "wb") as f:
            pickle.dump(obj, f)

        ###############################################################
        model_name = "LMF-KNN"
        if model_name not in obj["train_results"]:
            print(model_name)
            obj["train_results"][model_name] = {}
            obj["train_results"][model_name]["pred"], obj["train_results"][model_name]["params"] = \
                grid_search_cv_predict_knn(X2, Y, Yn, n_splits=N_SPLITS)
            score = np.mean([y[y > -1][np.argmax(obj["train_results"]
                            [model_name]["pred"][i][y > -1])] for i, y in enumerate(Y)])
            print("params:", obj["train_results"][model_name]["params"])
            print(f"mean {eval_metric}:",  score)
            print()

        with open(filename, "wb") as f:
            pickle.dump(obj, f)

        ###############################################################
        model_name = "AMF-KNN"
        if model_name not in obj["train_results"]:
            print(model_name)
            obj["train_results"][model_name] = {}
            obj["train_results"][model_name]["pred"], obj["train_results"][model_name]["params"] = grid_search_cv_predict_knn(
                X, Y, Yn, n_splits=N_SPLITS)
            score = np.mean([y[y > -1][np.argmax(obj["train_results"]
                            [model_name]["pred"][i][y > -1])] for i, y in enumerate(Y)])
            print("params:", obj["train_results"][model_name]["params"])
            print(f"mean {eval_metric}:",  score)
            print()

        with open(filename, "wb") as f:
            pickle.dump(obj, f)

        ###############################################################
        model_name = "LMF-FS-KNN"
        if model_name not in obj["train_results"]:
            print(model_name)
            obj["train_results"][model_name] = {}
            lmf_fs_knn, selected_feats, n_neighbors, metric, w, ga_instance = mfs_plus_hpo_knn(
                X2, Y, Yn, num_generations=600, pop_size=8)
            obj["train_results"][model_name]["pred"] = cross_val_predict(
                lmf_fs_knn, X2[:, selected_feats], Y, cv=N_SPLITS, n_jobs=-1)
            obj["train_results"][model_name]["params"] = lmf_fs_knn.get_params()
            obj["train_results"][model_name]["selected_features"] = selected_feats
            score = np.mean([y[y > -1][np.argmax(obj["train_results"]
                            [model_name]["pred"][i][y > -1])] for i, y in enumerate(Y)])
            print("params:", obj["train_results"][model_name]["params"])
            print(f"mean {eval_metric}:",  score)
            print()

        with open(filename, "wb") as f:
            pickle.dump(obj, f)

        ###############################################################
        model_name = "AMF-FS-KNN"
        if model_name not in obj["train_results"]:
            print(model_name)
            obj["train_results"][model_name] = {}
            amf_fs_knn, selected_feats, n_neighbors, metric, w, ga_instance = mfs_plus_hpo_knn(
                X, Y, Yn, num_generations=600, pop_size=8)
            obj["train_results"][model_name]["pred"] = cross_val_predict(
                amf_fs_knn, X[:, selected_feats], Y, cv=N_SPLITS, n_jobs=-1)
            obj["train_results"][model_name]["params"] = amf_fs_knn.get_params()
            obj["train_results"][model_name]["selected_features"] = selected_feats
            score = np.mean([y[y > -1][np.argmax(obj["train_results"]
                            [model_name]["pred"][i][y > -1])] for i, y in enumerate(Y)])
            print("params:", obj["train_results"][model_name]["params"])
            print(f"mean {eval_metric}:",  score)
            print()

        with open(filename, "wb") as f:
            pickle.dump(obj, f)

        ###############################################################
        # model_name = "LMF-AE-KNN"
        # if model_name not in obj["train_results"]:
        #     print(model_name)
        #     obj["train_results"][model_name] = {}
        #     obj["train_results"][model_name]["pred"] = np.zeros(Y.shape)
        #     best_fitness = 0
        #     for n_neurons in nn_structs:
        #         for n_neighbors in n_neighbors_values:
        #             for metric in metrics:
        #                 for w in weights:
        #                     print(
        #                         f"n_neurons: {n_neurons}, n_neighbors: {n_neighbors}, metric: {metric}, w: {w}", end=", ")
        #                     # fitness = 0
        #                     Y_pred = np.zeros(shape=Y.shape)
        #                     for train, test in KFold(n_splits=N_SPLITS).split(X2):
        #                         encoder, decoder = create_encoder_decoder(
        #                             X2.shape[1], n_neurons)
        #                         aeknn = AEKNN(encoder, decoder, n_neighbors,
        #                                     metric, w, device=device)
        #                         aeknn.fit(X2[train], Y[train], ae_fit_params)
        #                         Y_pred[test] = aeknn.predict(X2[test])
        #                         # fitness += scorer_func(Y[test], Y_pred[test])
        #                     # fitness /= N_SPLITS
        #                     fitness = scorer_func(Y, Y_pred)
        #                     print(f"fitness: {fitness}")
        #                     if fitness > best_fitness:
        #                         best_fitness = fitness
        #                         obj["train_results"][model_name]["pred"] = Y_pred
        #                         obj["train_results"][model_name]["params"] = {
        #                             "n_neurons": n_neurons,
        #                             "n_neighbors": n_neighbors,
        #                             "metric": metric,
        #                             "w": w,
        #                         }
        #     score = np.mean([y[y > -1][np.argmax(obj["train_results"]
        #                     [model_name]["pred"][i][y > -1])] for i, y in enumerate(Y)])
        #     print("params:", obj["train_results"][model_name]["params"])
        #     print(f"mean {eval_metric}:",  score)
        #     print()

        # with open(filename, "wb") as f:
        #     pickle.dump(obj, f)

        ###############################################################
        # model_name = "AMF-AE-KNN"
        # if model_name not in obj["train_results"]:
        #     print(model_name)
        #     obj["train_results"][model_name] = {}
        #     obj["train_results"][model_name]["pred"] = np.zeros(Y.shape)
        #     best_fitness = 0
        #     for n_neurons in nn_structs:
        #         for n_neighbors in n_neighbors_values:
        #             for metric in metrics:
        #                 for w in weights:
        #                     print(
        #                         f"n_neurons: {n_neurons}, n_neighbors: {n_neighbors}, metric: {metric}, w: {w}", end=", ")
        #                     # fitness = 0
        #                     Y_pred = np.zeros(shape=Y.shape)
        #                     for train, test in KFold(n_splits=N_SPLITS).split(X):
        #                         encoder, decoder = create_encoder_decoder(
        #                             X.shape[1], n_neurons)
        #                         aeknn = AEKNN(encoder, decoder, n_neighbors,
        #                                     metric, w, device=device)
        #                         aeknn.fit(X[train], Y[train], ae_fit_params)
        #                         Y_pred[test] = aeknn.predict(X[test])
        #                         # fitness += scorer_func(Y[test], Y_pred[test])
        #                     # fitness /= N_SPLITS
        #                     fitness = scorer_func(Y, Y_pred)
        #                     print(f"fitness: {fitness}")
        #                     if fitness > best_fitness:
        #                         best_fitness = fitness
        #                         obj["train_results"][model_name]["pred"] = Y_pred
        #                         obj["train_results"][model_name]["params"] = {
        #                             "n_neurons": n_neurons,
        #                             "n_neighbors": n_neighbors,
        #                             "metric": metric,
        #                             "w": w,
        #                         }
        #     score = np.mean([y[y > -1][np.argmax(obj["train_results"]
        #                     [model_name]["pred"][i][y > -1])] for i, y in enumerate(Y)])
        #     print("params:", obj["train_results"][model_name]["params"])
        #     print(f"mean {eval_metric}:",  score)
        #     print()

        # with open(filename, "wb") as f:
        #     pickle.dump(obj, f)





        ###############################################################
        # model_name = "LMF-DTree"
        # if model_name not in obj["train_results"]:
        #     print(model_name)
        #     obj["train_results"][model_name] = {}
        #     obj["train_results"][model_name]["pred"], obj["train_results"][model_name]["params"] = \
        #         grid_search_cv_predict_dtree(X2, Y, n_splits=N_SPLITS)
        #     score = np.mean([y[y > -1][np.argmax(obj["train_results"]
        #                     [model_name]["pred"][i][y > -1])] for i, y in enumerate(Y)])
        #     print("params:", obj["train_results"][model_name]["params"])
        #     print(f"mean {eval_metric}:",  score)
        #     print()

        # with open(filename, "wb") as f:
        #     pickle.dump(obj, f)

        # ###############################################################
        # model_name = "AMF-DTree"
        # if model_name not in obj["train_results"]:
        #     print(model_name)
        #     obj["train_results"][model_name] = {}
        #     obj["train_results"][model_name]["pred"], obj["train_results"][model_name]["params"] = grid_search_cv_predict_dtree(
        #         X, Y, n_splits=N_SPLITS)
        #     score = np.mean([y[y > -1][np.argmax(obj["train_results"]
        #                     [model_name]["pred"][i][y > -1])] for i, y in enumerate(Y)])
        #     print("params:", obj["train_results"][model_name]["params"])
        #     print(f"mean {eval_metric}:",  score)
        #     print()

        # with open(filename, "wb") as f:
        #     pickle.dump(obj, f)

        # ###############################################################
        # model_name = "LMF-FS-DTree"
        # if model_name not in obj["train_results"]:
        #     print(model_name)
        #     obj["train_results"][model_name] = {}
        #     lmf_fs_dtree, selected_feats, _, _, _ = mfs_plus_hpo_dtree(
        #         X2, Y, num_generations=200, pop_size=8)
        #     obj["train_results"][model_name]["pred"] = cross_val_predict(
        #         lmf_fs_dtree, X2[:, selected_feats], Y, cv=N_SPLITS, n_jobs=-1)
        #     obj["train_results"][model_name]["params"] = lmf_fs_dtree.get_params()
        #     obj["train_results"][model_name]["selected_features"] = selected_feats
        #     score = np.mean([y[y > -1][np.argmax(obj["train_results"]
        #                     [model_name]["pred"][i][y > -1])] for i, y in enumerate(Y)])
        #     print("params:", obj["train_results"][model_name]["params"])
        #     print(f"mean {eval_metric}:",  score)
        #     print()

        # with open(filename, "wb") as f:
        #     pickle.dump(obj, f)

        # ###############################################################
        # model_name = "AMF-FS-DTree"
        # if model_name not in obj["train_results"]:
        #     print(model_name)
        #     obj["train_results"][model_name] = {}
        #     amf_fs_dtree, selected_feats, _, _, _ = mfs_plus_hpo_dtree(
        #         X, Y, num_generations=200, pop_size=8)
        #     obj["train_results"][model_name]["pred"] = cross_val_predict(
        #         amf_fs_dtree, X[:, selected_feats], Y, cv=N_SPLITS, n_jobs=-1)
        #     obj["train_results"][model_name]["params"] = amf_fs_dtree.get_params()
        #     obj["train_results"][model_name]["selected_features"] = selected_feats
        #     score = np.mean([y[y > -1][np.argmax(obj["train_results"]
        #                     [model_name]["pred"][i][y > -1])] for i, y in enumerate(Y)])
        #     print("params:", obj["train_results"][model_name]["params"])
        #     print(f"mean {eval_metric}:",  score)
        #     print()

        # with open(filename, "wb") as f:
        #     pickle.dump(obj, f)
