import argparse
import numpy as np
import pickle
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from ga_based_optimization import mfs_plus_hpo_knn
from utils import load_meta_dataset, mean_top_r, lower_bound, ndcg
from joblib import Parallel, delayed
import time

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(FILE_DIR)
sys.path.append(PARENT_DIR)

from meta_model import KNN, create_pipeline

proposed_attributes_statistics = [
    f"{p}_{name}" for p in ["min", "q1", "mean", "q3", "max"] for name in [
        "means_squared_num_att", "std_squared_num_att",
        "means_internal_product_num_att", "std_internal_product_num_att",
        "mutual_info_cat_att",
        "std_freq_cat_att"
    ]
]

parser = argparse.ArgumentParser(description='Compute meta-features')
parser.add_argument("-b", "--benchmarkdir", help="Path to the directory containing benchmark the results", required=True)
parser.add_argument("-m", "--metafeaturesfile", help="Path to the meta-features file", required=True)
parser.add_argument("-o", "--outputdir", help="Path to the output directory", default=None)
parser.add_argument(
    "-j", "--jobs", help="The number of concurent workers", default=-1)
args = parser.parse_args()

BENCHMARK_RESULTS_DIR = args.benchmarkdir
META_FEATURES_FILE = args.metafeaturesfile
OUTPUT_DIR = args.outputdir
N_SPLITS = 10
ALPHA = 4

def scorer_func(yt, yp, cvi):
    # return mean_top_r(yt, yp, cvi, r=1)
    yn = np.array([(y-lower_bound(cvi))/max(y-lower_bound(cvi)) for y in yt])
    yn[yn > 0] **= ALPHA
    yn[yn <= 0] = -1
    return np.mean(ndcg(yn, yp, p=10))

def grid_search_cv_predict_knn(X, Y, cvi, n_splits=5, verbose=0, n_jobs=-1):
    parameters = {
        'n_neighbors': [v for v in range(1, 32, 2) if v <= X.shape[0]/2],
        'metric': ["euclidean", "manhattan", "cosine"],
        'weights': ["uniform", "distance"]
    }
    def evaluate(n_neighbors, metric, w):
        knn = KNN(n_neighbors=n_neighbors, metric=metric, weights=w)
        Y_pred = cross_val_predict(knn, X, Y, cv=n_splits)
        fitness = scorer_func(Y, Y_pred, cvi)
        return fitness
    list_params = [(n_neighbors, metric, w) for n_neighbors in parameters["n_neighbors"] for metric in parameters["metric"] for w in parameters["weights"]]
    list_ret = Parallel(n_jobs=n_jobs)(delayed(evaluate)(*t) for t in list_params)
    best_params = dict(zip(["n_neighbors", "metric", "weights"], list_params[np.argmax(list_ret)]))
    knn = KNN(**best_params)
    return cross_val_predict(knn, X, Y, cv=n_splits, n_jobs=-1), best_params

meta_features_df, benchmark_results = load_meta_dataset(META_FEATURES_FILE, BENCHMARK_RESULTS_DIR)

for algorithm in ['kprototypes', 'lshkprototypes', 'fasterpam', 'haverage']: # 'kprototypes', 'fasterpam', 'haverage', 'lshkprototypes'
    for cvi in ['sil', 'ari', 'acc']:
        print(algorithm, cvi,
            "##################################################")
        obj = {}
        filename = os.path.join(OUTPUT_DIR, f"training_results/{algorithm}_{cvi}.pickle")
        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                obj = pickle.load(f)
        
        if "X" not in obj:
            Y = benchmark_results[algorithm][cvi].to_numpy()
            lower_bound_cvi = lower_bound(cvi)
            Yn = np.array([(y-lower_bound_cvi)/max(y-lower_bound_cvi) for y in Y])
            Yn[Yn > 0] **= ALPHA
            Yn[Yn <= 0] = -1

            X_ = meta_features_df.loc[benchmark_results[algorithm][cvi].index].to_numpy()
            sc = StandardScaler().fit(X_)
            X = sc.transform(X_)
            X2 = X[:, [i for i in range(X.shape[1]) if meta_features_df.columns.values[i]
                    not in proposed_attributes_statistics]]
            obj["X_"] = X_
            obj["X"] = X
            obj["Y"] = Y
            obj["X2"] = X2
            obj["Yn"] = Yn
            obj["meta_feature_names"] = meta_features_df.columns.to_numpy()
            obj["similarity_pairs"] = benchmark_results[algorithm][cvi].columns.to_numpy()
        else:
            X_, X, Y, X2, Yn = obj["X_"], obj["X"], obj["Y"], obj["X2"], obj["Yn"]

        print(f"Shapes -> X: {obj['X'].shape}, X2: {obj['X2'].shape}, Y: {obj['Y'].shape}")

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
            print(f"mean {cvi}:",  score)
            print()

        # with open(filename, "wb") as f:
        #     pickle.dump(obj, f)

        ###############################################################
        model_name = "LMF-KNN"
        if model_name not in obj["train_results"]:
            print(model_name)
            obj["train_results"][model_name] = {}
            obj["train_results"][model_name]["pred"], obj["train_results"][model_name]["params"] = \
                grid_search_cv_predict_knn(X2, Y, cvi, n_splits=N_SPLITS)
            score = np.mean([y[y > -1][np.argmax(obj["train_results"]
                            [model_name]["pred"][i][y > -1])] for i, y in enumerate(Y)])
            print("params:", obj["train_results"][model_name]["params"])
            print(f"mean {cvi}:",  score)
            print()

        # with open(filename, "wb") as f:
        #     pickle.dump(obj, f)

        ###############################################################
        model_name = "AMF-KNN"
        if model_name not in obj["train_results"]:
            print(model_name)
            obj["train_results"][model_name] = {}
            obj["train_results"][model_name]["pred"], obj["train_results"][model_name]["params"] = grid_search_cv_predict_knn(
                X, Y, cvi, n_splits=N_SPLITS)
            score = np.mean([y[y > -1][np.argmax(obj["train_results"]
                            [model_name]["pred"][i][y > -1])] for i, y in enumerate(Y)])
            print("params:", obj["train_results"][model_name]["params"])
            print(f"mean {cvi}:",  score)
            print()

        # with open(filename, "wb") as f:
        #     pickle.dump(obj, f)

        ###############################################################
        model_name = "LMF-FS-KNN"
        if model_name not in obj["train_results"]:
            print(model_name)
            obj["train_results"][model_name] = {}
            lmf_fs_knn, selected_feats, n_neighbors, metric, w, ga_instance = mfs_plus_hpo_knn(
                X2, Y, lambda yt, yp: scorer_func(yt, yp, cvi), num_generations=600, pop_size=16)
            obj["train_results"][model_name]["pred"] = cross_val_predict(
                lmf_fs_knn, X2[:, selected_feats], Y, cv=N_SPLITS, n_jobs=-1)
            obj["train_results"][model_name]["params"] = lmf_fs_knn.get_params()
            obj["train_results"][model_name]["selected_features"] = selected_feats
            score = np.mean([y[y > -1][np.argmax(obj["train_results"]
                            [model_name]["pred"][i][y > -1])] for i, y in enumerate(Y)])
            print("params:", obj["train_results"][model_name]["params"])
            print(f"mean {cvi}:",  score)
            print()

        # with open(filename, "wb") as f:
        #     pickle.dump(obj, f)

        ###############################################################
        model_name = "AMF-FS-KNN"
        
        t0 = time.time()
        if model_name not in obj["train_results"]:
            print(model_name)
            obj["train_results"][model_name] = {}
            amf_fs_knn, selected_feats, n_neighbors, metric, w, ga_instance = mfs_plus_hpo_knn(
                X, Y, lambda yt, yp: scorer_func(yt, yp, cvi), num_generations=600, pop_size=16)
            obj["train_results"][model_name]["pred"] = cross_val_predict(
                amf_fs_knn, X[:, selected_feats], Y, cv=N_SPLITS, n_jobs=-1)
            obj["train_results"][model_name]["params"] = amf_fs_knn.get_params()
            obj["train_results"][model_name]["selected_features"] = selected_feats
            obj["train_results"][model_name]["selected_features_names"] = obj["meta_feature_names"][selected_feats]
            score = np.mean([y[y > -1][np.argmax(obj["train_results"]
                            [model_name]["pred"][i][y > -1])] for i, y in enumerate(Y)])
            print("params:", obj["train_results"][model_name]["params"])
            print(f"mean {cvi}:",  score)

        # with open(filename, "wb") as f:
        #     pickle.dump(obj, f)

        print("Final training and saving...")
        pipeline = create_pipeline(
            scaler=StandardScaler(),
            selected_features=obj["train_results"]["AMF-FS-KNN"]["selected_features"],
            meta_model=KNN(**obj["train_results"]["AMF-FS-KNN"]["params"])
        )
        pipeline = pipeline.fit(X_, Y)
        tf = time.time()
        pipeline.similarity_pairs = obj["similarity_pairs"]
        model_filename = os.path.join(OUTPUT_DIR, f"saved_models/meta_model_pipeline_{algorithm}_{cvi}.pickle")
        with open(model_filename, "wb") as f:
            pickle.dump(pipeline, f)
        print("DONE")
        print("training time", tf - t0)
