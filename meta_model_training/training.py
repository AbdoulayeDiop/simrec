import argparse
import numpy as np
import pickle
import os
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_predict
from ga_based_optimization import mfs_plus_hpo_knn
from utils import load_meta_dataset, top1_func
from joblib import Parallel, delayed
sys.path.append("..")
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


def grid_search_cv_predict_knn(X, Y, Yn, n_splits=5, verbose=0, n_jobs=-1):
    parameters = {
        'n_neighbors': [v for v in range(1, 32, 2) if v <= X.shape[0]/2],
        'metric': ["euclidean", "manhattan", "cosine"],
        'weights': ["uniform", "distance"]
    }
    def evaluate(n_neighbors, metric, w):
        knn = KNN(n_neighbors=n_neighbors, metric=metric, weights=w)
        Y_pred = cross_val_predict(knn, X, Y, cv=n_splits)
        fitness = top1_func(Y, Y_pred)
        return fitness
    list_params = [(n_neighbors, metric, w) for n_neighbors in parameters["n_neighbors"] for metric in parameters["metric"] for w in parameters["weights"]]
    list_ret = Parallel(n_jobs=n_jobs)(delayed(evaluate)(*t) for t in list_params)
    best_params = dict(zip(["n_neighbors", "metric", "weights"], list_params[np.argmax(list_ret)]))
    knn = KNN(**best_params)
    return cross_val_predict(knn, X, Y, cv=n_splits, n_jobs=-1), best_params

for algorithm in ['kprototypes', 'fasterpam', 'haverage']: #, 'fasterpam', 'haverage'
    for eval_metric in ["acc", "sil", "ari"]:
        print(algorithm, eval_metric,
            "##################################################")
        obj = {}
        filename = os.path.join(OUTPUT_DIR, f"training_results/{algorithm}_{eval_metric}.pickle")
        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                obj = pickle.load(f)
        if "meta_dataset" not in obj:
            scores_dir = os.path.join(
                BENCHMARK_RESULTS_DIR, algorithm, "scores")
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

        X_ = mixed_meta_df.to_numpy()
        sc = StandardScaler().fit(X_)
        X = sc.transform(X_)
        X2 = X[:, [i for i in range(X.shape[1]) if mixed_meta_df.columns.values[i]
                not in proposed_attributes_statistics]]

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

        with open(filename, "wb") as f:
            pickle.dump(obj, f)
        
        
        pipeline = create_pipeline(
            scaler=StandardScaler(),
            selected_features=obj["train_results"]["AMF-FS-KNN"]["selected_features"],
            meta_model=KNN(**obj["train_results"]["AMF-FS-KNN"]["params"])
        )
        pipeline = pipeline.fit(X_, Y)
        pipeline.similarity_pairs = benchmark_results.columns.to_numpy()
        model_filename = os.path.join(OUTPUT_DIR, f"saved_models/meta_model_pipeline_{algorithm}_{eval_metric}.pickle")
        with open(model_filename, "wb") as f:
            pickle.dump(pipeline, f)
