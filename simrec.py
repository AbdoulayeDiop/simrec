import pickle
from meta_features import compute_meta_features
import numpy as np
from utils import get_valid_similarity_pairs
from sklearn.preprocessing import minmax_scale
import os

ALGORITHMS = ["kprototypes", "fasterpam", "haverage"]
CVIS = ["ari", "acc", "sil"]

def load_meta_model(models_dir, algorithm, cvi):
    path = os.path.join(models_dir, f"meta_model_pipeline_{algorithm}_{cvi}.pickle")
    with open(path, "rb") as f:
        meta_model_pipeline = pickle.load(f)
    return meta_model_pipeline

def recommend(Xnum, Xcat, models_dir, algorithm="kprototypes", cvi="ari", k=5):
    assert algorithm in ALGORITHMS, f"algorithm should be from \
        {ALGORITHMS}, got {algorithm} instead"
    assert cvi in CVIS, f"cvi should be from \
        {CVIS}, got {cvi} instead"

    meta_model_pipeline = load_meta_model(models_dir, algorithm, cvi)
    meta_feature_vector = compute_meta_features(Xnum, Xcat)
    pred = meta_model_pipeline.predict([meta_feature_vector])[0]
    indices = np.argsort(-pred)
    res = list(zip(meta_model_pipeline.similarity_pairs[indices], pred[indices]))

    valid_similarity_pairs = get_valid_similarity_pairs(Xnum, Xcat)
    res = [val for val in res if val[0] in valid_similarity_pairs]
    return res[:k]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compute meta-features')
    parser.add_argument("-d", "--dataset", help="Path to the dataset", required=True)
    parser.add_argument("-m", "--modelsdir", help="Path to the models directory", required=True)
    parser.add_argument("-a", "--algorithm", help="The algorithm", required=True)
    parser.add_argument("-c", "--cvi", help="The cluster validity index to be optimized", default="ari")
    args = parser.parse_args()

    def load_dataset(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data["Xnum"], data["Xcat"]

    Xnum, Xcat = load_dataset(args.dataset)
    Xnum = minmax_scale(Xnum)
    recommendation = recommend(Xnum, Xcat, args.modelsdir, algorithm=args.algorithm, cvi=args.cvi)
    print(recommendation)
