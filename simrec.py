import pickle
from meta_features import compute_meta_features
import numpy as np

def load_dataset(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["Xnum"], data["Xcat"]

def load_meta_model(path):
    with open(path, "rb") as f:
        meta_model_pipeline = pickle.load(f)
    return meta_model_pipeline

def recommend(Xnum, Xcat, meta_model_pipeline, k=5):
    meta_feature_vector = compute_meta_features(Xnum, Xcat)
    pred = meta_model_pipeline.predict([meta_feature_vector])[0]
    indices = np.argsort(-pred)
    return list(zip(meta_model_pipeline.similarity_pairs[indices], pred[indices]))[:k]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Compute meta-features')
    parser.add_argument("-d", "--dataset", help="Path to the dataset", required=True)
    parser.add_argument("-m", "--model", help="Path to the meta-model", required=True)
    args = parser.parse_args()

    Xnum, Xcat = load_dataset(args.dataset)
    meta_model_pipeline = load_meta_model(args.model)
    recommendation = recommend(Xnum, Xcat, meta_model_pipeline)
    print(recommendation)
