from meta_features import compute_meta_features
import numpy as np
from utils import get_valid_similarity_pairs, load_meta_model

def recommend(Xnum, Xcat, algorithm="kprototypes", cvi="ari", k=-1):
    meta_model_pipeline = load_meta_model(algorithm, cvi)
    meta_feature_vector = compute_meta_features(Xnum, Xcat)
    pred = meta_model_pipeline.predict([meta_feature_vector])[0]
    indices = np.argsort(-pred)
    res = list(zip(meta_model_pipeline.similarity_pairs[indices], pred[indices]))

    valid_similarity_pairs = get_valid_similarity_pairs(Xnum, Xcat)
    res = [val for val in res if val[0] in valid_similarity_pairs]
    return res[:k]
