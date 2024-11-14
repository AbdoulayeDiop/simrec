import os

ALGORITHMS = ["kprototypes", "lshkprototypes", "fasterpam", "haverage"]
CVIS = ["ari", "acc", "sil"]

file_dir = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(file_dir, "meta_model_training/data/saved_models/")
