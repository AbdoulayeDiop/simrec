"""
Compute meta-features
"""

import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import minmax_scale
from meta_features import compute_meta_features, ALL_ATTRIBUTE_NAMES

parser = argparse.ArgumentParser(description='Compute meta-features')
parser.add_argument("-d", "--datasetsdir", default=None)
parser.add_argument("-o", "--outputdir",
                    help="Path to the output directory")
parser.add_argument(
    "-j", "--jobs", help="The number of concurent workers", default=-1)
args = parser.parse_args()

OUTPUT_FILE = os.path.join(args.outputdir, "meta_features.csv")
TIME_FILE = os.path.join(args.outputdir, "meta_features_times.json")

meta_df = pd.DataFrame(columns=ALL_ATTRIBUTE_NAMES)
if os.path.isfile(OUTPUT_FILE):
    meta_df = pd.read_csv(OUTPUT_FILE, index_col="id")
times = {}
if os.path.isfile(TIME_FILE):
    with open(TIME_FILE, "r", encoding="utf-8") as f:
        times = json.load(f)

filenames = []
if args.datasetsdir is not None:
    filenames += [os.path.join(args.datasetsdir, filename)
                    for filename in os.listdir(args.datasetsdir)
                    if filename.split('.')[0] not in meta_df.index]

datasets = []
for filename in filenames:
    with open(filename, "rb") as f:
        data = pickle.load(f)
    datasets.append(data)

print("Computing meta-features...")
start = time.time()
list_ret = Parallel(n_jobs=int(args.jobs), verbose=60)(
    delayed(compute_meta_features)(
        minmax_scale(data["Xnum"]), data["Xcat"], return_time=True
    ) for data in datasets
)
end = time.time()
print(f"END. total time = {end - start}")

print("SAVING...")
ids = [data["id"] for data in datasets]
meta_X = []
i = 0
for meta_x, t in list_ret:
    meta_X.append(meta_x)
    times[ids[i]] = t
    i += 1

meta_X = np.array(meta_X)

meta_df = pd.concat([
    meta_df,
    pd.DataFrame(columns=ALL_ATTRIBUTE_NAMES, data=meta_X, index=ids)
])
meta_df.index.name = "id"
print(meta_df.head())
meta_df.to_csv(OUTPUT_FILE, index="id")
with open(TIME_FILE, "w", encoding="utf-8") as f:
    json.dump(times, f, indent=4, ensure_ascii=False)
print("DONE")
