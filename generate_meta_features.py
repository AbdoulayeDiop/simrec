import argparse
import json
import pickle
from collections.abc import Iterable
from joblib import Parallel, delayed

import numpy as np
import openml
import pandas as pd
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

import meta_features as mf
from utils import handle_na
import time

parser = argparse.ArgumentParser(description='Compute meta-features')
parser.add_argument("-i", "--infosfile", help="The infos file")
parser.add_argument("-s", "--selected", help="Path to the file containing ids of selected datasets", default=None)
parser.add_argument("-o", "--outfile", help="The output file")
parser.add_argument("-j", "--jobs", help="The number of concurent workers", default=-1)
args = parser.parse_args()

checked_data = None
if args.selected is not None:
    checked_data = pd.read_csv(args.selected, sep=" ", index_col="id")
datasets_info = {}
infos_file = args.infosfile
with open(infos_file, "rb") as f:
    datasets_info = pickle.load(f)
    if checked_data is not None:
        for id_ in list(datasets_info.keys()):
            if int(id_) not in checked_data.index:
                datasets_info.pop(id_)

print("LOADING...")
datasets = {}
n_loaded = 0
for id_ in datasets_info:
    try:
        dataset = openml.datasets.get_dataset(id_)
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute
        )
        for k in datasets_info[id_]:
            Xnum = X.loc[datasets_info[id_][k]["samples"], datasets_info[id_][k]["num_columns"]]
            Xcat = X.loc[datasets_info[id_][k]["samples"], datasets_info[id_][k]["cat_columns"]]
            for col in Xcat.columns:
                Xcat.loc[:, col] = pd.Categorical(Xcat.loc[:, col]).codes
            Xnum = Xnum.loc[:, (Xnum != Xnum.iloc[0]).any()]
            Xnum = Xnum.to_numpy()
            Xnum = minmax_scale(Xnum)
            Xcat = Xcat.to_numpy(dtype=int)
            
            new_y = y.loc[datasets_info[id_][k]["samples"]]
            new_y = pd.Categorical(new_y).codes if new_y.dtype.name == 'category' else new_y
            # print(f"samples({new_X.shape[0]}), num({Xnum.shape[1]}), cat({Xcat.shape[1]})...")
            datasets[f"{id_}_{k}"] = [Xnum, Xcat, new_y]
        n_loaded += 1
    except:
        print(f"Not able to load data set with id {id_}")

print("END LOADING!")
print()
print(f"{n_loaded}/{len(datasets_info)} data sets loaded (-> {len(datasets)} after augmentation)")

def compute_meta_features(id_, data):
    start = time.time()
    Xnum, Xcat, y = data
    X = np.c_[Xnum, Xcat]
    meta_x = []
    for name, meta_feature in list(mf.get_general_meta_features().items()):
        if name in ["num_on_cat", "isolation_forest"]:
            v = meta_feature(Xnum=Xnum, Xcat=Xcat)
        else:
            v = meta_feature(X)
        if isinstance(v, Iterable):
            meta_x += list(v)
        else:
            meta_x.append(v)

    for name, meta_feature in list(mf.get_num_meta_features().items()):
        v = meta_feature(Xnum)
        if isinstance(v, Iterable):
            meta_x += list(v)
        else:
            meta_x.append(v)

    for name, meta_feature in list(mf.get_cat_meta_features().items()):
        v = meta_feature(Xcat)
        if isinstance(v, Iterable):
            meta_x += list(v)
        else:
            meta_x.append(v)
    end = time.time()
    return id_, meta_x, end-start

print()
print("Computing meta-features...")
list_ret = Parallel(n_jobs=args.jobs, verbose=60)(
    delayed(compute_meta_features)(augmented_id, data) for augmented_id, data in datasets.items()
)

list_ids = []
meta_X = []
times = []
for augmented_id, meta_x, t in list_ret:
    list_ids.append(augmented_id)
    meta_X.append(meta_x)
    times.append(t)

meta_X = np.array(meta_X)
attribute_names = []
for name in list(mf.get_general_meta_features().keys()):
    attribute_names += mf.get_attribute_names(name)
for name in list(mf.get_num_meta_features().keys()):
    attribute_names += mf.get_attribute_names(name)
for name in list(mf.get_cat_meta_features().keys()):
    attribute_names += mf.get_attribute_names(name)

# meta_X = minmax_scale(meta_X)
meta_df = pd.DataFrame(columns=attribute_names, data=meta_X, index=list_ids)
meta_df.index.name = "id"
print(meta_df.head())
meta_df.to_csv(args.outfile, index="id")

with open("output/meta_features_times.json", "w", encoding="utf-8") as f:
    json.dump(dict(zip(list_ids, times)), f, indent=4, ensure_ascii=False)
