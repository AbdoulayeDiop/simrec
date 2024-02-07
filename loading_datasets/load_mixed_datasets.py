""" 
Load mixed datasets from OpenML
"""

import os
import re
import argparse
import pickle
import openml
import numpy as np
import pandas as pd
from utils import load_openml_data  # pylint: disable=wrong-import-position

parser = argparse.ArgumentParser(
    description='Load mixed datasets from OpenML')
parser.add_argument("-o", "--outputdir",
                    help="Path for the output directory")
args = parser.parse_args()


def check_name(name):
    valid = re.match(r"BNG|autoUniv|seattlecrime|analcatdata|yeast", name) is None
    valid = valid and name not in \
        ["usp05", "KDDCup99", "musk" "haberman",
        "energy-efficiency", "albert", "rl",
        "AsterodidDataset", "RelevantImagesDatasetTEST"]
    return valid


openml_df = openml.datasets.list_datasets(output_format="dataframe")
df = openml_df[openml_df.NumberOfNumericFeatures >= 1]
df = df[df.NumberOfSymbolicFeatures >= 2]
df = df[df.NumberOfInstances >= 30]
df = df[df.NumberOfMissingValues <= 0.1 *
        df.NumberOfFeatures*df.NumberOfInstances]
df = df[df.NumberOfClasses >= 2]

dataset_list = []

for name, indices in df.groupby(["name"]).indices.items():
    if check_name(name):
        i = np.argmin(df.iloc[indices].version.values)
        did, version = df.iloc[indices[i]][["did", "version"]]
        dataset_list.append({
            "did": did,
            "name": name,
            "version": version
        })

# print(len(dataset_list))

clustering_tasks = openml.tasks.list_tasks(task_type=openml.tasks.TaskType.CLUSTERING, output_format='dataframe')
classification_tasks = openml.tasks.list_tasks(task_type=openml.tasks.TaskType.SUPERVISED_CLASSIFICATION, output_format='dataframe')
dataset_list = [d for d in dataset_list if d["name"] in clustering_tasks.name.tolist() + classification_tasks.name.tolist()]
# print(len(dataset_list))

n_matching_datasets = len(dataset_list)
print(f"{n_matching_datasets} matching data sets found!")
for d in dataset_list:
    print(d)

already_loaded_datasets = [int(filename.split('.')[0])
                           for filename in os.listdir(args.outputdir)]

for d in dataset_list:
    did = d["did"]
    if did not in already_loaded_datasets:
        print()
        print(
            f"Loading dataset with id ({did})...", end="")
        data = load_openml_data(did)
        if data is not None:
            n_samples = len(data["samples"])
            if n_samples > 1e4:
                print(
                    f"shape ({data['Xnum'].shape[0]}, {data['Xnum'].shape[1]+data['Xcat'].shape[1]})",
                    end="..."
                )
                sub_samples = np.random.choice(n_samples, size=10000, replace=False)
                data["Xnum"] = data["Xnum"][sub_samples]
                data["Xcat"] = data["Xcat"][sub_samples]
                data["y"] = data["y"][sub_samples]
                data["samples"] = data["samples"][sub_samples]
            filename = os.path.join(args.outputdir, f"{did}.pickle")
            with open(filename, "wb") as f:
                pickle.dump(data, f)
        print("DONE")
