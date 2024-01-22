""" 
Load mixed datasets from OpenML
"""

import os
import re
import argparse
import pickle
import openml
import numpy as np
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
        "Asteroid_Dataset", "RelevantImagesDatasetTEST"]
    return valid


openml_df = openml.datasets.list_datasets(output_format="dataframe")
df = openml_df[openml_df.NumberOfNumericFeatures >= 1]
df = df[df.NumberOfSymbolicFeatures >= 2]
df = df[df.NumberOfInstances >= 30]
df = df[df.NumberOfMissingValues <= 0.1 *
        df.NumberOfFeatures*df.NumberOfInstances]
# df = df[df.NumberOfMissingValues == 0]
# df = df[df.NumberOfClasses < 100]
df = df[df.NumberOfClasses >= 2]
# df = df[df.version == 1]
for indices in df.groupby(["did"]).indices:
    print(min(df.iloc[indices].version.values))

# tasks = openml.tasks.list_tasks(task_type=openml.tasks.TaskType.CLUSTERING)
# datasets_used_for_clustering = np.unique([v["did"] for v in tasks.values()])

# df = df[df.index.isin(datasets_used_for_clustering)]
# df = df[df.name.apply(check_name)]

# n_matching_datasets = len(df)
# print(f"{n_matching_datasets} matching data sets found!")
# for id_, name in zip(df.index.values, df.name.values):
#     print(id_, name)

# already_loaded_datasets = [int(filename.split('.')[0])
#                            for filename in os.listdir(args.outputdir)]

# for data_id in df.index:
#     if data_id not in already_loaded_datasets:
#         print()
#         print(
#             f"Loading dataset with id ({data_id}) shape \
# ({df.loc[data_id, 'NumberOfInstances']}, {df.loc[data_id, 'NumberOfFeatures']})...")
#         data = load_openml_data(data_id)
#         if data is not None:
#             n_samples = len(data["samples"])
#             if n_samples > 1e4:
#                 sub_samples = np.random.choice(n_samples, size=10000, replace=False)
#                 data["Xnum"] = data["Xnum"][sub_samples]
#                 data["Xcat"] = data["Xcat"][sub_samples]
#                 data["y"] = data["y"][sub_samples]
#                 data["samples"] = data["samples"][sub_samples]
#             filename = os.path.join(args.outputdir, f"{data_id}.pickle")
#             with open(filename, "wb") as f:
#                 pickle.dump(data, f)
#         print("DONE")
