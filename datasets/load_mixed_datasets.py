import openml
import numpy as np
import pandas as pd
import re
import pickle
import argparse
from ..utils import load_openml_data


def check_name(name):
    return re.match(r"BNG|autoUniv|seattlecrime", name) is None


openml_df = openml.datasets.list_datasets(output_format="dataframe")
df = openml_df[openml_df.NumberOfNumericFeatures >= 1]
df = df[df.NumberOfSymbolicFeatures >= 2]
df = df[df.NumberOfInstances >= 30]
df = df[df.NumberOfMissingValues <= 0.1 *
        df.NumberOfFeatures*df.NumberOfInstances]
# df = df[df.NumberOfMissingValues == 0]
# df = df[df.NumberOfClasses < 100]
df = df[df.NumberOfClasses >= 2]
df = df[df.version == 1]

tasks = openml.tasks.list_tasks(task_type=openml.tasks.TaskType.CLUSTERING)
datasets_used_for_clustering = np.unique([v["did"] for v in tasks.values()])

df = df[df.index.isin(datasets_used_for_clustering)]
df = df[df.name.apply(check_name)]

n_matching_datasets = len(df)
print(f"{n_matching_datasets} matching data sets found!")

for data_id in df.index:
    print(
        f"Loading dataset with id ({data_id}) shape \
({df.loc[data_id, 'NumberOfInstances']}, {df.loc[data_id, 'NumberOfFeatures']})", end="...")
    data = load_openml_data(data_id)
    parser = argparse.ArgumentParser(
        description='Load mixed datasets from OpenML')
    parser.add_argument("-o", "--outputdir",
                        help="Path for the output directory")
    args = parser.parse_args()
    filename = os.path.join(args.outputdir, f"{data_id}.pickle")
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    print("DONE")
