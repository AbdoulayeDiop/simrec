import numpy as np
from utils import contain_na
import pickle

np.random.seed(0)

def augment(X, num_columns, cat_columns, n_newdata=10):
    n, c = len(num_columns), len(cat_columns)
    res = []
    if X.shape[0] <= 2000:
        res.append((X, X.index, num_columns, cat_columns))
    else:
        new_X = X.sample(2000)
        res.append((new_X, new_X.index, num_columns, cat_columns))
    n_samples = np.arange(min(2000, X.shape[0])*4//10, min(2000, X.shape[0]*8//10) + 1)
    n_num_columns = np.arange(max(1, n//3), n + 1)
    n_cat_columns = np.arange(max(1, c//3), c + 1)
    for i in np.random.choice(len(n_samples)*len(n_num_columns)*len(n_cat_columns), size=n_newdata):
        # if n_samples[i] != X.shape[0] or n_num_columns[i] != n or n_cat_columns[i] != c:
        new_X = X.sample(n_samples[i//(len(n_num_columns)*len(n_cat_columns))])

        nums_to_drop = np.random.choice(num_columns, n - n_num_columns[i%(len(n_num_columns)*len(n_cat_columns))//len(n_cat_columns)])
        new_X = new_X.drop(columns=nums_to_drop)
        
        cats_to_drop = np.random.choice(cat_columns, c - n_cat_columns[i%len(n_cat_columns)])
        new_X = new_X.drop(columns=cats_to_drop)

        res.append((new_X, new_X.index, [col for col in num_columns if col not in nums_to_drop], [col for col in cat_columns if col not in cats_to_drop]))
    # print(f"{len(res)} augmented data generated...")
    return res

if __name__=="__main__":
    import openml
    import os
    import argparse
    parser = argparse.ArgumentParser(description='Fiend data sets from OpenML')
    parser.add_argument("-o", "--outdir", help="The directory where results will be saved")
    parser.add_argument("-r", "--replace", help="Update or replace result if exist", action="store_true")
    args = parser.parse_args()

    openml_df = openml.datasets.list_datasets(output_format="dataframe")
    df = openml_df[openml_df.NumberOfNumericFeatures >= 1]

    df = df[df.NumberOfInstances >= 50]
    df = df[df.NumberOfSymbolicFeatures >= 2]
    # df = df[df.NumberOfMissingValues <= 0.2*df.NumberOfFeatures*df.NumberOfInstances]
    df = df[df.NumberOfMissingValues == 0]
    df = df[df.NumberOfClasses > 0]
    df = df[df.NumberOfClasses < 100]
    df = df[df.version == 1]
    n_matching_datasets = len(df)
    print(f"{n_matching_datasets} matching data sets found!")
    print(df.describe())

    datasets_info = {}
    infos_file = os.path.join(args.outdir, "infos_found_datasets.pickle")
    if os.path.isfile(infos_file):
        with open(infos_file, "rb") as f:
            datasets_info = pickle.load(f)

    list_ids = [str(id_) for id_ in df.did.to_list() if str(id_) not in datasets_info]

    n_datasets = len(list_ids)
    print(f"{n_datasets} new data sets to load")

    print()
    print("LOADING AND AUGMENTATION...")
    n_loaded = 0
    n_augmented = 0
    for i, id_ in enumerate(list_ids):
        # print(f"id:{id_}, {i+1}/{len(list_ids)}")
        try:
            dataset = openml.datasets.get_dataset(id_)
            X, y, categorical_indicator, attribute_names = dataset.get_data(
                target=dataset.default_target_attribute
            )
        except:
            X = None
            print(f"Not able to load data set with id {id_}")
            
        if X is not None and not contain_na(X):
            num_columns = X.select_dtypes(include=["number"]).columns
            cat_columns = [col for col in X.columns if col not in num_columns]
            if len(num_columns) > 0 and len(cat_columns) > 0:
                datasets_info[id_] = {}
                k = 0
                for new_X, samples, new_num_columns, new_cat_columns in augment(X, num_columns, cat_columns):
                    datasets_info[id_][k] = {"samples": samples, "num_columns": new_num_columns, "cat_columns": new_cat_columns}
                    k += 1
                n_loaded += 1
                n_augmented += k

    print("END LOADING!")
    print()
    print(f"{n_loaded}/{n_datasets} new data sets loaded (-> {n_augmented} after augmentation)")
    print(f"TOTAL: {len(datasets_info)} data sets (-> {sum([len(v) for v in datasets_info.values()])} after augmentation)")
    with open(infos_file, "wb") as f:
        pickle.dump(datasets_info, f)