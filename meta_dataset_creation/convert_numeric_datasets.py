import numpy as np

def discretize(x, n):
    """Discretize a list/array x by dividing its values into n intervals. 
    Each value in x is then associated with a category corresponding to one of the intervals.

    Parameters
    ----------
    x : list / 1D array
        The values to discretize
    n : int
        The number of categories

    Returns
    -------
    numpy array
        The discretized values
    """
    eps=1e-9
    x = np.array(x)
    min_, max_ = min(x), max(x)
    x = (x - min_)/(max_ - min_)
    bins = np.linspace(0, 1 + eps, n+1)
    x_discrete = np.digitize(x, bins) - 1
    permutation = np.random.permutation(n)
    if max(x_discrete) >= len(permutation):
        print(min_, max_)
        print(n, len(bins))
        print(np.digitize(x, bins))
        print(permutation)
    return permutation[x_discrete]

def transform(X, min_, max_, q2):
    n_features = np.random.randint(
        min_["n_features"], min(X.shape[1], max_["n_features"]) + 1)
    n_num_features = np.random.randint(
        min_["n_num_att"], min(n_features-1, max_["n_num_att"]) + 1)
    selected_features = np.random.choice(
        range(X.shape[1]), size=n_features, replace=False)
    num_indices = selected_features[:n_num_features]
    cat_indices = selected_features[n_num_features:]
    Xnum, Xcat = X[:, num_indices], X[:, cat_indices]
    for j in range(Xcat.shape[1]):
        n_cat = np.random.randint(
            2, int(np.random.lognormal(q2["max_card_cat_att"]/3, 1.2)) + 3)
        Xcat[:, j] = discretize(Xcat[:, j], n_cat)
    return Xnum, Xcat, num_indices, cat_indices

if __name__=="__main__":
    import argparse
    import os
    import json
    import pickle
    import pandas as pd
    import sys
    from sklearn.metrics import pairwise_distances
    from sklearn.preprocessing import minmax_scale, StandardScaler
    sys.path.append(os.path.dirname(__file__))
    from meta_features import compute_meta_features, ALL_ATTRIBUTE_NAMES
    
    parser = argparse.ArgumentParser(description='Compute meta-features')
    parser.add_argument("-n", "--numericdatasetsdir", default=None)
    parser.add_argument("-m", "--mixedmetafeaturesfile", default=None)
    parser.add_argument("-c", "--generateddatasetsdir",
                        help="Path to the output directory for converted data")
    parser.add_argument("-o", "--outputdir",
                        help="Path to the output directory for computed meta-features")
    parser.add_argument(
        "-j", "--jobs", help="The number of concurent workers", default=-1)
    args = parser.parse_args()

    mixed_meta_df = pd.read_csv(args.mixedmetafeaturesfile, index_col="id")
    min_ = mixed_meta_df.min(axis=0).to_dict()
    max_ = mixed_meta_df.max(axis=0).to_dict()
    q2 = mixed_meta_df.median(axis=0).to_dict()
    mixed_meta_X = mixed_meta_df.to_numpy()
    sc = StandardScaler().fit(mixed_meta_X)
    mixed_meta_X = sc.transform(mixed_meta_X)

    OUTPUT_FILE = os.path.join(args.outputdir, "meta_features.csv")
    TIME_FILE = os.path.join(args.outputdir, "meta_features_times.json")
    numeric_meta_df = pd.DataFrame(columns=ALL_ATTRIBUTE_NAMES)
    if os.path.isfile(OUTPUT_FILE):
        numeric_meta_df = pd.read_csv(OUTPUT_FILE, index_col="id")
    times = {}
    if os.path.isfile(TIME_FILE):
        with open(TIME_FILE, "r", encoding="utf-8") as f:
            times = json.load(f)

    filenames = []
    if args.numericdatasetsdir is not None:
        filenames += [os.path.join(args.numericdatasetsdir, filename)
                        for filename in os.listdir(args.numericdatasetsdir)]

    k = 5
    meta_X = mixed_meta_X if len(numeric_meta_df)==0 else \
        np.concatenate((mixed_meta_X, sc.transform(numeric_meta_df.to_numpy())))
    D = pairwise_distances(meta_X)
    max_d = np.max(D)
    np.fill_diagonal(D, np.inf)
    d_neighbors = np.sort(D, axis=1)[:, :k]
    threshold = min(np.sum(d_neighbors, axis=1)/k)
    
    n_datasets_to_generate = 500
    n_generated_datasets = len(os.listdir(args.generateddatasetsdir))
    i = n_generated_datasets
    n_trials = 0
    print("START")
    while i < n_generated_datasets + n_datasets_to_generate:
        filename = np.random.choice(filenames)
        with open(filename, "rb") as f:
            data = pickle.load(f)
        print(f"trial {n_trials + 1}, dataset id: {data['id']}.", end=" ")
        X = data["X"]
        if X.shape[1] >= 2:
            Xnum, Xcat, num_indices, cat_indices = transform(X, min_, max_, q2)
            meta_x, t = compute_meta_features(minmax_scale(Xnum), Xcat, return_time=True)
            d = pairwise_distances(
                sc.transform([meta_x]),
                meta_X
            )[0]
            # print(min(d), threshold, end=".")
            if max_d > min(d) > threshold:
                id_ = f"{i}_{data['id']}"
                numeric_meta_df = pd.concat([
                    numeric_meta_df,
                    pd.DataFrame(columns=ALL_ATTRIBUTE_NAMES, data=[meta_x], index=[id_])
                ])
                numeric_meta_df.index.name = "id"
                times[id_] = t
                numeric_meta_df.to_csv(OUTPUT_FILE, index="id")
                with open(TIME_FILE, "w", encoding="utf-8") as f:
                    json.dump(times, f, indent=4, ensure_ascii=False)

                new_data = {
                    "id": data["id"],
                    "data_type": "mixed",
                    "numeric_attributes": data["attributes_names"][num_indices],
                    "categorical_attributes": data["attributes_names"][cat_indices],
                    "samples": data["samples"],
                    "Xnum": Xnum,
                    "Xcat": Xcat,
                    "y": data["y"],
                }
                filename = os.path.join(args.generateddatasetsdir, f"{i}_{data['id']}.pickle")
                with open(filename, "wb") as f:
                    pickle.dump(new_data, f)

                meta_X = mixed_meta_X if len(numeric_meta_df)==0 else \
                    np.concatenate((mixed_meta_X, sc.transform(numeric_meta_df.to_numpy())))
                D = pairwise_distances(meta_X)
                np.fill_diagonal(D, np.inf)
                d_neighbors = np.sort(D, axis=1)[:, :k]
                threshold = min(np.sum(d_neighbors, axis=1)/k)
                i += 1
        n_trials += 1
        print(i - n_generated_datasets, "datasets generated")
    print("END")