import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Node():
    def __init__(self, node_id=None, parent=None, elements=[], bounds=[], feature=None, split_value=None):
        self.id = node_id
        self.feature = feature
        self.split_value = split_value
        self.children_left = -1
        self.children_right = -1
        self.elements = elements
        self.parent = parent
        self.bounds = bounds

class Tree():
    def __init__(self, nodes={}):
        self.nodes = nodes

    def set_root(self, node):
        self.root = node
        self.nodes[node.id] = node

    def add_node(self, node):
        self.nodes[node.id] = node

    def get_node(self, node_id):
        return self.nodes[node_id]

def create_random_tree(X, l):
    n, m = X.shape
    t = Tree()
    root = Node(node_id=0, elements=list(range(n)), bounds=[(min(X[:,j]), max(X[:,j])) for j in range(m)])
    t.add_node(root)
    
    to_visit = []
    to_visit.append(0)
    next_id = 1
    while len(to_visit) < l:
        node_id = to_visit.pop(0)
        j = np.random.randint(m)
        s = np.random.uniform(*t.nodes[node_id].bounds[j])
        t.nodes[node_id].feature = j
        t.nodes[node_id].split_value = s

        children_left = Node(
            node_id=next_id, 
            parent=node_id, 
            elements=[i for i in range(n) if X[i, j] < s],
            bounds=[v if k != j else (v[0], s) for k, v in enumerate(t.nodes[node_id].bounds)]
        )
        t.add_node(children_left)
        t.nodes[node_id].children_left = next_id
        to_visit.append(next_id)
        next_id += 1

        children_right = Node(
            node_id=next_id, 
            parent=node_id, 
            elements=[i for i in range(n) if X[i, j] >= s],
            bounds=[v if k != j else (s, v[1]) for k, v in enumerate(t.nodes[node_id].bounds)]
        )
        t.add_node(children_right)
        t.nodes[node_id].children_right = next_id
        to_visit.append(next_id)
        next_id += 1
    t.leaf_nodes = sorted(to_visit)
    return t

def isolation_forest_embedding(Xnum=None, Xcat=None, n_components=300, n_trees=1000, n_leaves=32):
    if Xnum is not None and Xcat is not None:
        Xnum = Xnum.reshape((Xnum.shape[0], -1))
        Xcat = Xcat.reshape((Xcat.shape[0], -1))
        X = np.c_[Xnum, OneHotEncoder().fit_transform(Xcat).toarray()]
    elif Xnum is not None:
        Xnum = Xnum.reshape((Xnum.shape[0], -1))
        X = Xnum
    elif Xcat is not None:
        Xcat = Xcat.reshape((Xcat.shape[0], -1)) 
        X = Xcat
    X_emb = np.zeros((X.shape[0], n_trees*n_leaves))

    for k in range(n_trees):
        # Creating the tree
        t = create_random_tree(X, n_leaves)
        for j, node_id in enumerate(t.leaf_nodes):
            for i in t.get_node(node_id).elements:
                X_emb[i, k*n_leaves + j] = 1
    leaves_samples_portion = np.mean(X_emb, axis=0)
    x, _ = np.histogram(leaves_samples_portion, bins=n_components, range=[0, 1], density=True)
    return x

if __name__=="__main__":
    # from sklearn.datasets import make_blobs, make_moons
    # from scipy.stats import wasserstein_distance
    # X1, y = make_blobs(n_samples=500, centers=3, n_features=8)
    # X2, y = make_blobs(n_samples=250, centers=6, n_features=6)
    # # X2, y = make_moons(n_samples=500)
    # x1 = isolation_forest_embedding(Xnum=X1)
    # x2 = isolation_forest_embedding(Xnum=X2)
    # # print(np.mean(np.abs(x1 - x2)), wasserstein_distance(x1, x2))
    # import matplotlib.pyplot as plt
    # a = np.arange(len(x1))
    # plt.bar(a, x1, alpha=0.5, width=1)
    # plt.bar(a, x2, alpha=0.5, width=1)
    # plt.xticks([])
    # plt.show()

    ############################################################################

    # import pandas as pd
    # import openml
    # import pickle
    # from sklearn.preprocessing import minmax_scale
    # from tqdm import tqdm
    # import matplotlib.pyplot as plt
    # checked_data = pd.read_csv("output/checked_datasets.csv", sep=" ", index_col="id")
    # datasets_info = {}
    # infos_file = "output/infos_found_datasets.pickle"
    # with open(infos_file, "rb") as f:
    #     datasets_info = pickle.load(f)
    #     if checked_data is not None:
    #         for id_ in list(datasets_info.keys()):
    #             if int(id_) not in checked_data.index:
    #                 datasets_info.pop(id_)

    # print("LOADING...")
    # datasets = {}
    # n_loaded = 0
    # for id_ in tqdm(list(datasets_info.keys())[10:16]):
    #     try:
    #         dataset = openml.datasets.get_dataset(id_)
    #         X, y, categorical_indicator, attribute_names = dataset.get_data(
    #             target=dataset.default_target_attribute
    #         )
    #         k = 0
    #         Xnum = X.loc[datasets_info[id_][k]["samples"], datasets_info[id_][k]["num_columns"]]
    #         Xcat = X.loc[datasets_info[id_][k]["samples"], datasets_info[id_][k]["cat_columns"]]
    #         for col in Xcat.columns:
    #             Xcat.loc[:, col] = pd.Categorical(Xcat.loc[:, col]).codes
    #         Xnum = Xnum.to_numpy()
    #         Xnum = minmax_scale(Xnum)
    #         Xcat = Xcat.to_numpy()
            
    #         new_y = y.loc[datasets_info[id_][k]["samples"]]
    #         new_y = pd.Categorical(new_y).codes if new_y.dtype.name == 'category' else new_y
    #         # print(f"samples({new_X.shape[0]}), num({Xnum.shape[1]}), cat({Xcat.shape[1]})...")
    #         datasets[id_] = [Xnum, Xcat, new_y]
    #         n_loaded += 1
    #     except:
    #         print(f"Not able to load data set with id {id_}")

    # print("END LOADING!")
    # print()
    # rows, cols = 2, 3
    # i = 1
    # for id_, data in datasets.items():
    #     plt.subplot(rows, cols, i)
    #     Xnum, Xcat, y = data
    #     x1 = isolation_forest_embedding(Xnum=Xnum, Xcat=Xcat)
    #     x2 = isolation_forest_embedding(Xnum=Xnum, Xcat=Xcat)

    #     a = np.arange(len(x1))
    #     plt.bar(a, x1, alpha=0.5, width=1)
    #     plt.bar(a, x2, alpha=0.5, width=1)
    #     plt.xticks([])
    #     plt.title(id_)
    #     i += 1
    # plt.tight_layout()
    # plt.show()

    #######################################################################

    import argparse
    import pickle
    from joblib import Parallel, delayed

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import minmax_scale
    import time
    import os
    import json
    
    def compute_meta_features(data):
        Xnum, Xcat, y = minmax_scale(data["Xnum"]), data["Xcat"], data["y"]
        start = time.time()
        x = isolation_forest_embedding(Xnum=Xnum, Xcat=Xcat)
        end = time.time()
        return x, end - start
    
    parser = argparse.ArgumentParser(description='Compute meta-features')
    parser.add_argument("-d", "--datasetsdir", default=None)
    parser.add_argument("-o", "--outputdir",
                        help="Path to the output directory")
    parser.add_argument(
        "-j", "--jobs", help="The number of concurent workers", default=-1)
    args = parser.parse_args()

    OUTPUT_FILE = os.path.join(args.outputdir, "if_meta_features.csv")
    TIME_FILE = os.path.join(args.outputdir, "if_meta_features_times.json")

    meta_df = None
    if os.path.isfile(OUTPUT_FILE):
        meta_df = pd.read_csv(OUTPUT_FILE, index_col="id")
        meta_df.index = meta_df.index.astype(str)
    times = {}
    if os.path.isfile(TIME_FILE):
        with open(TIME_FILE, "r", encoding="utf-8") as f:
            times = json.load(f)

    filenames = []
    if args.datasetsdir is not None:
        filenames += [os.path.join(args.datasetsdir, filename)
                        for filename in os.listdir(args.datasetsdir)
                        if filename.split('.')[0] not in (meta_df.index if meta_df is not None else [])]

    datasets = []
    for filename in filenames:
        with open(filename, "rb") as f:
            data = pickle.load(f)
            data["id"] = filename.split("/")[-1].split(".")[0]
        datasets.append(data)

    print(f"{len(datasets)} to handle")
    print("Computing meta-features...")
    start = time.time()
    list_ret = Parallel(n_jobs=int(args.jobs), verbose=60)(
        delayed(compute_meta_features)(data) for data in datasets
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
    attribute_names = np.arange(meta_X.shape[1])

    new_meta_df = pd.DataFrame(columns=attribute_names, data=meta_X, index=ids)
    meta_df = new_meta_df if meta_df is None else pd.concat([
        meta_df,
        new_meta_df
    ])
    meta_df.index.name = "id"
    print(meta_df.head())
    meta_df.to_csv(OUTPUT_FILE, index="id")
    with open(TIME_FILE, "w", encoding="utf-8") as f:
        json.dump(times, f, indent=4, ensure_ascii=False)
    print("DONE")
