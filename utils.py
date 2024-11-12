from metrics import base_metrics
from sklearn.preprocessing import OneHotEncoder

def get_valid_similarity_measures(X, data_type="numeric"):
    l = []
    for metric in base_metrics.get_available_metrics(data_type=data_type):
        m = base_metrics.get_metric(metric)
        if m.is_valid_data(X):
            ok = True
            if metric=="mahalanobis":
                try:
                    m.fit(X)
                except:
                    ok = False
            if ok:
                l.append(metric)
    return l

def get_valid_similarity_pairs(Xnum, Xcat):
    enc = OneHotEncoder(handle_unknown='ignore')
    Xdummy = enc.fit_transform(Xcat).toarray()
    l = []
    for num_metric in get_valid_similarity_measures(Xnum, data_type="numeric"):
        for cat_metric in get_valid_similarity_measures(Xcat, data_type="categorical"):
            l.append(f"{num_metric}_{cat_metric}")
        for bin_metric in get_valid_similarity_measures(Xdummy, data_type="binary"):
            l.append(f"{num_metric}_{bin_metric}")
    return l