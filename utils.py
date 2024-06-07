from metrics import base_metrics
from sklearn.preprocessing import OneHotEncoder

def get_valid_similarity_pairs(Xnum, Xcat):
    enc = OneHotEncoder(handle_unknown='ignore')
    Xdummy = enc.fit_transform(Xcat).toarray()
    l = []
    for num_metric in base_metrics.get_available_metrics():
        m = base_metrics.get_metric(num_metric)
        if m.is_valid_data(Xnum):
            ok = True
            if num_metric=="mahalanobis":
                try:
                    m.fit(Xnum)
                except:
                    ok = False
            if ok:
                for cat_metric in base_metrics.get_available_metrics(data_type="categorical"):
                    if base_metrics.get_metric(cat_metric).is_valid_data(Xcat):
                        l.append(f"{num_metric}_{cat_metric}")
                for bin_metric in base_metrics.get_available_metrics(data_type="binary"):
                    if base_metrics.get_metric(bin_metric).is_valid_data(Xdummy):
                        l.append(f"{num_metric}_{bin_metric}")
    return l