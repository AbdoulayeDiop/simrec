import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import confusion_matrix, adjusted_rand_score as ari, silhouette_score
from sklearn.metrics.cluster import contingency_matrix


EXTERNAL_EVAL_METRICS = ["acc", "ari", "purity"]
INTERNAL_EVAL_METRICS = ["sil"]


def accuracy(labels, predicted_labels):
    cm = confusion_matrix(labels, predicted_labels)

    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)
    rows, cols = linear_assignment(_make_cost_m(cm))
    indexes = zip(rows, cols)
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    return np.trace(cm2) / np.sum(cm2)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    cm = contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)


def get_score(labels, predicted_labels, eval_metric="acc"):
    if eval_metric == "acc":
        return accuracy(labels, predicted_labels)
    elif eval_metric == "ari":
        return ari(labels, predicted_labels)
    elif eval_metric == "purity":
        return purity_score(labels, predicted_labels)


def get_unsupervised_score(X, predicted_labels, eval_metric="sil", **kwargs):
    if eval_metric == "sil":
        try:
            return silhouette_score(X, predicted_labels, **kwargs)
        except:
            return -1
