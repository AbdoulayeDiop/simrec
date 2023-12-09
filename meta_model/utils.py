import numpy as np

def _ndcg(y1, y2, p=None):
    y1 = np.array(y1)
    y2 = np.array(y2)
    sorted_ind1 = np.argsort(-y1)
    sorted_ind2 = np.argsort(-y2)
    new_p = sum(y1 > 0)
    if p is not None:
        new_p = min(p, new_p)
    idcg = np.sum((y1[sorted_ind1[:new_p]])/np.log2(np.arange(2, new_p+2)))
    ind = [k for k in sorted_ind2 if y1[k] > 0]
    dcg = np.sum(y1[ind[:new_p]]/np.log2(np.arange(2, new_p+2)))
    return dcg/idcg

def ndcg(y_true, y_pred, p=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if len(y_true.shape) <= 1:
        return _ndcg(y_true, y_pred, p)
    return np.array([_ndcg(y, y_pred[i]) for i, y in enumerate(y_true)])

def ndcg_sim(y1, y2, p=None):
    return _ndcg(y1, y2, p) * _ndcg(y2, y1, p) #pylint: disable=arguments-out-of-order

def custom_sim(y1, y2, threshold=0.95):
    set1 = set([j for j, yj in enumerate(y1) if yj/max(y1) > threshold])
    set2 = set([j for j, yj in enumerate(y2) if yj/max(y2) > threshold])
    return len(set1.intersection(set2)) / len(set1.union(set2)) #pylint: disable=arguments-out-of-order
