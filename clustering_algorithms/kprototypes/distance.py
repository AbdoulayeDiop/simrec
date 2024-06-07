import numpy as np


def check_distance(distance):
    """Resolve distance function.

    If ``distance`` is a string, only ``"euclidean"``, ``"manhattan"`` and
    ``"matching"`` are accepted. If it is a callable, then is is return as-is.
    If it is ``None``, it defaults to ``"euclidean"``.

    See Also
    --------
    :meth:`euclidean_distance`, :meth:`manhattan_distance`,
    :meth:`matching_distance`

    """

    if distance is None:
        return euclidean_distance
    if callable(distance):
        return distance
    if distance == "euclidean":
        return euclidean_distance
    if distance == "manhattan":
        return manhattan_distance
    if distance == "matching":
        return matching_distance
    raise KeyError(distance)


def euclidean_distance(a, b):
    """Squared euclidean distance.

    This is the sum of squared differences for each feature pair, also known as
    squared L\\ :sub:`2` norm.

    Example
    --------
    >>> a = np.array([0.0, 0.0, 0.0])
    >>> b = np.array([1.0, 2.0, 3.0])
    >>> euclidean_distance(a, b)
    14.0

    """

    return np.sum((a - b) ** 2, axis=-1)


def manhattan_distance(a, b):
    """Manhattan distance.

    This is the sum of absolute differences for each feature pair, also known
    as L\\ :sub:`1` norm.

    Example
    --------
    >>> a = np.array([0.0, 0.0, 0.0])
    >>> b = np.array([1.0, 2.0, 3.0])
    >>> manhattan_distance(a, b)
    6.0

    """

    return np.abs(a - b).sum(axis=-1)


def matching_distance(a, b):
    """Matching distance.

    Each feature pair that does not match adds one to the distance. This
    distance measure is often used for categorical features.

    Example
    --------
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> b = np.array([1, 8, 3, 4, 0])
    >>> matching_distance(a, b)
    2

    """

    return np.sum(a != b, axis=-1)


# TODO Jaccard distance
